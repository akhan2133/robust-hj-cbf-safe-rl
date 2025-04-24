import numpy as np
import cvxpy as cp
import gymnasium as gym

from redexp.utils import spa_deriv, V_spa_deriv_at_state


class HJCBFQPSolver:
    def __init__(self, grid, brt, dyn, gamma, maximum_slack, env_type="dubins_3d"):
        self.grid = grid
        self.brt = brt
        self.dyn = dyn

        self.env_type = env_type

        def hamiltonian(state, control, nabla_V):
            v = self.dyn.speed
            x_dot = v * np.cos(state[2])
            y_dot = v * np.sin(state[2])
            theta_dot = control[0]

            dynamics = np.array([x_dot, y_dot, theta_dot])

            return nabla_V @ dynamics

        self.hamiltonian_fn = hamiltonian

        u_dim = 1
        self.gamma = gamma
        self.slack_penalty = 1e7
        self.maximum_slack = maximum_slack

        self.filtered_control = cp.Variable(u_dim)  # control we are solving for
        self.safe_control_cp = cp.Parameter(u_dim)  # contrl we are going to use
        self.slack_cp = cp.Variable(u_dim, pos=True)

    def project_ctrl(self, state):
        obj = cp.Minimize(cp.norm(self.filtered_control - self.safe_control_cp))
        contraints = []
        # Eq. 28, Borquez, Javier, et al. "On safety and liveness filtering using hamilton-jacobi reachability analysis." IEEE Transactions on Robotics (2024).
        V_at_state, nabla_V_at_state = V_spa_deriv_at_state(state, self.brt, self.grid)
        contraints.append(
            self.hamiltonian_fn(state, self.filtered_control, nabla_V_at_state)
            >= -self.gamma * V_at_state
        )
        safest_control = self.dyn.opt_ctrl_non_hcl(0, state, nabla_V_at_state)
        contraints.append(self.filtered_control <= self.dyn.wMax)
        contraints.append(self.filtered_control >= -self.dyn.wMax)

        self.safe_control_cp.value = safest_control
        QP = cp.Problem(obj, contraints)

        result = QP.solve(solver=cp.ECOS)
        if result != np.inf:
            safe_control = self.filtered_control.value
            return safe_control
        obj_relaxed = cp.Minimize(
            cp.norm(self.filtered_control - self.safe_control_cp)
            + self.slack_penalty * cp.norm(self.slack_cp, 1)
        )
        contraints = []
        contraints.append(self.filtered_control <= self.dyn.wMax)
        contraints.append(self.filtered_control >= -self.dyn.wMax)
        contraints.append(
            self.hamiltonian_fn(state, self.filtered_control, nabla_V_at_state)
            >= -self.gamma * V_at_state - self.slack_cp
        )
        QP = cp.Problem(obj_relaxed, contraints)
        self.safe_control_cp.value = safest_control
        try:
            result = QP.solve(solver=cp.ECOS)
            slack = self.slack_cp.value
            self.maximum_slack = max(slack, self.maximum_slack)
        except Exception as e:
            print("WARN: nominal control returned for relaxed QP")
            return safest_control

        safe_control = self.filtered_control.value
        return safe_control


class HJCBFSafetyFilter(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym.Env,
        env_type="dubins_3d",
        cbf_gamma=1.0,
        cbf_init_max_slack=0.1,
    ):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)

        self.env_type = env_type
        if env_type == "dubins_3d":
            dyn = self.unwrapped.car
            brt = self.unwrapped.brt
            grid = self.unwrapped.grid
        elif env_type == "turtlebot":
            dyn = self.unwrapped.turtlebot.dyn
            brt = self.unwrapped.turtlebot.brt
            grid = self.unwrapped.turtlebot.grid

        self.hj_cbf_qp_solver = HJCBFQPSolver(
            grid, brt, dyn, cbf_gamma, cbf_init_max_slack, env_type
        )

    def step(self, action):
        filter_info = {"used_filter": False}
        state = self.unwrapped.state
        value = self._value(state)
        filter_info = {"used_filter": False}

        if value < self.hj_cbf_qp_solver.maximum_slack:
            safe_action = self._cbf_action(state)
            filter_info["used_filter"] = True
            filter_info["maximum_slack"] = self.hj_cbf_qp_solver.maximum_slack
            action = safe_action
        (
            observation,
            reward,
            termination,
            truncation,
            info,
        ) = self.env.step(action)
        info = {**info, **filter_info}
        return observation, reward, termination, truncation, info

    def _value(self, state):
        if self.env_type == "deepreach":
            value, _, _, _ = self.hj_cbf_qp_solver.dr.value_and_opt_ctrl(state)
            return value
        else:
            return self.hj_cbf_qp_solver.grid.get_value(
                self.hj_cbf_qp_solver.brt, state
            )

    def _cbf_action(self, state):
        safest_action = self.hj_cbf_qp_solver.project_ctrl(state)
        return safest_action


class LeastRestrictiveControlSafetyFilter(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym.Env,
        env_type,
        eps,
    ):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)

        if env_type == "dubins_3d":
            self.dyn = self.unwrapped.car
            self.brt = self.unwrapped.brt
            self.grid = self.unwrapped.grid
        elif env_type == "turtle_bot":
            self.dyn = self.unwrapped.turtlebot.dyn
            self.brt = self.unwrapped.turtlebot.brt
            self.grid = self.unwrapped.turtlebot.grid
        self.eps = eps

    def step(self, action):
        filter_info = {"used_filter": False}

        state = self.unwrapped.state
        if self._value(state) < self.eps:
            safe_action = self._safe_action(state, action)
            filter_info["used_filter"] = True
            action = safe_action

        (
            observation,
            reward,
            termination,
            truncation,
            info,
        ) = self.env.step(action)
        info = {**info, **filter_info}
        return observation, reward, termination, truncation, info

    def _value(self, state):
        return self.grid.get_value(self.brt, state)

    def _safe_action(self, state, action):
        index = self.grid.get_index(state)
        nabla_V = spa_deriv(index, self.brt, self.grid)
        safest_action = self.dyn.opt_ctrl_non_hcl(0, state, nabla_V)
        return safest_action
