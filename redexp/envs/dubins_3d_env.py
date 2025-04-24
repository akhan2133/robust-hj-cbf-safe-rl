import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

from redexp.config.dubins_3d import OBSTACLE_RADIUS
from redexp.utils import normalize_angle
from redexp.brts.dubins_3d import (
    dubins_3d_omega_0_25,
    dubins_3d_omega_0_5,
    dubins_3d_omega_0_75,
    grid,
)


class Dubins3dEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render.modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        render_mode=None,
        car="dubins_3d_omega_0_5",
        brt="dubins_3d_omega_0_5",
        goal_conditioned=False,
    ):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.render_mode = render_mode

        assert car in [
            "dubins_3d_omega_0_25",
            "dubins_3d_omega_0_5",
            "dubins_3d_omega_0_75",
        ], "unknown car"

        if car == "dubins_3d_omega_0_25":
            self.car = dubins_3d_omega_0_25
            self.true_brt = np.load("./redexp/brts/dubins_3d_omega_0_25_brt.npy")
        elif car == "dubins_3d_omega_0_5":
            self.car = dubins_3d_omega_0_5
            self.true_brt = np.load("./redexp/brts/dubins_3d_omega_0_5_brt.npy")
        elif car == "dubins_3d_omega_0_75":
            self.car = dubins_3d_omega_0_75
            self.true_brt = np.load("./redexp/brts/dubins_3d_omega_0_75_brt.npy")

        assert brt in [
            "dubins_3d_omega_0_25",
            "dubins_3d_omega_0_5",
            "dubins_3d_omega_0_75",
        ], "unknown brt"

        if brt == "dubins_3d_omega_0_25":
            self.brt = np.load("./redexp/brts/dubins_3d_omega_0_25_brt.npy")
        elif brt == "dubins_3d_omega_0_5":
            self.brt = np.load("./redexp/brts/dubins_3d_omega_0_5_brt.npy")
        elif brt == "dubins_3d_omega_0_75":
            self.brt = np.load("./redexp/brts/dubins_3d_omega_0_75_brt.npy")

        self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.dt = 0.05

        self.action_space = gym.spaces.Box(
            low=np.array([-self.car.wMax], dtype=np.float32),
            high=np.array([self.car.wMax], dtype=np.float32),
            dtype=np.float32,
        )

        self.goal_conditioned = goal_conditioned
        if self.goal_conditioned:
            self.observation_space = gym.spaces.Box(
                low=np.array([-4, -4, -1.0, -1.0, -4, -4], dtype=np.float32),
                high=np.array([4, 4, 1.0, 1.0, 4, 4], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=np.array([-4, -4, -1.0, -1.0], dtype=np.float32),
                high=np.array([4, 4, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

        self.goal_location = np.array([2.0, 2.0], dtype=np.float32)
        self.goal_r = 0.3
        self.obstacle_location = np.array([0.0, 0.0], dtype=np.float32)
        self.obstacle_r = OBSTACLE_RADIUS
        self.grid = grid

    def reset(self, seed=None, options={}):
        self.state = np.array([-2.0, -2.0, np.pi / 4], dtype=np.float32)

        self.traj_min_brt_value = self.grid.get_value(self.true_brt, self.state)
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {"cost": 0}

    def step(self, action):
        self.state = (
            self.car.dynamics_non_hcl(0, self.state, action) * self.dt + self.state
        )
        self.state[2] = normalize_angle(self.state[2])

        info = {}

        obs = self._get_obs()
        reward = -np.linalg.norm(self.state[:2] - self.goal_location)

        terminated = False

        cost = np.linalg.norm(self.state[:2] - self.obstacle_location) < (
            self.obstacle_r + self.car.r
        )
        if cost > 0:
            terminated = True
        info["cost"] = cost

        if np.linalg.norm(self.state[:2] - self.goal_location) < (
            self.goal_r + self.car.r
        ):
            terminated = True
            info["reach_goal"] = True
        info["reach_goal"] = False

        self.traj_min_brt_value = min(
            self.traj_min_brt_value, self.grid.get_value(self.true_brt, self.state)
        )
        info["traj_min_brt_value"] = self.traj_min_brt_value

        return obs, reward, terminated, False, info

    def _get_obs(self):
        if self.goal_conditioned:
            return np.array(
                [
                    self.state[0],
                    self.state[1],
                    np.cos(self.state[2]),
                    np.sin(self.state[2]),
                    self.goal_location[0],
                    self.goal_location[1],
                ]
            )
        else:
            return np.array(
                [
                    self.state[0],
                    self.state[1],
                    np.cos(self.state[2]),
                    np.sin(self.state[2]),
                ]
            )

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        self._render_frame()
        self.fig.canvas.flush_events()
        plt.pause(1 / self.metadata["render_fps"])

    def _render_frame(self):
        self.ax.clear()

        def add_circle(state, r, color="green"):
            self.ax.add_patch(plt.Circle(state[:2], radius=r, color=color))

            dir = state[:2] + r * np.array([np.cos(state[2]), np.sin(state[2])])

            self.ax.plot([state[0], dir[0]], [state[1], dir[1]], color="c")

        add_circle(self.state, self.car.r, color="blue")
        goal = plt.Circle(self.goal_location[:2], radius=self.goal_r, color="g")
        self.ax.add_patch(goal)
        obstacle = plt.Circle(self.obstacle_location, radius=self.obstacle_r, color="r")
        self.ax.add_patch(obstacle)

        X, Y = np.meshgrid(
            np.linspace(self.grid.min[0], self.grid.max[0], self.grid.pts_each_dim[0]),
            np.linspace(self.grid.min[1], self.grid.max[1], self.grid.pts_each_dim[1]),
            indexing="ij",
        )

        index = self.grid.get_index(self.state)

        self.ax.contour(
            X,
            Y,
            self.true_brt[:, :, index[2]],
            # self.brt[:, :, index[2]],
            levels=[0.1],
        )
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        return img

    def close(self):
        plt.close()
