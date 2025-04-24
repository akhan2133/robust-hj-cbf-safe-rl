try:
    import heterocl as hcl
except ImportError:
    print("HeteroCL not installed. Some functions may throw errors.")
import numpy as np


class DubinsCar:
    def __init__(
        self,
        wMax=1.5,
        speed=1,
        r=0.2,
        x=np.array([0, 0, 0]),
        dMin=np.array([0, 0, 0]),
        dMax=np.array([0, 0, 0]),
        uMode="min",
        dMode="max",
    ):
        self.x = x
        self.wMax = wMax
        self.speed = speed
        self.dMin = dMin
        self.dMax = dMax
        self.uMode = uMode
        self.dMode = dMode

        self.r = r

    def opt_ctrl(self, t, state, spat_deriv):
        opt_w = hcl.scalar(self.wMax, "opt_w")
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")
        with hcl.if_(spat_deriv[2] > 0):
            with hcl.if_(self.uMode == "min"):
                opt_w[0] = -opt_w
        with hcl.elif_(spat_deriv[2] < 0):
            with hcl.if_(self.uMode == "max"):
                opt_w[0] = -opt_w
        return (opt_w[0], in3[0], in4[0])

    def opt_dstb(self, t, state, spat_deriv):
        d1 = hcl.scalar(self.dMax[0], "d1")
        d2 = hcl.scalar(self.dMax[1], "d2")
        d3 = hcl.scalar(self.dMax[2], "d3")
        with hcl.if_(self.dMode == "max"):
            with hcl.if_(spat_deriv[0] >= 0):
                d1[0] = self.dMax[0]
            with hcl.else_():
                d1[0] = self.dMin[0]
            with hcl.if_(spat_deriv[1] >= 0):
                d2[0] = self.dMax[1]
            with hcl.else_():
                d2[0] = self.dMin[1]
            with hcl.if_(spat_deriv[2] >= 0):
                d3[0] = self.dMax[2]
            with hcl.else_():
                d3[0] = self.dMin[2]
        with hcl.else_():
            with hcl.if_(spat_deriv[0] >= 0):
                d1[0] = self.dMin[0]
            with hcl.else_():
                d1[0] = self.dMax[0]
            with hcl.if_(spat_deriv[1] >= 0):
                d2[0] = self.dMin[1]
            with hcl.else_():
                d2[0] = self.dMax[1]
            with hcl.if_(spat_deriv[2] >= 0):
                d3[0] = self.dMin[2]
            with hcl.else_():
                d3[0] = self.dMax[2]
        return d1, d2, d3

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        x_dot[0] = self.speed * hcl.cos(state[2]) + dOpt[0]
        y_dot[0] = self.speed * hcl.sin(state[2]) + dOpt[1]
        theta_dot[0] = uOpt[0] + dOpt[2]

        return (x_dot[0], y_dot[0], theta_dot[0])

    def opt_ctrl_non_hcl(self, t, state, spat_deriv):
        opt_w = None
        if spat_deriv[2] >= 0:
            if self.uMode == "max":
                opt_w = self.wMax
            else:
                opt_w = -self.wMax
        elif spat_deriv[2] <= 0:
            if self.uMode == "max":
                opt_w = -self.wMax
            else:
                opt_w = self.wMax

        return np.array([opt_w])

    def opt_dist_non_hcl(self, t, state, spat_deriv):
        d_opt = np.zeros(3)
        if self.dMode == "max":
            for i in range(3):
                if spat_deriv[i] >= 0:
                    d_opt[i] = self.dMax[i]
                else:
                    d_opt[i] = self.dMin[i]
        elif self.dMode == "min":
            for i in range(3):
                if spat_deriv[i] >= 0:
                    d_opt[i] = self.dMin[i]
                else:
                    d_opt[i] = self.dMax[i]
        return d_opt

    def dynamics_non_hcl(self, t, state, u_opt, disturbance=np.zeros(3)):
        if not isinstance(u_opt, np.ndarray):
            u_opt = np.array([u_opt])
        x_dot = self.speed * np.cos(state[2]) + disturbance[0]
        y_dot = self.speed * np.sin(state[2]) + disturbance[1]
        theta_dot = u_opt[0] + disturbance[2]
        return np.array([x_dot, y_dot, theta_dot], dtype=np.float32)

    def ham(self, state, u_opt, spat_deriv, disturbance=np.zeros(3)):
        x_dot = self.speed * np.cos(state[2]) + disturbance[0]
        y_dot = self.speed * np.sin(state[2]) + disturbance[1]
        # theta_dot = self.wMax + disturbance[2]
        theta_dot = u_opt + disturbance[2]

        dyn = np.array([x_dot, y_dot, theta_dot])
        # if self.uMode == 'max':
        #     spat_deriv[2] = np.abs(spat_deriv[2])
        # else:
        #     spat_deriv[2] = np.abs(spat_deriv[2]) * -1

        return dyn @ spat_deriv


class DubinsCar4D:
    def __init__(
        self,
        x=[0, 0, 0, 0],
        uMin=[-1.5, -np.pi / 8],  # 0.39
        uMax=[1.5, np.pi / 8],
        dMin=[0.0, 0.0, 0.0, 0.0],
        dMax=[0.0, 0.0, 0.0, 0.0],
        uMode="max",
        dMode="min",
        r=0.2,
    ):
        """Creates a Dublin Car with the following states:
           X position, Y position, acceleration, heading

           The first element of user control and disturbance is acceleration
           The second element of user control and disturbance is heading


        Args:
            x (list, optional): Initial state . Defaults to [0,0,0,0].
            uMin (list, optional): Lowerbound of user control. Defaults to [-1,-1].
            uMax (list, optional): Upperbound of user control.
                                   Defaults to [1,1].
            dMin (list, optional): Lowerbound of disturbance to user control, . Defaults to [-0.25,-0.25].
            dMax (list, optional): Upperbound of disturbance to user control. Defaults to [0.25,0.25].
            uMode (str, optional): Accepts either "min" or "max".
                                   * "min" : have optimal control reach goal
                                   * "max" : have optimal control avoid goal
                                   Defaults to "min".
            dMode (str, optional): Accepts whether "min" or "max" and should be opposite of uMode.
                                   Defaults to "max".
        """
        self.x = x
        self.uMax = np.array(uMax)
        self.uMin = np.array(uMin)
        self.dMax = dMax
        self.dMin = dMin
        assert uMode in ["min", "max"]
        self.uMode = uMode
        if uMode == "min":
            assert dMode == "max"
        else:
            assert dMode == "min"
        self.dMode = dMode
        self.r = r

    def opt_ctrl(self, t, state, spat_deriv):
        """
        :param t: time t
        :param state: tuple of coordinates
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return:
        """
        # System dynamics
        # x_dot     = v * cos(theta) + d_1
        # y_dot     = v * sin(theta) + d_2
        # v_dot = a
        # theta_dot = v * tan(delta) / L

        # Graph takes in 4 possible inputs, by default, for now
        opt_a = hcl.scalar(self.uMax[0], "opt_a")
        opt_w = hcl.scalar(self.uMax[1], "opt_w")
        # Just create and pass back, even though they're not used
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")

        with hcl.if_(self.uMode == "min"):
            with hcl.if_(spat_deriv[2] > 0):
                opt_a[0] = self.uMin[0]
            with hcl.if_(spat_deriv[3] > 0):
                opt_w[0] = self.uMin[1]
        with hcl.else_():  # uMode == max
            with hcl.if_(spat_deriv[2] < 0):
                opt_a[0] = self.uMin[0]
            with hcl.if_(spat_deriv[3] < 0):
                opt_w[0] = self.uMin[1]
        # return 3, 4 even if you don't use them
        return (opt_a[0], opt_w[0], in3[0], in4[0])

    def opt_ctrl_non_hcl(self, t, state, spat_deriv):
        if self.uMode == "min":
            if spat_deriv[2] > 0:
                opt_a = self.uMin[0]
            else:
                opt_a = self.uMax[0]

            if spat_deriv[3] > 0:
                opt_w = self.uMin[1]
            else:
                opt_w = self.uMax[1]
        else:
            if spat_deriv[2] < 0:
                opt_a = self.uMin[0]
            else:
                opt_a = self.uMax[0]

            if spat_deriv[3] < 0:
                opt_w = self.uMin[1]
            else:
                opt_w = self.uMax[1]

        return np.array([opt_a, opt_w])

    def opt_dstb(self, t, state, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal disturbances
        """
        # Graph takes in 4 possible inputs, by default, for now
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        # Just create and pass back, even though they're not used
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")

        with hcl.if_(self.dMode == "max"):
            with hcl.if_(spat_deriv[0] > 0):
                d1[0] = self.dMax[0]
            with hcl.elif_(spat_deriv[0] < 0):
                d1[0] = self.dMin[0]
            with hcl.if_(spat_deriv[1] > 0):
                d2[0] = self.dMax[1]
            with hcl.elif_(spat_deriv[1] < 0):
                d2[0] = self.dMin[1]
            with hcl.if_(spat_deriv[2] >= 0):
                d3[0] = self.dMax[2]
            with hcl.else_():
                d3[0] = self.dMin[2]
            with hcl.if_(spat_deriv[3] >= 0):
                d4[0] = self.dMax[3]
            with hcl.else_():
                d4[0] = self.dMin[3]
        with hcl.else_():
            with hcl.if_(spat_deriv[0] > 0):
                d1[0] = self.dMin[0]
            with hcl.elif_(spat_deriv[0] < 0):
                d1[0] = self.dMax[0]

            with hcl.if_(spat_deriv[1] > 0):
                d2[0] = self.dMin[1]
            with hcl.elif_(spat_deriv[1] < 0):
                d2[0] = self.dMax[1]

            with hcl.if_(spat_deriv[2] > 0):
                d3[0] = self.dMin[2]
            with hcl.elif_(spat_deriv[2] < 0):
                d3[0] = self.dMax[2]

            with hcl.if_(spat_deriv[3] > 0):
                d4[0] = self.dMin[3]
            with hcl.elif_(spat_deriv[3] < 0):
                d4[0] = self.dMax[3]
        return (d1[0], d2[0], d3[0], d4[0])

    def opt_dstb_non_hcl(self, t, state, spat_deriv):
        d_opt = np.zeros(4)
        if self.dMode == "max":
            for i in range(4):
                if spat_deriv[i] >= 0:
                    d_opt[i] = self.dMax[i]
                else:
                    d_opt[i] = self.dMin[i]
        elif self.dMode == "min":
            for i in range(4):
                if spat_deriv[i] >= 0:
                    d_opt[i] = self.dMin[i]
                else:
                    d_opt[i] = self.dMax[i]
        return d_opt

    def dynamics(self, t, state, uOpt, dOpt):
        # wheelbase of Tamiya TT02
        L = hcl.scalar(self.r, "L")

        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        v_dot = hcl.scalar(0, "v_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        x_dot[0] = state[2] * hcl.cos(state[3]) + dOpt[0]
        y_dot[0] = state[2] * hcl.sin(state[3]) + dOpt[1]
        v_dot[0] = uOpt[0] + dOpt[2]
        theta_dot[0] = state[2] * (hcl.sin(uOpt[1]) / hcl.cos(uOpt[1])) / L[0] + dOpt[3]

        return (x_dot[0], y_dot[0], v_dot[0], theta_dot[0])

    def dynamics_non_hcl(self, t, state, uOpt, dOpt=np.zeros(4)):
        # wheelbase of Tamiya TT02
        x_dot = state[2] * np.cos(state[3]) + dOpt[0]
        y_dot = state[2] * np.sin(state[3]) + dOpt[1]
        v_dot = uOpt[0] + dOpt[2]
        theta_dot = state[2] * np.tan(uOpt[1]) / self.r + dOpt[3]

        return np.array([x_dot, y_dot, v_dot, theta_dot], dtype=np.float32)

    def ham(self, state, uOpt, spat_deriv, dOpt=np.zeros(4)):
        x_dot = state[2] * np.cos(state[3]) + dOpt[0]
        y_dot = state[2] * np.sin(state[3]) + dOpt[1]
        v_dot = uOpt[0] + dOpt[2]
        theta_dot = state[2] * np.tan(uOpt[1]) / self.r + dOpt[3]

        dyn = np.array([x_dot, y_dot, v_dot, theta_dot])
        return dyn @ spat_deriv
