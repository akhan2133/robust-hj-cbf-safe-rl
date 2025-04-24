from odp.Grid.GridProcessing import Grid
from redexp.dynamics.dubins_car import DubinsCar
import numpy as np
from redexp.config.dubins_3d import (
    SPEED,
    RADIUS,
    OMEGA_BAD_MODEL_MISMATCH,
    OMEGA_NO_MODEL_MISMATCH,
    OMEGA_GOOD_MODEL_MISMATCH,
    OBSTACLE_RADIUS,
)

grid = Grid(
    np.array([-4, -4, -np.pi]),
    np.array([4, 4, np.pi]),
    3,
    np.array([201, 201, 201]),
    [2],
)
dubins_3d_omega_0_25 = DubinsCar(
    r=RADIUS, uMode="max", dMode="min", speed=SPEED, wMax=OMEGA_BAD_MODEL_MISMATCH 
)
dubins_3d_omega_0_5 = DubinsCar(
    r=RADIUS, uMode="max", dMode="min", speed=SPEED, wMax=OMEGA_NO_MODEL_MISMATCH
)
dubins_3d_omega_0_75 = DubinsCar(
    r=RADIUS, uMode="max", dMode="min", speed=SPEED, wMax=OMEGA_GOOD_MODEL_MISMATCH
)

cylinder_r = RADIUS + OBSTACLE_RADIUS

if __name__ in "__main__":
    from odp.Shapes.ShapesFunctions import *
    from odp.Plots.plot_options import *
    from odp.solver import HJSolver

    ivf = CylinderShape(grid, [2], np.zeros(3), cylinder_r)

    lookback_length = 20.0
    t_step = 0.05
    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

    compMethods = {"TargetSetMode": "minVWithV0"}

    result = HJSolver(
        dubins_3d_omega_0_25,
        grid,
        ivf,
        tau,
        compMethods,
        PlotOptions(do_plot=True, plot_type="set", plotDims=[0, 1, 2], slicesCut=[]),
        saveAllTimeSteps=False,
        untilConvergent=True,
        epsilon=0.000005,
    )
    np.save("./redexp/brts/dubins_3d_omega_0_25_brt.npy", result)

    result = HJSolver(
        dubins_3d_omega_0_5,
        grid,
        ivf,
        tau,
        compMethods,
        PlotOptions(do_plot=True, plot_type="set", plotDims=[0, 1, 2], slicesCut=[]),
        saveAllTimeSteps=False,
        untilConvergent=True,
        epsilon=0.000005,
    )

    np.save("./redexp/brts/dubins_3d_omega_0_5_brt.npy", result)

    result = HJSolver(
        dubins_3d_omega_0_75,
        grid,
        ivf,
        tau,
        compMethods,
        PlotOptions(do_plot=True, plot_type="set", plotDims=[0, 1, 2], slicesCut=[]),
        saveAllTimeSteps=False,
        untilConvergent=True,
        epsilon=0.000005,
    )

    np.save("./redexp/brts/dubins_3d_omega_0_75_brt.npy", result)
