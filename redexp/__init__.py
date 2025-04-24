from gymnasium.envs.registration import register

register(
    id="Safe-Dubins3d-NoModelMismatch-v1",
    entry_point="redexp.envs:Dubins3dEnv",
    max_episode_steps=200,
    kwargs={"car": "dubins_3d_omega_0_5", "brt": "dubins_3d_omega_0_5"},
)

register(
    id="Safe-Dubins3d-BadModelMismatch-v1",
    entry_point="redexp.envs:Dubins3dEnv",
    max_episode_steps=200,
    kwargs={"car": "dubins_3d_omega_0_5", "brt": "dubins_3d_omega_0_75"},
)

register(
    id="Safe-Dubins3d-GoodModelMismatch-v1",
    entry_point="redexp.envs:Dubins3dEnv",
    max_episode_steps=200,
    kwargs={"car": "dubins_3d_omega_0_75", "brt": "dubins_3d_omega_0_5"},
)

register(
    id="TurtlebotEnv-ModelMismatch-v1",
    entry_point="redexp.envs:TurtlebotEnv",
    max_episode_steps=1000,
    kwargs={"model_mismatch": True},
)
