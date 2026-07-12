import gymnasium as gym

gym.register(
    id="IsaacRLEnv-V0",
    entry_point="robodojo.sim.environment.environment.isaac.isaac_rl_env:IsaacRLEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs={"env_cfg_entry_point": "robodojo.sim.environment.environment.isaac.isaac_rl_env:IsaacRLEnvCfg"},
)
