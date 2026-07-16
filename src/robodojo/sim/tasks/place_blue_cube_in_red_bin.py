from robodojo.sim.environment.environment.task_env import TaskEnv
from robodojo.sim.environment.reward_manager.reward_manager import RewardManager


class place_blue_cube_in_red_bin(TaskEnv):
    """Place the complete 35 mm cube inside the bin for 15 policy steps."""

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = 750

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()

    def soft_reset(self):
        """Clear task-owned success/hold state between soft-reset episodes."""
        self.reward_manager.reset()

    def _contained(self):
        return self.reward_manager.is_object_in_functional_volume(
            label_A="cube",
            label_B="bin",
            B_volume_tag="pickplace_cavity",
            margin=0.003,
        )

    def run_reward(self):
        self.reward_manager.check([self._contained()], hold_steps=15)

    def get_score(self):
        self.reward_manager.score([[self._contained()]], [100])

    def gen_instruction(self, env_idx):
        return ["pick up blue cube and place in red bin"]

    def get_episode_metadata(self, env_idx):
        return {
            "cube_category": "piper_blue_cube",
            "bin_category": "piper_red_bin",
            "success_margin_m": 0.003,
            "success_hold_policy_steps": 15,
        }
