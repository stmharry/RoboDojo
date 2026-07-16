from robodojo.sim.environment.environment.task_env import TaskEnv
from robodojo.sim.environment.reward_manager.reward_manager import RewardManager
from robodojo.sim.packing_progress import SINGLE_PACKING_PROMPT, PackingProgressTracker

VARIANTS = {
    0: {"variant": "spoon", "item": "pink measuring spoon", "category": "moonlake_measuring_spoon"},
    1: {"variant": "phone", "item": "pink phone", "category": "moonlake_phone15_dummy"},
    2: {"variant": "fruit", "item": "red apple", "category": "moonlake_fake_apple"},
    3: {
        "variant": "screwdriver",
        "item": "Phillips screwdriver",
        "category": "moonlake_phillips_screwdriver",
    },
    4: {"variant": "cable", "item": "black USB cable", "category": "moonlake_anker_cable"},
    5: {"variant": "block", "item": "wooden alphabet block", "category": "moonlake_abc_block"},
}


class PackItemIntoContainerCommon:
    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.packing_progress = PackingProgressTracker(self.num_envs, ("item",))
        self.step_lim = 1300

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()
        self.packing_progress.reset()

    def _completion_checks(self):
        rm = self.reward_manager
        return [
            rm.is_object_in_functional_volume(
                label_A="item",
                label_B="container",
                B_volume_tag="packing_cavity",
                margin=0.002,
            )
        ]

    def run_reward(self):
        self.reward_manager.check(self._completion_checks(), hold_steps=15)

    def update_task_metrics(self, env_idx_list=None):
        self.packing_progress.update(self, env_idx_list=env_idx_list)

    def get_task_metrics(self, env_idx):
        return self.packing_progress.episode_metrics(env_idx)

    def get_episode_progress_score(self, env_idx):
        return self.packing_progress.progress_score(env_idx)

    def _variant(self, env_idx):
        layout_id = int(self.env_seeds[env_idx])
        return VARIANTS.get(layout_id, {"variant": "item", "item": "item", "category": "unknown"})

    def gen_instruction(self, env_idx):
        return [SINGLE_PACKING_PROMPT]

    def get_episode_metadata(self, env_idx):
        variant = self._variant(env_idx)
        return {
            "variant": variant["variant"],
            "item_category": variant["category"],
            "container_category": "moonlake_magnetic_gift_box",
        }


class pack_item_into_container(PackItemIntoContainerCommon, TaskEnv):
    pass
