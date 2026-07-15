from robodojo.sim.environment.environment.task_env import TaskEnv
from robodojo.sim.environment.reward_manager.reward_manager import RewardManager

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
        self.step_lim = 1300

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()

    def _completion_checks(self):
        rm = self.reward_manager
        return [
            rm.is_object_in_functional_volume(
                label_A="item",
                label_B="container",
                B_volume_tag="packing_cavity",
                margin=0.002,
            ),
            rm.is_joint_position_below_ratio(label="container", percentage=8.0 / 110.0, tag="lid", inclusive=True),
        ]

    def run_reward(self):
        self.reward_manager.check(self._completion_checks(), hold_steps=15)

    def get_score(self):
        rm = self.reward_manager
        inside = rm.is_object_in_functional_volume(
            label_A="item",
            label_B="container",
            B_volume_tag="packing_cavity",
            margin=0.002,
        )
        closed = rm.is_joint_position_below_ratio(label="container", percentage=8.0 / 110.0, tag="lid", inclusive=True)
        rm.score([[inside], [inside, closed]], [50, 100], score_mode="transition")

    def _variant(self, env_idx):
        layout_id = int(self.env_seeds[env_idx])
        return VARIANTS.get(layout_id, {"variant": "item", "item": "item", "category": "unknown"})

    def gen_instruction(self, env_idx):
        return ["Put the <item> into the black box, then close the lid."]

    def get_episode_metadata(self, env_idx):
        variant = self._variant(env_idx)
        return {
            "variant": variant["variant"],
            "item_category": variant["category"],
            "container_category": "moonlake_magnetic_gift_box",
        }


class pack_item_into_container(PackItemIntoContainerCommon, TaskEnv):
    pass
