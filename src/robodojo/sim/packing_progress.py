"""Per-control-step progress metrics shared by Moonlake packing tasks."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

SINGLE_PACKING_PROMPT = "Place the <item> into the container."

MILESTONES = (
    "moved",
    "approached",
    "grasp_attempted",
    "lifted",
    "carried",
    "placed",
    "released_inside",
)


def _new_item_state() -> dict:
    return {
        **{milestone: False for milestone in MILESTONES},
        "associated_gripper": None,
        "placement_streak": 0,
    }


class PackingProgressTracker:
    """Track geometric and gripper milestones without claiming jaw contact."""

    def __init__(self, num_envs: int, item_labels: tuple[str, ...]):
        self.num_envs = num_envs
        self.item_labels = item_labels
        self.reset()

    def reset(self, env_idx_list=None):
        if not hasattr(self, "_state"):
            self._state = [{} for _ in range(self.num_envs)]
        if env_idx_list is None:
            env_idx_list = range(self.num_envs)
        for env_idx in env_idx_list:
            self._state[env_idx] = {
                "items": {label: _new_item_state() for label in self.item_labels},
                "max_items_inside": 0,
                "total_items": len(self.item_labels),
                "complete": False,
                "all_inside_streak": 0,
            }

    @staticmethod
    def _normalized_gripper_opening(robot, raw_positions) -> float:
        value = float(np.asarray(raw_positions, dtype=float).reshape(-1)[0])
        lower, upper = (float(v) for v in robot.gripper_scale)
        if upper <= lower:
            raise ValueError(f"invalid gripper scale for {robot.gripper_name}: {robot.gripper_scale}")
        fraction = (value - lower) / (upper - lower)
        if robot.gripper_move["sign"] != 1:
            fraction = 1.0 - fraction
        return float(np.clip(fraction, 0.0, 1.0))

    @staticmethod
    def _initial_position(env, env_idx: int, inst_name: str, fallback) -> np.ndarray:
        pre_state = env.reward_manager.func_parser.pre_state[env_idx].get(inst_name, {})
        pose = pre_state.get("pose")
        if pose is None:
            return fallback.copy()
        return np.asarray(pose, dtype=float).reshape(-1)[:3]

    def update(self, env, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = range(self.num_envs)
        layout_manager = env.scene_manager.layout_manager
        target_robots = [robot for robot in env.robot_manager.robot_list if robot.type == "target"]

        for env_idx in env_idx_list:
            state = self._state[env_idx]
            end_effector_positions = []
            gripper_openings = []
            for robot in target_robots:
                pose = env.robot_manager.get_real_endpose(
                    robot,
                    env_idx_list=[env_idx],
                    is_relative=True,
                )[env_idx]
                end_effector_positions.append(np.asarray(pose, dtype=float).reshape(-1)[:3])
                raw = env.robot_manager.get_end_effector_real_val(robot, env_idx_list=[env_idx])[env_idx]
                gripper_openings.append(self._normalized_gripper_opening(robot, raw))

            inside_count = 0
            for label, item_state in state["items"].items():
                inst_name = layout_manager.get_instance_name(env_idx, label)
                if inst_name is None:
                    raise RuntimeError(f"packing metric label {label!r} has no active layout instance")
                position, _ = layout_manager.get_instance_pose(env_idx=env_idx, inst_name=inst_name)
                position = np.asarray(position, dtype=float).reshape(-1)[:3]
                if not np.all(np.isfinite(position)):
                    raise RuntimeError(f"packing metric label {label!r} has a non-finite pose")
                initial = self._initial_position(env, env_idx, inst_name, position)
                displacement = position - initial
                if np.linalg.norm(displacement) >= 0.03:
                    item_state["moved"] = True
                if displacement[2] >= 0.02:
                    item_state["lifted"] = True
                if item_state["lifted"] and np.linalg.norm(displacement[:2]) >= 0.08:
                    item_state["carried"] = True

                associated_index = None
                distances = []
                if end_effector_positions:
                    distances = [float(np.linalg.norm(ee_pos - position)) for ee_pos in end_effector_positions]
                    closest_index = int(np.argmin(distances))
                    if distances[closest_index] <= 0.075:
                        item_state["approached"] = True
                        if item_state["associated_gripper"] is None:
                            item_state["associated_gripper"] = target_robots[closest_index].gripper_name
                    if item_state["associated_gripper"] is not None:
                        associated_index = next(
                            (
                                index
                                for index, robot in enumerate(target_robots)
                                if robot.gripper_name == item_state["associated_gripper"]
                            ),
                            None,
                        )
                if (
                    associated_index is not None
                    and distances[associated_index] <= 0.075
                    and gripper_openings[associated_index] < 0.8
                ):
                    item_state["grasp_attempted"] = True

                inside = bool(
                    env.reward_manager.call_func_parser(
                        env.reward_manager.is_object_in_functional_volume(
                            label_A=label,
                            label_B="container",
                            B_volume_tag="packing_cavity",
                            margin=0.002,
                        ),
                        env_idx,
                    )
                )
                if inside:
                    inside_count += 1
                    item_state["placement_streak"] += 1
                else:
                    item_state["placement_streak"] = 0
                if item_state["placement_streak"] >= 5:
                    item_state["placed"] = True
                if (
                    inside
                    and item_state["placed"]
                    and associated_index is not None
                    and gripper_openings[associated_index] > 0.8
                ):
                    item_state["released_inside"] = True

            state["max_items_inside"] = max(state["max_items_inside"], inside_count)
            if inside_count == state["total_items"]:
                state["all_inside_streak"] += 1
            else:
                state["all_inside_streak"] = 0
            if state["all_inside_streak"] >= 15:
                state["complete"] = True

    def episode_metrics(self, env_idx: int) -> dict:
        result = deepcopy(self._state[env_idx])
        result.pop("all_inside_streak", None)
        for item in result["items"].values():
            item.pop("placement_streak", None)
        return result

    def progress_score(self, env_idx: int) -> float:
        state = self._state[env_idx]
        items = list(state["items"].values())
        placed_count = sum(bool(item["placed"]) for item in items)
        if placed_count:
            return 100.0
        if any(item["lifted"] for item in items):
            return 50.0
        if any(item["moved"] for item in items):
            return 25.0
        return 0.0
