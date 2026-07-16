"""Contract tests for corrected Moonlake single-item packing."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robodojo.sim.environment.scene_manager.active_pose import validate_active_pose
from robodojo.sim.environment.scene_manager.joint_layout import resolve_initial_joint_positions
from robodojo.sim.environment.scene_manager.services.layout_objects import LayoutObjectsService
from robodojo.sim.evaluation.services.episodes import summarize_task_metrics
from robodojo.sim.packing_progress import SINGLE_PACKING_PROMPT, PackingProgressTracker


def test_saved_layout_instance_names_are_stable_across_reloads_and_mixed_categories():
    service = LayoutObjectsService()
    service.env_roots = ["/World/envs/env_0"]
    service._object_id_counters = [{}]
    first = service._generate_object_paths(0, "spoon", 0, "rigid")
    duplicate = service._generate_object_paths(0, "spoon", 0, "rigid")
    other = service._generate_object_paths(0, "phone", 0, "rigid")

    service._object_id_counters[0] = {}
    assert service._generate_object_paths(0, "spoon", 0, "rigid") == first
    assert service._generate_object_paths(0, "spoon", 0, "rigid") == duplicate
    assert service._generate_object_paths(0, "phone", 0, "rigid") == other
    assert first[1].endswith("_0")
    assert duplicate[1].endswith("_1")


def test_saved_pose_restore_rejects_an_active_offscreen_wrapper():
    with pytest.raises(RuntimeError, match="offscreen pool"):
        validate_active_pose("active", np.array([100000.0, 0.0, 0.0]))


def test_initial_articulation_joint_mapping_validates_names_limits_and_radians():
    result = resolve_initial_joint_positions(
        [0.0, 0.25],
        ["lid_hinge", "other"],
        [0.0, -1.0],
        [1.919862177, 1.0],
        {"lid_hinge": 1.658062789},
    )
    assert result.tolist() == [1.658062789, 0.25]

    with pytest.raises(ValueError, match="not present"):
        resolve_initial_joint_positions([0.0], ["lid_hinge"], [0.0], [2.0], {"unknown": 1.0})
    with pytest.raises(ValueError, match="outside limits"):
        resolve_initial_joint_positions([0.0], ["lid_hinge"], [0.0], [1.0], {"lid_hinge": 1.1})
    with pytest.raises(ValueError, match="finite"):
        resolve_initial_joint_positions([0.0], ["lid_hinge"], [0.0], [2.0], {"lid_hinge": float("nan")})


def test_single_packing_prompt_matches_the_upstream_annotation():
    assert SINGLE_PACKING_PROMPT == "Place the <item> into the container."


class _FakeRobot:
    type = "target"
    gripper_scale = [-0.0475, 0.0]
    gripper_move = {"sign": -1.0}

    def __init__(self, name):
        self.gripper_name = name


class _FakeLayoutManager:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.75])

    def get_instance_name(self, env_idx, label):
        return f"{label}_instance"

    def get_instance_pose(self, **kwargs):
        return self.position.copy(), np.array([1.0, 0.0, 0.0, 0.0])


class _FakeRobotManager:
    def __init__(self):
        self.robot_list = [_FakeRobot("left_gripper"), _FakeRobot("right_gripper")]
        self.end_effectors = [np.array([0.07, 0.0, 0.75]), np.array([0.5, 0.0, 0.75])]
        self.grippers = [-0.0475, -0.0475]

    def get_real_endpose(self, robot, **kwargs):
        index = self.robot_list.index(robot)
        return {0: np.concatenate([self.end_effectors[index], [1.0, 0.0, 0.0, 0.0]])}

    def get_end_effector_real_val(self, robot, **kwargs):
        return {0: np.array([self.grippers[self.robot_list.index(robot)]])}


class _FakeRewardManager:
    def __init__(self):
        self.inside = False
        self.func_parser = SimpleNamespace(
            pre_state=[{"item_instance": {"pose": np.array([0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0])}}]
        )

    def is_object_in_functional_volume(self, **kwargs):
        return ("inside", kwargs)

    def call_func_parser(self, predicate, env_idx):
        return float(self.inside)


def test_packing_progress_tracks_geometric_and_gripper_milestones_per_control_step():
    layout_manager = _FakeLayoutManager()
    robot_manager = _FakeRobotManager()
    reward_manager = _FakeRewardManager()
    env = SimpleNamespace(
        scene_manager=SimpleNamespace(layout_manager=layout_manager),
        robot_manager=robot_manager,
        reward_manager=reward_manager,
    )
    tracker = PackingProgressTracker(1, ("item",))

    tracker.update(env, [0])
    robot_manager.grippers[0] = 0.0
    tracker.update(env, [0])
    layout_manager.position = np.array([0.09, 0.0, 0.78])
    robot_manager.end_effectors[0] = np.array([0.10, 0.0, 0.78])
    tracker.update(env, [0])
    reward_manager.inside = True
    robot_manager.grippers[0] = -0.0475
    for _ in range(5):
        tracker.update(env, [0])

    metrics = tracker.episode_metrics(0)
    item = metrics["items"]["item"]
    assert item == {
        "moved": True,
        "approached": True,
        "grasp_attempted": True,
        "lifted": True,
        "carried": True,
        "placed": True,
        "released_inside": True,
        "associated_gripper": "left_gripper",
    }
    assert metrics["max_items_inside"] == 1
    assert metrics["total_items"] == 1
    assert metrics["complete"] is False
    assert tracker.progress_score(0) == 100.0


def test_task_metric_summary_is_recomputed_only_from_episode_details():
    details = {
        0: {"task_metrics": {"items": {"item": {"lifted": True, "placed": False}}, "max_items_inside": 0}},
        1: {
            "task_metrics": {
                "items": {"item": {"lifted": True, "placed": True}},
                "max_items_inside": 1,
                "complete": True,
            }
        },
    }
    assert summarize_task_metrics(details) == {
        "episodes_with_metrics": 2,
        "episodes_with_milestone": {
            "moved": 0,
            "approached": 0,
            "grasp_attempted": 0,
            "lifted": 2,
            "carried": 0,
            "placed": 1,
            "released_inside": 0,
        },
        "max_items_inside": 1,
        "completed_episodes": 1,
    }
