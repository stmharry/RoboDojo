import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robodojo.sim.environment.reward_manager.func_parser import Func_Parser
from robodojo.sim.environment.reward_manager.reward_manager import RewardManager

ROOT = Path(__file__).resolve().parents[1]


def test_every_task_reward_predicate_exists_on_the_manager_and_parser():
    predicate_prefixes = ("is_", "all_robot", "get_label", "find_relative", "select_label")
    referenced = set()
    for path in (ROOT / "src/robodojo/sim/tasks").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            if (
                isinstance(owner, ast.Attribute)
                and owner.attr == "reward_manager"
                and node.func.attr.startswith(predicate_prefixes)
            ):
                referenced.add(node.func.attr)

    manager = RewardManager(1)
    missing_wrappers = sorted(name for name in referenced if not callable(getattr(manager, name, None)))
    missing_predicates = sorted(name for name in referenced if not callable(getattr(manager.func_parser, name, None)))
    assert referenced
    assert missing_wrappers == []
    assert missing_predicates == []


def test_reward_check_requires_consecutive_hold_steps_and_reset_clears_state():
    manager = RewardManager(1)
    manager.func_parser._check_env_success = lambda env_idx: True
    state = {"value": True}
    manager.check_once = lambda check, env_idx: state["value"]
    manager.check([("condition", {})], hold_steps=3)

    manager.step()
    manager.step()
    assert manager.check_hold_counts == [[2]]
    state["value"] = False
    manager.step()
    assert manager.check_hold_counts == [[0]]
    state["value"] = True
    manager.step()
    manager.step()
    assert manager.get_reward(final_check=False) == [0.0]
    manager.step()
    assert manager.get_reward(final_check=False) == [1.0]

    manager.reset()
    assert manager.check_list == [[]]
    assert manager.check_hold_steps == [[]]
    assert manager.check_hold_counts == [[]]


def test_reward_check_rejects_invalid_hold_steps():
    manager = RewardManager(1)
    for value in (0, -1, 1.5, True):
        try:
            manager.check([], hold_steps=value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"hold_steps={value!r} should be rejected")


def test_transition_scoring_advances_one_state_per_step_and_gates_completion():
    manager = RewardManager(1)
    manager.func_parser._check_env_success = lambda _env_idx: True
    status = {"first": True, "second": True}
    manager.check_once = lambda check, _env_idx, **_kwargs: status[check[0]]
    manager.score(
        [[("first", {})], [("second", {})]],
        [25, 100],
        score_mode="transition",
    )

    manager.step()
    assert manager.score_completed_count == [1]
    assert manager.get_score([0.0]) == [25.0]
    manager.step()
    assert manager.score_completed_count == [2]
    assert manager.get_score([0.0]) == [25.0]
    assert manager.get_score([1.0]) == [100.0]


def test_rising_edge_trigger_waits_for_transition_then_completes():
    manager = RewardManager(1)
    manager.func_parser._check_env_success = lambda _env_idx: True
    status = {"condition": False, "completion": True}
    manager.check_once = lambda check, _env_idx, **_kwargs: status[check[0]]
    manager.trigger_check([("condition", {})], [("completion", {})], trigger_mode="rising_edge")

    manager.step()
    assert len(manager.trigger_check_list[0]) == 1
    status["condition"] = True
    manager.step()
    assert manager.trigger_check_list == [[]]


def test_collection_predicate_counts_exact_membership():
    parser = Func_Parser(1)
    inside = {"a", "c"}
    parser.is_A_in_B = lambda args: float(args["label_A"] in inside)

    assert parser.is_N_A_in_B({"env_idx": 0, "label_A_list": ["a", "b", "c"], "label_B": "box", "N": 2}) == 1.0
    assert parser.is_N_A_in_B({"env_idx": 0, "label_A_list": ["a", "b", "c"], "label_B": "box", "N": 1}) == 0.0


class _Object:
    def __init__(self, link_poses=None):
        self.link_poses = link_poses or {}

    def get_link_pose(self, link_name):
        return np.asarray(self.link_poses[link_name], dtype=float)


class _Layout:
    def __init__(self, *, item_type="rigid", item_pose=None, item_metadata=None, item_object=None):
        self.instance_type_by_env = [{"item_0": item_type, "container_0": "articulation"}]
        self.scene_manager = SimpleNamespace(env_origins=[np.array([4.0, 0.0, 0.0])])
        self.item_pose = item_pose or [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        self.metadata = {
            "item_0": item_metadata
            or {
                "geometry": {
                    "oriented_bbox": {
                        "vertices": [[x, y, z] for x in (-0.2, 0.2) for y in (-0.2, 0.2) for z in (-0.2, 0.2)]
                    }
                }
            },
            "container_0": {
                "passive": {
                    "volumes": {
                        "packing_cavity": {
                            "base_link": "base",
                            "minimum": [-0.5, -0.5, -0.5],
                            "maximum": [0.5, 0.5, 0.5],
                        }
                    }
                }
            },
        }
        self.objects = {
            "item_0": item_object or _Object(),
            "container_0": _Object({"base": [4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}),
        }

    def get_instance_name(self, *, label, env_idx):
        return {"item": "item_0", "container": "container_0"}.get(label)

    def get_instance_metadata(self, *, inst_name, env_idx):
        return self.metadata[inst_name]

    def get_scene_object(self, *, inst_name, env_idx):
        return self.objects[inst_name]

    def get_instance_bbox_vertices(self, *, inst_name, env_idx):
        return self.metadata[inst_name]["geometry"]["oriented_bbox"]["vertices"]

    def get_instance_pose(self, *, inst_name, env_idx):
        pose = np.asarray(self.item_pose, dtype=float)
        return pose[:3], pose[3:]


def _volume_check(layout, margin=0.0):
    parser = Func_Parser(1)
    parser.layout_manager = layout
    return parser.is_object_in_functional_volume(
        {
            "env_idx": 0,
            "label_A": "item",
            "label_B": "container",
            "B_volume_tag": "packing_cavity",
            "margin": margin,
        }
    )


def test_functional_volume_checks_every_rigid_bbox_vertex_in_link_frame():
    assert _volume_check(_Layout(), margin=0.05) == 1.0
    assert _volume_check(_Layout(item_pose=[0.35, 0, 0, 1, 0, 0, 0]), margin=0.05) == 0.0


def test_functional_volume_uses_current_articulation_link_bboxes():
    link_vertices = [[x, y, z] for x in (-0.05, 0.05) for y in (-0.05, 0.05) for z in (-0.05, 0.05)]
    metadata = {"geometry": {"link_bboxes": {"link0": {"vertices": link_vertices}}}}
    inside = _Object({"link0": [4.3, 0, 0, 1, 0, 0, 0]})
    outside = _Object({"link0": [4.6, 0, 0, 1, 0, 0, 0]})
    assert _volume_check(_Layout(item_type="articulation", item_metadata=metadata, item_object=inside)) == 1.0
    assert _volume_check(_Layout(item_type="articulation", item_metadata=metadata, item_object=outside)) == 0.0


def test_joint_ratio_can_include_the_exact_closure_boundary():
    joint = SimpleNamespace(get_joint_info=lambda _name: {"position": 8.0, "lower": 0.0, "upper": 110.0})
    parser = Func_Parser(1)
    parser.layout_manager = SimpleNamespace(
        get_instance_name=lambda **_kwargs: "container_0",
        get_scene_object=lambda **_kwargs: joint,
        get_instance_metadata=lambda **_kwargs: {"passive": {"functional": {"lid": {"parent_joint": "lid_hinge"}}}},
    )
    args = {
        "env_idx": 0,
        "label": "container",
        "percentage": 8.0 / 110.0,
        "tag": "lid",
    }

    assert parser.is_joint_position_below_ratio(args) == 0.0
    assert parser.is_joint_position_below_ratio({**args, "inclusive": True}) == 1.0
