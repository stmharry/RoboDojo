from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest
import yaml

from robodojo.core.experiments.catalogs import load_protocol_catalog
from robodojo.core.experiments.selection import resolve_recipe
from robodojo.core.layouts import resolve_layout_set
from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles.environment import load_environment_profile
from robodojo.core.profiles.scene import load_scene_profile
from robodojo.core.storage import assets_root
from robodojo.core.workspace import validate_resolved_layout_set
from robodojo.workflows.assets_piper_pickplace import _metadata

ROOT = Path(__file__).resolve().parents[1]
PATHS = RepositoryPaths.resolve(ROOT)
RECIPE = "pi05-bimanual_piper-piper_pickplace_tabletop-place_blue_cube_in_red_bin"
TASK = "place_blue_cube_in_red_bin"


def test_piper_recipe_resolves_exact_policy_environment_scene_and_protocol_contract():
    experiment = resolve_recipe(PATHS, RECIPE)

    assert experiment.policy_name == "pi05_bimanual_piper_pickplace"
    assert experiment.policy.checkpoint == "pi05_piper_bimanual_v1"
    assert experiment.environment.name == "bimanual_piper_pickplace"
    assert experiment.environment.embodiment == "bimanual_piper"
    assert experiment.scene.name == "piper_pickplace_tabletop"
    assert experiment.task_protocol == TASK
    assert experiment.policy_reference_match == "reference_match"
    assert experiment.policy_descriptor.execution.model_dump() == {
        "strategy": "fixed_prefix",
        "prediction_horizon": 50,
        "nominal_execution_horizon": 8,
        "maximum_execution_horizon": 8,
    }
    assert experiment.policy_descriptor.interface.state.dimension == 14
    assert experiment.policy_descriptor.interface.action.dimension == 14
    assert experiment.policy_descriptor.interface.action.rate_hz == 30


def test_piper_environment_exposes_three_exact_policy_tuned_camera_streams():
    environment = load_environment_profile(PATHS, "bimanual_piper_pickplace")
    interface = environment.policy_interface

    assert interface.state_dimension == interface.action_dimension == 14
    assert interface.action_rate_hz == 30
    assert [(camera.role, camera.resolutions, camera.rate_hz) for camera in interface.cameras] == [
        ("top", ((224, 224),), 30),
        ("left_wrist", ((224, 224),), 30),
        ("right_wrist", ((224, 224),), 30),
    ]
    assert environment.document.variant.kind == "policy_tuned"
    assert environment.matched_replay_manifest == ROOT / "configs/reference/bimanual_piper_pickplace.yml"
    replay = yaml.safe_load(environment.matched_replay_manifest.read_text(encoding="utf-8"))
    assert replay["classification"] == "policy_tuned"
    assert replay["hardware_calibrated"] is False
    assert replay["camera_fit"]["status"] == "failed_acceptance_experimental"
    assert replay["camera_fit"]["acceptance_px"] == {
        "top_cube_bin_landmarks": 12,
        "wrist_gripper_landmarks": 15,
    }
    assert replay["camera_fit"]["simulator_replay"]["median_state_applied"] is True
    assert replay["camera_fit"]["simulator_replay"]["wrist_workspace_visible"] is True
    assert replay["camera_fit"]["simulator_replay"]["wrist_gripper_landmarks_visible"] is False
    gate = replay["camera_fit"]["qualification_gate"]
    assert gate["passed"] is False
    assert gate["behavioral_benchmark_permitted"] is False
    assert gate["disposition"] == "experimental_diagnostic"
    assert "PiPER-X" in gate["reason"]

    rig = yaml.safe_load((ROOT / "configs/camera/bimanual_piper_pickplace.yml").read_text(encoding="utf-8"))
    cameras = rig["camera_rig"]["cameras"]
    assert [cameras[name]["type"] for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")] == [
        "piper_top",
        "piper_wrist",
        "piper_wrist",
    ]
    for name, target in (
        ("cam_left_wrist", "robot0/wrist_camera_mount"),
        ("cam_right_wrist", "robot1/wrist_camera_mount"),
    ):
        mount = cameras[name]["mount"]
        assert mount["target"] == target
        assert mount["position"] == [0.03, 0.06, 0.10]
        assert mount["orientation"] == pytest.approx([0.70710678, 0.0, 0.5, 0.5])
        assert mount["pose_convention"] == "sapien_robotics"


def test_piper_robot_and_source_asset_contract_are_pinned():
    source = yaml.safe_load((ROOT / "configs/tooling/piper.yml").read_text(encoding="utf-8"))
    robot = yaml.safe_load((ROOT / "configs/robot/dual_piper.yml").read_text(encoding="utf-8"))
    dimensions = json.loads((ROOT / "configs/robot/_robot_info.json").read_text(encoding="utf-8"))

    assert source["sources"]["piper_ros"] == {
        **source["sources"]["piper_ros"],
        "revision": "ac41fcbcdda598f01b51cf6175ed9a24d0dacadc",
        "license": "MIT",
        "urdf_path": "src/piper_description/urdf/piper_description.urdf",
    }
    assert source["joint_limits"] == {
        "joint1": [-2.618, 2.618],
        "joint2": [0.0, 3.14],
        "joint3": [-2.967, 0.0],
        "joint4": [-1.745, 1.745],
        "joint5": [-1.22, 1.22],
        "joint6": [-2.0944, 2.0944],
        "joint7": [0.0, 0.035],
        "joint8": [-0.035, 0.0],
    }
    assert source["robot_config"]["arm_joints_name"] == [f"joint{index}" for index in range(1, 7)]
    assert source["robot_config"]["gripper_move"] == {
        "base": "joint7",
        "sign": 1.0,
        "mimic": ["joint8", -1.0, 0.0],
    }
    assert dimensions["dual_piper"] == {"arm_dim": [6, 6], "ee_dim": [1, 1]}
    assert all(entry["robot_name"] == "piper" and entry["need_planner"] is False for entry in robot["robots"])
    assert [entry["default_root_pos"][:2] for entry in robot["robots"]] == [[-0.25, -0.38], [0.25, -0.38]]


def test_dataset_median_state_is_replayed_with_half_jaw_simulator_values():
    replay = yaml.safe_load((ROOT / "configs/reference/bimanual_piper_pickplace.yml").read_text(encoding="utf-8"))
    robot = yaml.safe_load((ROOT / "configs/robot/dual_piper.yml").read_text(encoding="utf-8"))
    median = replay["state_replay"]["state"]
    simulator = []
    for entry in robot["robots"]:
        initial = entry["initial_joint_positions"]
        simulator.extend(initial[f"joint{index}"] for index in range(1, 7))
        simulator.append(initial["joint7"])
    arm_indices = (*range(6), *range(7, 13))
    assert [simulator[index] for index in arm_indices] == pytest.approx([median[index] for index in arm_indices])
    assert simulator[6] == pytest.approx(median[6] / 2.0)
    # The source q50 is a tiny negative sensor artifact; the simulator starts
    # at the physical hard stop instead of reproducing an invalid opening.
    assert median[13] < 0.0
    assert simulator[13] == 0.0


def test_pickplace_assets_define_exact_cube_and_functional_bin_cavity():
    manifest = yaml.safe_load((ROOT / "configs/tooling/piper_pickplace.yml").read_text(encoding="utf-8"))
    cube = manifest["assets"]["cube"]
    bin_spec = manifest["assets"]["bin"]

    assert cube["dimensions_m"] == [0.035, 0.035, 0.035]
    assert bin_spec["cavity_dimensions_m"] == [0.160, 0.080, 0.055]
    metadata = _metadata(
        bin_spec,
        (-0.084, -0.044, 0.0),
        (0.084, 0.044, 0.059),
        physics_type="articulation",
    )
    assert metadata["geometry"]["aligned_bbox"]["extents"] == pytest.approx([0.168, 0.088, 0.059])
    assert manifest["references"]["training_dataset"]["revision"] == (
        "60e9d08e7f1b40ed213738374dba7acb093b456c"
    )


def test_bundled_layouts_cover_ten_deterministic_cube_and_bin_variants():
    scene = load_scene_profile(PATHS, "piper_pickplace_tabletop")
    layouts = resolve_layout_set(
        config_root=PATHS.environment_configs,
        assets_root=assets_root(),
        benchmark="RoboDojo",
        layout_set=scene.document.layout_set,
        layout_source=scene.document.layout_source,
        task=TASK,
        seed=0,
    )

    assert [layout.layout_id for layout in layouts.layouts] == list(range(10))
    environment = load_environment_profile(PATHS, "bimanual_piper_pickplace")
    validate_resolved_layout_set(
        layouts,
        task_config_path=PATHS.task_configs / f"{TASK}.yml",
        workspace=environment.document.workspace,
        robot_config_path=environment.component_paths["robot"],
    )
    cube_positions = []
    bin_positions = []
    for layout in layouts.layouts:
        payload = json.loads(layout.path.read_text(encoding="utf-8"))
        [cube] = payload["Rigid"]["piper_blue_cube"]
        [bin_entry] = payload["Articulation"]["piper_red_bin"]
        cube_positions.append(cube["default_pos"])
        bin_positions.append(bin_entry["default_pos"])
        cube_yaw_deg = math.degrees(2.0 * math.atan2(cube["default_ori"][3], cube["default_ori"][0]))
        assert abs(cube_yaw_deg) <= 10.0001
        assert cube["label"] == "cube"
        assert bin_entry["label"] == "bin"
    assert max(abs(position[0] + 0.12) for position in cube_positions) <= 0.0300001
    assert max(abs(position[1] + 0.04) for position in cube_positions) <= 0.0200001
    assert max(abs(position[0] - 0.10) for position in bin_positions) <= 0.0100001
    assert max(abs(position[1] - 0.13) for position in bin_positions) <= 0.0100001


def test_task_prompt_success_hold_and_reset_contract_are_explicit():
    path = ROOT / "src/robodojo/sim/tasks/place_blue_cube_in_red_bin.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    [task_class] = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == TASK]
    methods = {node.name: node for node in task_class.body if isinstance(node, ast.FunctionDef)}

    assert "pick up blue cube and place in red bin" in source
    assert "margin=0.003" in source
    assert "hold_steps=15" in source
    assert "return home" not in source.lower()
    for method in ("reset", "soft_reset"):
        calls = [
            node
            for node in ast.walk(methods[method])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "reset"
        ]
        assert calls, f"{method} must reset task-owned reward/hold state"

    protocol = load_protocol_catalog(PATHS).protocols[TASK]
    assert protocol.task == TASK
    assert protocol.episode_horizon == 750
    assert protocol.evaluation_episodes == 10
    assert protocol.compatible_scenes == ["piper_pickplace_tabletop"]
