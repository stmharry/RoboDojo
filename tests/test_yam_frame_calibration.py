from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles import load_environment_profile
from robodojo.sim.environment.camera_manager.mount_registry import (
    align_hardware_frame_pose,
    apply_mount_calibration,
    mount_orientation,
)
from robodojo.sim.environment.camera_manager.rig_spec import normalize_camera_rig
from robodojo.sim.environment.robot_manager.visual_calibration import (
    correction_matrix,
    plan_visual_calibration_matrices,
    validate_visual_calibration,
    visual_only_local_matrix,
)
from robodojo.sim.environment.scene_manager.appearance import merge_fixture_appearance

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_SOURCE = "yam_hardware_calibration_v1"


def test_mast_pose_records_max_open_alignment():
    config = yaml.safe_load((ROOT / "configs/camera/bimanual_yam.yml").read_text())
    mast = config["camera_rig"]["cameras"]["cam_head"]["mount"]
    assert mast["position"] == [-0.037, -0.30, 1.635]

    reference = yaml.safe_load((ROOT / "configs/reference/bimanual_yam.yml").read_text())
    source = reference["sources"]["max_open_endpoint_calibration"]
    alignment = reference["camera_contract"]["max_open_alignment"]
    assert source["method"] == "exact_state_render_alignment"
    assert source["simulator_endpoint_m"] == -0.0475
    assert alignment["mast_world_pose"] == {
        "position_m": mast["position"],
        "orientation_wxyz": mast["orientation"],
    }
    assert alignment["wrist_camera_parameters_changed"] is False
    assert alignment["jaw_visual_parameters_changed"] is False


def test_cloth_workspace_scene_is_independent_of_the_yam_profile():
    profile = load_environment_profile(RepositoryPaths.resolve(ROOT), "bimanual_yam")
    assert profile.document.config.scene == "default"
    assert profile.matched_replay_manifest is None

    default = yaml.safe_load((ROOT / "configs/scene/default.yml").read_text())
    workspace = yaml.safe_load((ROOT / "configs/scene/molmo_yam.yml").read_text())
    comparable = deepcopy(workspace)
    assert comparable["Table"].pop("replay_material_override") == "material_0122"
    assert comparable["Room"].pop("visual_color") == [0.75, 0.75, 0.72]
    assert comparable == default
    assert workspace["Table"]["default"] == "material_0122"
    assert min(workspace["Room"]["visual_color"]) >= 0.7

    replayed = {
        "default": "write_material",
        "default_pos": [0.0, -0.05, 0.74],
        "scale": [1.4, 1.1, 0.05],
        "collision": True,
        "visual_color": [1.0, 1.0, 1.0],
    }
    merged = merge_fixture_appearance(replayed, workspace["Table"])
    assert merged["replay_material_override"] == "material_0122"
    assert "visual_color" not in merged
    assert merged["collision"] is True


def test_final_wrist_camera_values_are_embodiment_owned():
    config = yaml.safe_load((ROOT / "configs/camera/bimanual_yam.yml").read_text())
    cameras = config["camera_rig"]["cameras"]
    expected = {
        "cam_left_wrist": {
            "target": "robot0/wrist_camera_mount",
            "translation_m": [0.0003368883833682653, -0.003866168248125935, 0.005781196674983154],
            "rotation_rotvec_deg": [-1.5213745097970042, 0.2893656817085859, 0.672802376740453],
        },
        "cam_right_wrist": {
            "target": "robot1/wrist_camera_mount",
            "translation_m": [-0.0003081117949818329, -0.003900024586207093, 0.0057492578421972276],
            "rotation_rotvec_deg": [-1.9193526354538613, 0.01345363932629845, -0.5723172103038849],
        },
    }
    for name, values in expected.items():
        mount = cameras[name]["mount"]
        assert mount["target"] == values["target"]
        assert mount["basis"] == "yam_simulation_camera_contract_v1"
        assert mount["calibration_correction"] == {
            "translation_m": values["translation_m"],
            "rotation_rotvec_deg": values["rotation_rotvec_deg"],
            "source": CALIBRATION_SOURCE,
        }

    reference = yaml.safe_load((ROOT / "configs/reference/bimanual_yam.yml").read_text())
    calibration = reference["camera_contract"]["hardware_calibration"]
    provenance = reference["sources"]["historical_hardware_calibration"]
    assert provenance["usage"] == "provenance_only_final_values_retained"
    assert set(provenance["datasets"]) == {
        "allenai/18122025-foldclo-01",
        "allenai/18122025-foldclo-13",
        "allenai/24122025-foldclo-05",
    }
    assert calibration["source"] == CALIBRATION_SOURCE
    assert calibration["wrist_extrinsic_corrections"] == {
        "left": {
            "translation_m": expected["cam_left_wrist"]["translation_m"],
            "rotation_rotvec_deg": expected["cam_left_wrist"]["rotation_rotvec_deg"],
        },
        "right": {
            "translation_m": expected["cam_right_wrist"]["translation_m"],
            "rotation_rotvec_deg": expected["cam_right_wrist"]["rotation_rotvec_deg"],
        },
    }
    assert "policy_checkpoint" not in reference["sources"]
    assert "predicted_horizon" not in reference["state_action_contract"]
    assert "executed_horizon" not in reference["state_action_contract"]
    assert "joint_convention_bridge" not in reference["state_action_contract"]
    assert "policy_mapping" not in reference["camera_contract"]


def test_d405_proxy_is_wrist_only_and_preserves_normalized_optical_pose():
    rig = normalize_camera_rig(yaml.safe_load((ROOT / "configs/camera/bimanual_yam.yml").read_text()))
    top, left, right = rig.cameras
    assert "hardware" not in top.mount
    for wrist in (left, right):
        runtime = wrist.runtime_camera()
        assert runtime["mount_hardware_asset"] == "Robots/yam/D405_proxy.usd"
        assert runtime["mount_hardware_collision"] is False
        assert runtime["mount_hardware_camera_frame"] == "OpticalFrame"
        target_orientation = mount_orientation(
            runtime["ori"], runtime["mount_pose_convention"], runtime["optical_roll_deg"]
        )
        target_position, target_orientation = apply_mount_calibration(
            runtime["pos"],
            target_orientation,
            runtime["mount_calibration_translation_m"],
            runtime["mount_calibration_rotation_rotvec_deg"],
        )
        position, orientation = align_hardware_frame_pose(target_position, target_orientation, np.eye(4))
        assert position == pytest.approx(target_position)
        assert abs(float(np.dot(orientation, target_orientation))) == pytest.approx(1.0)


def test_final_jaw_transforms_remain_visual_only_and_mirrored():
    robot_config = yaml.safe_load((ROOT / "configs/robot/dual_yam.yml").read_text())
    parameter_mirror = np.diag([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0])
    expected_left = [
        [
            0.0014766420522317357,
            0.00025120446006139137,
            -0.002599312326141951,
            -0.15068164338723466,
            0.4066846516683395,
            0.9010564002567725,
        ],
        [
            0.00204980967513032,
            0.0014503406143189645,
            -0.0016415822849309197,
            -0.4996359255700045,
            0.7081368370462536,
            0.4989049627914445,
        ],
    ]
    for robot, expected in zip(robot_config["robots"], expected_left, strict=True):
        calibration = validate_visual_calibration(robot["visual_calibration"])
        assert calibration["frame"] == "gripper/wrist_camera_mount"
        assert calibration["source"] == CALIBRATION_SOURCE
        left = np.asarray(calibration["visuals"]["tip_left/visuals"])
        right = np.asarray(calibration["visuals"]["tip_right/visuals"])
        assert left == pytest.approx(expected)
        assert right == pytest.approx(parameter_mirror @ left)
        assert np.linalg.norm(left[:3]) <= 0.003 + 1e-12
        assert np.linalg.norm(left[3:]) <= 1.0 + 1e-12
        assert "collision" not in calibration and "rigid_body" not in calibration

    reference = yaml.safe_load((ROOT / "configs/reference/bimanual_yam.yml").read_text())
    recorded = reference["camera_contract"]["hardware_calibration"]["visual_only_jaw_transforms"]
    for index, arm in enumerate(("left_arm", "right_arm")):
        assert recorded[arm]["tip_left"] == expected_left[index]
        assert recorded[arm]["tip_right"] == pytest.approx(parameter_mirror @ expected_left[index])

    calibration = robot_config["robots"][0]["visual_calibration"]
    collision_path = "tip_left/collisions"
    transforms = {
        "tip_left/visuals": np.eye(4),
        "tip_right/visuals": np.eye(4),
        collision_path: np.diag([1.0, 1.0, 1.0, 2.0]),
    }
    collision_before = transforms[collision_path].copy()
    contexts = {path: (np.eye(4), transforms[path].copy()) for path in calibration["visuals"]}
    transforms.update(plan_visual_calibration_matrices(calibration, np.eye(4), contexts))
    assert transforms[collision_path] == pytest.approx(collision_before)
    assert set(plan_visual_calibration_matrices(calibration, np.eye(4), contexts)) == set(
        calibration["visuals"]
    )

    tip_world = np.eye(4)
    tip_world[:3, 3] = [0.1, 0.2, 0.3]
    correction = [0.001, 0.0, 0.0, 0.0, 0.0, 0.5]
    local = visual_only_local_matrix(tip_world, np.eye(4), np.eye(4), correction)
    assert tip_world @ local == pytest.approx(correction_matrix(correction) @ tip_world)
