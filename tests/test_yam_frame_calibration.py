from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles import load_environment_profile
from robodojo.sim.calibration.matched_replay import _control_from_manifest
from robodojo.sim.calibration.wrist_camera import (
    CorrectionBounds,
    fit_bounded_mirrored_correction,
    fit_yam_matched_manifest,
    load_yam_matched_manifest,
    pinhole_pose_jacobian,
    pinhole_project,
    yam_matched_manifest_status,
)
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
MANIFEST = ROOT / "configs/reference/bimanual_yam_matched_frames.yml"


def test_molmo_scene_is_profile_isolated_and_clones_default_nonappearance_contract():
    profile = load_environment_profile(RepositoryPaths.resolve(ROOT), "bimanual_yam")
    assert profile.document.config.scene == "molmo_yam"
    assert profile.matched_replay_manifest == MANIFEST

    default = yaml.safe_load((ROOT / "configs/scene/default.yml").read_text())
    molmo = yaml.safe_load((ROOT / "configs/scene/molmo_yam.yml").read_text())
    comparable = deepcopy(molmo)
    assert comparable["Table"].pop("visual_color") == [0.20, 0.085, 0.035]
    assert comparable["Room"].pop("visual_color") == [0.75, 0.75, 0.72]
    assert comparable == default
    assert "visual_color" not in default["Table"] and "visual_color" not in default["Room"]
    assert molmo["Table"]["default"] == "material_0122"
    assert max(molmo["Table"]["visual_color"]) < 0.5
    assert min(molmo["Room"]["visual_color"]) >= 0.7
    assert molmo["Background"] == {"intensity": 1000, "default": "brown_photostudio_02_4k.hdr"}


def test_active_scene_appearance_overlays_replayed_fixture_without_changing_geometry():
    replayed = {
        "default_pos": [0.0, -0.05, 0.74],
        "scale": [1.4, 1.1, 0.05],
        "collision": True,
        "visual_color": [1.0, 1.0, 1.0],
    }
    molmo = merge_fixture_appearance(replayed, {"visual_color": [0.20, 0.085, 0.035]})
    assert molmo["visual_color"] == [0.20, 0.085, 0.035]
    assert {key: value for key, value in molmo.items() if key != "visual_color"} == {
        key: value for key, value in replayed.items() if key != "visual_color"
    }

    default = merge_fixture_appearance(replayed, {})
    assert "visual_color" not in default
    assert default["default_pos"] == replayed["default_pos"]
    assert default["scale"] == replayed["scale"]
    assert default["collision"] is True


def test_d405_proxy_is_wrist_only_and_preserves_normalized_optical_pose():
    config = yaml.safe_load((ROOT / "configs/camera/bimanual_yam.yml").read_text())
    rig = normalize_camera_rig(config)
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


def test_released_frame_manifest_is_pinned_and_fail_closed_until_fit_is_accepted():
    manifest = load_yam_matched_manifest(MANIFEST)
    status = yam_matched_manifest_status(manifest)
    assert status["status"] == "complete"
    assert status["selected_frames"] == 24
    assert status["expected_frames"] == 24
    assert status["fit_enabled"] is True
    assert [dataset["revision"] for dataset in manifest["datasets"].values()] == [
        "605ebe5de5fffa11ade4ed17f3ad66cd2cf6dac9",
        "a84f36156e064c4ad70592743639cbfe233000ca",
        "a57ba8d100cbf4d085e2f84da937f5eba4ecd7f4",
    ]
    assert [dataset["split"] for dataset in manifest["datasets"].values()] == [
        "training",
        "training",
        "held_out",
    ]
    artifact = ROOT / "configs/reference/bimanual_yam_landmark_annotations.json"
    assert manifest["annotation_artifact"]["sha256"] == (
        "29b71657dd5564d0d209ac482d383c76b1e56fa99679c07d9357d1bce35667fa"
    )
    assert len(manifest["selection_contract"]["selected_sample_ids"]) == 24
    assert sum(
        len(frame["wrist_annotations"][side]["landmarks"])
        for frame in manifest["selection_contract"]["frames"]
        for side in ("left", "right")
    ) == 192
    assert artifact.is_file()
    assert load_yam_matched_manifest(MANIFEST, require_complete=True)["status"] == "complete"


def test_matched_frame_manifest_fails_closed_without_landmark_artifact(tmp_path: Path):
    document = yaml.safe_load(MANIFEST.read_text())
    document["annotation_artifact"]["path"] = "absent.json"
    local_manifest = tmp_path / MANIFEST.name
    local_manifest.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="landmark artifact is absent"):
        load_yam_matched_manifest(local_manifest)


def test_annotated_fit_is_deterministic_and_matches_persisted_runtime_application():
    manifest = load_yam_matched_manifest(MANIFEST)
    camera_config = yaml.safe_load((ROOT / "configs/camera/bimanual_yam.yml").read_text())
    fit = fit_yam_matched_manifest(manifest, camera_config)
    persisted = manifest["fit_contract"]["fit_result"]
    assert fit.held_out_corrected_median_px == pytest.approx(persisted["held_out"]["corrected_median_px"])
    assert fit.held_out_corrected_median_px <= 8.0
    assert fit.held_out_improvement_fraction >= 0.30
    for side, metrics in fit.held_out_by_side.items():
        assert metrics == pytest.approx(persisted["held_out"]["by_side"][side])
    assert all(metrics["corrected_median_px"] <= 8.0 for metrics in fit.held_out_by_side.values())
    assert all(metrics["improvement_fraction"] >= 0.30 for metrics in fit.held_out_by_side.values())
    for name, norms in fit.correction.to_dict()["norms"].items():
        assert norms == pytest.approx(persisted["norms"][name])
    left_mount = camera_config["camera_rig"]["cameras"]["cam_left_wrist"]["mount"]
    right_mount = camera_config["camera_rig"]["cameras"]["cam_right_wrist"]["mount"]
    assert left_mount["position"] == [0.0, 0.09, 0.06]
    assert right_mount["position"] == [0.0, 0.09, 0.06]
    assert left_mount["calibration_correction"]["translation_m"] == pytest.approx(fit.correction.left[:3])
    mirror = np.diag([-1.0, 1.0, 1.0])
    parameter_mirror = np.diag([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0])
    assert right_mount["calibration_correction"]["translation_m"] == pytest.approx(
        (parameter_mirror @ fit.correction.right_mirrored)[:3]
    )
    assert manifest["fit_contract"]["mirror_matrix"] == mirror.tolist()


def test_visual_only_clamp_application_is_bounded_mirrored_and_changes_no_physics_contract():
    robot_config = yaml.safe_load((ROOT / "configs/robot/dual_yam.yml").read_text())
    parameter_mirror = np.diag([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0])
    for robot in robot_config["robots"]:
        calibration = validate_visual_calibration(robot["visual_calibration"])
        left = np.asarray(calibration["visuals"]["tip_left/visuals"])
        right = np.asarray(calibration["visuals"]["tip_right/visuals"])
        assert right == pytest.approx(parameter_mirror @ left)
        assert np.linalg.norm(left[:3]) <= 0.003 + 1e-12
        assert np.linalg.norm(left[3:]) <= 1.0 + 1e-12
        assert "collision" not in calibration and "rigid_body" not in calibration

    tip_world = np.eye(4)
    tip_world[:3, 3] = [0.1, 0.2, 0.3]
    frame_world = np.eye(4)
    original = np.eye(4)
    correction = [0.001, 0.0, 0.0, 0.0, 0.0, 0.5]
    local = visual_only_local_matrix(tip_world, frame_world, original, correction)
    expected_world = correction_matrix(correction) @ tip_world
    assert tip_world @ local == pytest.approx(expected_world)

    calibration = robot_config["robots"][0]["visual_calibration"]
    collision_path = "tip_left/collisions"
    transforms = {
        "tip_left/visuals": np.eye(4),
        "tip_right/visuals": np.eye(4),
        collision_path: np.diag([1.0, 1.0, 1.0, 2.0]),
    }
    collision_before = transforms[collision_path].copy()
    contexts = {
        path: (np.eye(4), transforms[path].copy()) for path in calibration["visuals"]
    }
    transforms.update(plan_visual_calibration_matrices(calibration, np.eye(4), contexts))
    assert transforms[collision_path] == pytest.approx(collision_before)
    assert set(plan_visual_calibration_matrices(calibration, np.eye(4), contexts)) == set(
        calibration["visuals"]
    )


def test_yam_replay_adapter_preserves_left_first_order_and_profile_sign_bridge():
    manifest = load_yam_matched_manifest(MANIFEST)
    state = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.25]
    control = _control_from_manifest(manifest, state).control_info_dict
    assert control["left_arm_joint_state"]["position"] == pytest.approx([0.1, 0.2, 0.3, 0.4, -0.5, 0.6])
    assert control["right_arm_joint_state"]["position"] == pytest.approx([-0.1, -0.2, -0.3, -0.4, 0.5, -0.6])
    assert control["left_ee_joint_state"]["position"] == pytest.approx([-0.0475])
    assert control["right_ee_joint_state"]["position"] == pytest.approx([-0.011875])


def test_pinhole_projection_and_runtime_jacobian_use_usd_camera_axes():
    points = np.asarray(
        [[0.0, 0.0, -1.0], [0.1, 0.0, -1.0], [0.0, 0.1, -1.0], [0.1, 0.1, -1.2]]
    )
    intrinsics = np.asarray([100.0, 100.0, 50.0, 40.0])
    pixels = pinhole_project(points, np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0]), intrinsics)
    assert pixels[0] == pytest.approx([50.0, 40.0])
    assert pixels[1, 0] > 50.0
    assert pixels[2, 1] < 40.0
    jacobian = pinhole_pose_jacobian(
        points, np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0]), intrinsics
    )
    assert jacobian.shape == (8, 6)
    assert np.linalg.matrix_rank(jacobian) == 6


def _bounds() -> CorrectionBounds:
    return CorrectionBounds(
        shared_translation_m=0.005,
        shared_rotation_deg=2.0,
        per_arm_residual_translation_m=0.002,
        per_arm_residual_rotation_deg=0.5,
        visual_clamp_translation_m=0.003,
        visual_clamp_rotation_deg=1.0,
    )


def test_bounded_mirrored_fit_is_deterministic_and_rejects_out_of_contract_results():
    mirror = np.diag([-1.0, 1.0, 1.0])
    parameter_mirror = np.diag([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0])
    desired = np.asarray([0.001, -0.0005, 0.00025, 0.2, -0.1, 0.05])
    identity = np.eye(6)
    fit = fit_bounded_mirrored_correction(
        identity,
        desired,
        identity,
        parameter_mirror @ desired,
        mirror,
        _bounds(),
    )
    assert fit.shared == pytest.approx(desired)
    assert fit.left == pytest.approx(desired)
    assert fit.right_mirrored == pytest.approx(desired)

    too_large = desired.copy()
    too_large[0] = 0.006
    bounded = fit_bounded_mirrored_correction(
        identity,
        too_large,
        identity,
        parameter_mirror @ too_large,
        mirror,
        _bounds(),
    )
    assert np.linalg.norm(bounded.shared[:3]) <= _bounds().shared_translation_m + 1e-9

    left = np.asarray([0.003, 0, 0, 0, 0, 0], dtype=float)
    right_mirrored = -left
    asymmetric = fit_bounded_mirrored_correction(
        identity,
        left,
        identity,
        parameter_mirror @ right_mirrored,
        mirror,
        _bounds(),
    )
    assert asymmetric.left_residual_translation_m <= _bounds().per_arm_residual_translation_m + 1e-9
    assert asymmetric.right_residual_translation_m <= _bounds().per_arm_residual_translation_m + 1e-9
