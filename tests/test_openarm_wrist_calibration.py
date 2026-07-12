from pathlib import Path

import numpy as np
import pytest
import yaml

from robodojo.sim.calibration.matched_replay import _control_from_right_first
from robodojo.sim.calibration.wrist_camera import (
    fisheye_project,
    fit_manifest,
    fit_metrics,
    held_out_geometry_metrics,
    load_manifest,
    validate_frame,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/reference/openarm_lerobot_wrist_calibration.yml"


def test_usd_to_opencv_fisheye_projection_convention():
    points = np.array([[0.0, 0.0, -1.0], [0.1, 0.0, -1.0], [0.0, 0.1, -1.0]])
    pixels = fisheye_project(
        points,
        np.zeros(3),
        np.zeros(3),
        np.array([100.0, 100.0, 50.0, 40.0]),
        np.zeros(4),
    )
    assert pixels[0] == pytest.approx([50.0, 40.0])
    assert pixels[1, 0] > 50.0
    assert pixels[2, 1] < 40.0  # USD +Y is image-up.


def test_checksum_validation_rejects_modified_frame(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"pinned frame")
    validate_frame(frame, "200064d262d9d4531527b4ca9e41d4eb8f20e27db54e0d12502d94e49d3c2b8f")
    frame.write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_frame(frame, "200064d262d9d4531527b4ca9e41d4eb8f20e27db54e0d12502d94e49d3c2b8f")


def test_wrist_fit_is_independent_deterministic_and_meets_acceptance_limits():
    manifest = load_manifest(MANIFEST)
    first = fit_manifest(manifest)
    second = fit_manifest(manifest)
    assert not np.allclose(first["left"].position, first["right"].position)
    assert not np.allclose(first["left"].intrinsics, first["right"].intrinsics)
    for side in ("left", "right"):
        assert first[side].position == pytest.approx(second[side].position, abs=1e-10)
        assert first[side].orientation == pytest.approx(second[side].orientation, abs=1e-10)
        metrics = fit_metrics(first[side])
        assert metrics["held_out_median_px"] <= 12.0
        assert metrics["held_out_p95_px"] <= 25.0
        initial_roll = manifest["cameras"][side]["initial_guess"]["orientation"][2]
        assert abs(first[side].orientation[2] - initial_roll) <= 2.0
        assert metrics["held_out_p95_px"] / 1280.0 <= 0.02
        geometry = held_out_geometry_metrics(manifest["cameras"][side], first[side])
        assert geometry["roll_error_deg"] <= 2.0
        assert geometry["normalized_jaw_centroid_error"] <= 0.02
        assert geometry["normalized_jaw_separation_error"] <= 0.02


def test_fitted_values_are_committed_to_independent_camera_blocks():
    manifest = load_manifest(MANIFEST)
    fits = fit_manifest(manifest)
    rig = yaml.safe_load((ROOT / "configs/camera/openarm_lerobot.yml").read_text())["camera_rig"]["cameras"]
    for side, key in (("left", "cam_left_wrist"), ("right", "cam_right_wrist")):
        camera = rig[key]
        fit = fits[side]
        assert camera["mount"]["position"] == pytest.approx(fit.position, abs=1e-6)
        assert camera["mount"]["orientation"] == pytest.approx(fit.orientation, abs=1e-6)
        assert [camera["projection"][name] for name in ("fx", "fy", "cx", "cy")] == pytest.approx(
            fit.intrinsics, abs=1e-6
        )
        assert camera["projection"]["distortion_coefficients"] == pytest.approx(fit.distortion, abs=1e-6)


def test_manifest_provenance_and_holder_occlusion_gate():
    manifest = load_manifest(MANIFEST)
    assert manifest["dataset"]["revision"] == "2e1b2e913cd367d74dc4481736954eed4a051ddc"
    assert manifest["holder_audit"]["optical_frame_registered"] is False
    assert manifest["holder_audit"]["rendered"] is False
    for side in ("left", "right"):
        observations = manifest["cameras"][side]["observations"]
        assert {item["split"] for item in observations} == {"training", "held_out"}
        assert all(len(item["state"]) == 16 for item in observations)
        assert all(len(item["sha256"]) == 64 for item in observations)


def test_matched_replay_preserves_right_first_state_packing():
    state = list(range(16))
    control = _control_from_right_first(state).control_info_dict
    assert control["right_arm_joint_state"]["position"] == pytest.approx(np.deg2rad(state[:7]))
    assert control["right_ee_joint_state"]["position"] == [0.0]
    assert control["left_arm_joint_state"]["position"] == pytest.approx(np.deg2rad(state[8:15]))
    assert control["left_ee_joint_state"]["position"] == [0.0]

    open_state = [0.0] * 16
    open_state[7] = open_state[15] = -65.0
    open_control = _control_from_right_first(open_state).control_info_dict
    assert open_control["right_ee_joint_state"]["position"] == [0.044]
    assert open_control["left_ee_joint_state"]["position"] == [0.044]
