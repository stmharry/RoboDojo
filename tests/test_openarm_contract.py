import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from env.camera_manager.mount_registry import (
    CameraMountRegistry,
    align_hardware_frame_pose,
    apply_optical_roll,
    compose_pose,
    pose_matrix,
)
from env.camera_manager.rig_spec import CameraSpec, hardware_camera_parent, normalize_camera_rig
from env.scene_manager.appearance_overrides import apply_appearance_overrides
from scripts.assets.openarm_camera_calibration import (
    HEAD_CAD_TO_FIXTURE,
    HEAD_HOLDER,
    WRIST_HOLDER,
    calibration_manifest,
    holder_optical_frame,
    wrist_points_m,
)
from scripts.validate_openarm_visuals import wrist_mask_failures

ROOT = Path(__file__).resolve().parents[1]


def load_rig(name):
    config = yaml.safe_load((ROOT / f"env_cfg/camera/{name}.yml").read_text())
    return normalize_camera_rig(config)


def test_openarm_timing_and_dimensions():
    env_cfg = yaml.safe_load((ROOT / "env_cfg/openarm_cloth_folding.yml").read_text())
    sim_cfg = yaml.safe_load((ROOT / "env_cfg/sim/openarm_cloth_folding.yml").read_text())
    robot_info = json.loads((ROOT / "env_cfg/robot/_robot_info.json").read_text())
    assert env_cfg["observation"]["collect_freq"] == 30
    assert 1.0 / (sim_cfg["dt"] * 30) == 8.0
    assert sum(robot_info["dual_openarm"]["arm_dim"]) + sum(robot_info["dual_openarm"]["ee_dim"]) == 16
    robot_config = yaml.safe_load((ROOT / "scripts/assets/openarm_robot_config.yml").read_text())
    assert robot_config["left"]["camera_mount_links"]["left_wrist_camera_holder"] == "openarm_left_link7"
    assert robot_config["right"]["camera_mount_links"]["right_wrist_camera_holder"] == "openarm_right_link7"


@pytest.mark.parametrize(
    ("name", "profile", "vendor", "fov", "focal"),
    [
        ("openarm_cloth_folding", "openarm_policy_original", "Fafeicy", 140.0, 327.4045),
        ("openarm_cloth_folding_dyna", "openarm_dyna", "Waveshare", 145.0, 316.1146),
    ],
)
def test_openarm_camera_profiles(name, profile, vendor, fov, focal):
    rig = load_rig(name)
    assert rig.profile_id == profile
    assert [camera.observation_key for camera in rig.cameras] == [
        "cam_head", "cam_left_wrist", "cam_right_wrist"
    ]
    base, left, right = rig.cameras
    assert base.sensor["vendor"] == vendor
    assert base.sensor["stream_resolution"] == [640, 480]
    assert base.sensor["diagonal_fov_deg"] == fov
    assert base.projection["fx"] == focal
    assert base.mount["kind"] == "scene_fixture"
    assert base.mount["target"] == "camera_stand"
    assert base.mount["basis"] == "lerobot_head_camera_holder_v4_optical_frame"
    assert base.mount["hardware"]["asset"].endswith("head_camera_holder.usd")
    assert left.sensor["vendor"] == right.sensor["vendor"] == "Arducam"
    assert left.sensor["stream_resolution"] == right.sensor["stream_resolution"] == [1280, 720]
    assert base.mount["position"] == [0.0, -0.33587426, 0.04106626]
    assert base.mount["orientation"] == [120.0, 0.0, -180.0]
    assert base.mount["optical_roll_deg"] == 0.0
    assert base.mount["hardware"]["camera_frame"] == "OpticalFrame"
    assert left.mount["target"] == "robot0/left_wrist_camera_holder"
    assert right.mount["target"] == "robot0/right_wrist_camera_holder"
    assert left.mount["position"] == [0.05, 0.0, 0.12]
    assert right.mount["position"] == [0.035, 0.0, 0.12]
    for wrist in (left, right):
        assert wrist.mount["orientation"] == [180.0, 0.0, -90.0]
        assert wrist.mount["optical_roll_deg"] == 0.0
        assert wrist.mount["hardware"]["camera_frame"] == "OpticalFrame"
        assert "position" not in wrist.mount["hardware"]
        assert "orientation" not in wrist.mount["hardware"]


def test_profiles_differ_only_in_base_sensor_projection_and_id():
    policy = load_rig("openarm_cloth_folding")
    dyna = load_rig("openarm_cloth_folding_dyna")
    assert policy.default_frequency == dyna.default_frequency
    assert policy.annotator == dyna.annotator
    for index in (1, 2):
        assert policy.cameras[index] == dyna.cameras[index]
    policy_base, dyna_base = policy.cameras[0], dyna.cameras[0]
    assert policy_base.role == dyna_base.role == "base"
    assert policy_base.mount == dyna_base.mount
    assert policy_base.camera_type == dyna_base.camera_type
    assert policy_base.mesh == dyna_base.mesh


def test_cad_head_pose_hangs_from_upstream_fixture_tip():
    world_position, world_orientation = compose_pose(
        [0.0, -0.47, 0.765],
        [-90.0, 0.0, 0.0],
        [0.0, -0.33587426, 0.04106626],
        [120.0, 0.0, -180.0],
    )
    assert np.allclose(world_position, [0.0, -0.42893374, 1.10087426], atol=1e-6)
    expected = compose_pose(
        [0, 0, 0], [0, 0, 0], [0, 0, 0], apply_optical_roll([30, 0, 0], 180)
    )[1]
    assert abs(np.dot(world_orientation, expected)) == pytest.approx(1.0)


def test_pinned_cad_anchors_and_mirrored_holder_geometry():
    assert np.allclose(HEAD_CAD_TO_FIXTURE @ HEAD_CAD_TO_FIXTURE.T, np.eye(3))
    assert np.linalg.det(HEAD_CAD_TO_FIXTURE) == pytest.approx(1.0)
    assert HEAD_HOLDER.optical_position_m() == pytest.approx([0.0, 0.24369901916, -0.0669690397])
    assert WRIST_HOLDER.optical_position_m() == pytest.approx([0.0, 0.0669370412, 0.0086344558])
    assert WRIST_HOLDER.optical_direction_mount() == pytest.approx([0.0, 0.6427876097, -0.7660444431])
    for side in ("head", "left", "right"):
        rotation = holder_optical_frame(side)[:3, :3]
        assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-9)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
    left_frame = holder_optical_frame("left")
    right_frame = holder_optical_frame("right")
    assert right_frame[0, 3] == pytest.approx(-left_frame[0, 3])
    assert right_frame[:3, 2] == pytest.approx(left_frame[:3, 2])
    sample = np.array([[1.0, 2.0, 3.0], [20.0, -5.0, 7.0]])
    left = wrist_points_m(sample, "left")
    right = wrist_points_m(sample, "right")
    assert right[:, 0] == pytest.approx(-left[:, 0])
    assert right[:, 1:] == pytest.approx(left[:, 1:])
    manifest = calibration_manifest()
    assert manifest["blog_space_revision"] == "170e1d479579e0b4be1afe0c99ebf868b24803db"
    assert manifest["hardware_revision"] == "ffe34b93c070343042eb9412fbfeffce16139947"
    sources = json.loads((ROOT / "scripts/assets/openarm_sources.json").read_text())
    hashes = sources["hardware_modifications"]["sha256"]
    assert hashes["head camera holder v4.stl"] == "959ae5e0ad6e0870465e361df30db3d1bbdeebb9ba8001274c3ce9e1712f03d3"
    assert hashes["arducam_holder.step"] == "b51c4d565afe4a632c61af15b42a9319c9361271c98840ccd9c670a893b7291d"


def test_wrist_targets_are_roll_only_corrections_from_original_formulation():
    base = Rotation.from_euler("XYZ", [180.0, 0.0, 90.0], degrees=True)
    target = base * Rotation.from_euler("Z", 180.0, degrees=True)
    for legacy_roll, expected_visual in ((-90.0, 90.0), (90.0, -90.0)):
        legacy = base * Rotation.from_euler("Z", legacy_roll, degrees=True)
        frame_delta = (legacy.inv() * target).as_rotvec(degrees=True)
        assert frame_delta == pytest.approx([0.0, 0.0, -expected_visual], abs=1e-8)
        assert target.apply([0.0, 0.0, -1.0]) == pytest.approx(
            legacy.apply([0.0, 0.0, -1.0]), abs=1e-8
        )


def test_wrist_visual_gate_requires_hardware_to_enter_from_bottom():
    hardware = np.zeros((100, 160), dtype=bool)
    hardware[82:, 112:145] = True
    failures, metrics = wrist_mask_failures("left_wrist", hardware, hardware)
    assert failures == []
    assert metrics["edge_fraction"]["bottom"] > 0.05

    top_only = np.zeros_like(hardware)
    top_only[:18, 112:145] = True
    failures, _ = wrist_mask_failures("left_wrist", top_only, top_only)
    assert any("does not enter from the bottom" in failure for failure in failures)


def test_named_hardware_camera_frame_is_relative_and_parented():
    assert hardware_camera_parent("/World/Holder", "OpticalFrame") == "/World/Holder/OpticalFrame"
    with pytest.raises(ValueError, match="relative"):
        hardware_camera_parent("/World/Holder", "../Camera")
    rig = load_rig("openarm_cloth_folding_dyna")
    runtime = rig.runtime_config()
    for key in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        camera = runtime[key].camera
        assert camera.mount_hardware_camera_frame == "OpticalFrame"
        rig_camera = next(item for item in rig.cameras if item.observation_key == key)
        assert list(camera.pos) == list(rig_camera.mount["position"])
    invalid = yaml.safe_load((ROOT / "env_cfg/camera/openarm_cloth_folding.yml").read_text())
    invalid["camera_rig"]["cameras"]["cam_left_wrist"]["mount"]["hardware"]["position"] = [0, 0, 0]
    with pytest.raises(ValueError, match="derive hardware pose"):
        normalize_camera_rig(invalid)


def test_derived_holder_attachments_reconstruct_exact_optical_targets():
    rig = load_rig("openarm_cloth_folding_dyna")
    for camera, side in zip(rig.cameras[1:], ("left", "right"), strict=True):
        frame = holder_optical_frame(side)
        position, orientation = align_hardware_frame_pose(
            camera.mount["position"], camera.mount["orientation"], frame
        )
        reconstructed = pose_matrix(position, orientation) @ frame
        target = pose_matrix(camera.mount["position"], camera.mount["orientation"])
        assert reconstructed == pytest.approx(target, abs=1e-9)


def test_appearance_overlay_cannot_remove_or_mutate_task_objects():
    layout = {
        "Table": {"default": "wood"},
        "Geometry": {"camera_stand": [{"label": "camera_stand"}]},
        "Garment": {"pose": [1, 2, 3]},
    }
    result = apply_appearance_overrides(layout, {"Table": {"default": "white"}})
    assert result["Geometry"] == layout["Geometry"]
    assert result["Garment"] == layout["Garment"]
    with pytest.raises(ValueError, match="non-appearance"):
        apply_appearance_overrides(layout, {"Garment": {"pose": [0, 0, 0]}})
    with pytest.raises(ValueError, match="non-appearance"):
        apply_appearance_overrides(layout, {"remove_fixtures": ["Geometry.camera_stand"]})


def test_legacy_camera_normalization_preserves_flat_contract():
    config = yaml.safe_load((ROOT / "env_cfg/camera/camera_config.yml").read_text())
    rig = normalize_camera_rig(config)
    runtime = rig.runtime_config()
    assert rig.profile_id == "legacy"
    assert runtime.default_frequency == config.get("default_frequency", 30)
    assert "cam_head" in runtime


def test_mount_registry_delegates_to_scene_and_robot_publishers():
    class Scene:
        def resolve_camera_fixture_mount(self, env_id, label):
            return f"/fixture/{env_id}/{label}"

    class Robot:
        def resolve_camera_link_mount(self, env_id, target):
            return f"/robot/{env_id}/{target}"

    registry = CameraMountRegistry(Scene(), Robot())
    assert registry.resolve_parent_path(2, {"mount_kind": "world"}) == "/World/envs/env_2"
    fixture = registry.resolve_parent_path(2, {"mount_kind": "scene_fixture", "mount_target": "stand"})
    robot = registry.resolve_parent_path(2, {"mount_kind": "robot_link", "mount_target": "arm/link"})
    assert fixture == "/fixture/2/stand"
    assert robot == "/robot/2/arm/link"


@pytest.mark.parametrize(
    ("mount", "projection", "message"),
    [
        (
            {"kind": "scene_fixture", "position": [0, 0, 0], "orientation": [0, 0, 0]},
            {"model": "pinhole"},
            "requires a target",
        ),
        ({"kind": "world", "position": [0, 0, 0], "orientation": [0, 0, 0]}, {"model": "unknown"}, "projection model"),
    ],
)
def test_camera_schema_rejects_invalid_layers(mount, projection, message):
    with pytest.raises(ValueError, match=message):
        CameraSpec(
            "camera", "base", "openarm_base", "pinhole",
            {"stream_resolution": [640, 480], "fps": 30, "diagonal_fov_deg": 140},
            mount, projection,
        )
