import json
from pathlib import Path

import yaml

from robodojo.sim.environment.camera_manager.mount_registry import CameraMountRegistry
from robodojo.sim.environment.camera_manager.rig_spec import hardware_camera_parent, normalize_camera_rig

ROOT = Path(__file__).resolve().parents[1]


def load_rig():
    config = yaml.safe_load((ROOT / "configs/camera/openarm_cloth_folding.yml").read_text())
    return normalize_camera_rig(config)


def test_openarm_timing_and_dimensions():
    env_cfg = yaml.safe_load((ROOT / "configs/openarm_cloth_folding.yml").read_text())
    sim_cfg = yaml.safe_load((ROOT / "configs/sim/openarm_cloth_folding.yml").read_text())
    robot_info = json.loads((ROOT / "configs/robot/_robot_info.json").read_text())["dual_openarm"]

    assert env_cfg["config"]["camera"] == "openarm_cloth_folding"
    assert env_cfg["observation"]["collect_freq"] == 30
    assert 1.0 / (sim_cfg["dt"] * 30) == 8.0
    assert sum(robot_info["arm_dim"]) + sum(robot_info["ee_dim"]) == 16


def test_openarm_uses_the_canonical_dyna_camera_rig():
    rig = load_rig()

    assert rig.profile_id == "openarm"
    assert [camera.observation_key for camera in rig.cameras] == [
        "cam_head",
        "cam_left_wrist",
        "cam_right_wrist",
    ]
    base, left, right = rig.cameras
    assert base.sensor["vendor"] == "Waveshare"
    assert base.sensor["model"] == "OV2710_2MP_USB_Camera_A_SKU_14121"
    assert base.sensor["stream_resolution"] == [640, 480]
    assert base.sensor["diagonal_fov_deg"] == 145.0
    assert base.projection["fx"] == base.projection["fy"] == 316.1146
    assert base.mount["kind"] == "scene_fixture"
    assert base.mount["target"] == "camera_stand"
    assert left.sensor["vendor"] == right.sensor["vendor"] == "Arducam"
    assert left.sensor["stream_resolution"] == right.sensor["stream_resolution"] == [1280, 720]
    assert left.mount["target"] == "robot0/left_wrist_camera_holder"
    assert right.mount["target"] == "robot0/right_wrist_camera_holder"
    assert left.mount["position"] == [0.05, 0.0, 0.12]
    assert right.mount["position"] == [0.035, 0.0, 0.12]
    assert left.mount["orientation"] == right.mount["orientation"] == [180.0, 0.0, -90.0]


def test_openarm_asset_inputs_and_mounts_are_pinned():
    sources = json.loads((ROOT / "configs/tooling/openarm/sources.json").read_text())
    robot_config = yaml.safe_load((ROOT / "configs/tooling/openarm/robot_config.yml").read_text())

    assert sources["openarm_isaac_lab"]["revision"] == "bad82e23716e6941c2de78ccb978f57c78b37734"
    assert sources["hardware_modifications"]["revision"] == "ffe34b93c070343042eb9412fbfeffce16139947"
    hashes = sources["hardware_modifications"]["sha256"]
    assert hashes["head camera holder v4.stl"] == "959ae5e0ad6e0870465e361df30db3d1bbdeebb9ba8001274c3ce9e1712f03d3"
    assert hashes["arducam_holder.stl"] == "1d31e0ac9ac2b118fb0925dc45bb3736dff087a9e6c2f9c27e64b24ee488074c"
    assert robot_config["left"]["camera_mount_links"]["left_wrist_camera_holder"] == "openarm_left_link7"
    assert robot_config["right"]["camera_mount_links"]["right_wrist_camera_holder"] == "openarm_right_link7"


def test_named_hardware_camera_frames_are_relative():
    assert hardware_camera_parent("/World/Holder", "OpticalFrame") == "/World/Holder/OpticalFrame"
    runtime = load_rig().runtime_config()
    for key in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        assert runtime[key].camera.mount_hardware_camera_frame == "OpticalFrame"


def test_openarm_disables_only_wrist_holder_construction():
    runtime = load_rig().runtime_config()

    assert runtime.cam_head.camera.mount_hardware_enabled is True
    assert runtime.cam_head.camera.mount_hardware_asset.endswith("head_camera_holder.usd")
    assert runtime.cam_left_wrist.camera.mount_hardware_enabled is False
    assert runtime.cam_right_wrist.camera.mount_hardware_enabled is False
    assert runtime.cam_left_wrist.camera.mount_hardware_asset.endswith("left_wrist_camera_holder.usd")
    assert runtime.cam_right_wrist.camera.mount_hardware_asset.endswith("right_wrist_camera_holder.usd")


def test_mount_registry_resolves_scene_and_robot_targets():
    class Scene:
        def resolve_camera_fixture_mount(self, env_id, label):
            return f"/fixture/{env_id}/{label}"

    class Robot:
        def resolve_camera_link_mount(self, env_id, target):
            return f"/robot/{env_id}/{target}"

    registry = CameraMountRegistry(Scene(), Robot())
    assert (
        registry.resolve_parent_path(2, {"mount_kind": "scene_fixture", "mount_target": "stand"}) == "/fixture/2/stand"
    )
    assert (
        registry.resolve_parent_path(2, {"mount_kind": "robot_link", "mount_target": "arm/link"}) == "/robot/2/arm/link"
    )


def test_legacy_camera_normalization_is_unchanged():
    config = yaml.safe_load((ROOT / "configs/camera/camera_config.yml").read_text())
    rig = normalize_camera_rig(config)

    assert rig.profile_id == "legacy"
    assert rig.runtime_config().default_frequency == config.get("default_frequency", 30)
    assert "cam_head" in rig.runtime_config()
