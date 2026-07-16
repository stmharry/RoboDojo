import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles.environment import load_environment_profile
from robodojo.sim.environment.camera_manager.mount_registry import CameraMountRegistry
from robodojo.sim.environment.camera_manager.rig_spec import normalize_camera_rig

ROOT = Path(__file__).resolve().parents[1]
PATHS = RepositoryPaths.resolve(ROOT)


def test_moonlake_setup_owns_head_mount_and_dark_wrist_housings():
    profile = load_environment_profile(PATHS, "bimanual_yam_moonlake_office")
    camera_config = yaml.safe_load(profile.component_paths["camera"].read_text(encoding="utf-8"))
    rig = normalize_camera_rig(camera_config)
    cameras = {camera.observation_key: camera.runtime_camera() for camera in rig.cameras}
    assert cameras["cam_head"]["mount_kind"] == "scene_fixture"
    assert cameras["cam_head"]["mount_target"] == "moonlake_office_fixture"
    assert cameras["cam_head"]["mount_frame"] == "Mounts/D435OpticalFrame"
    assert cameras["cam_head"]["pos"] == [0.0, 0.0, 0.0]
    assert cameras["cam_head"]["ori"] == [1.0, 0.0, 0.0, 0.0]
    assert cameras["cam_head"]["near_clip_m"] == pytest.approx(0.1)
    assert cameras["cam_left_wrist"]["mount_kind"] == "robot_link"
    assert cameras["cam_right_wrist"]["mount_kind"] == "robot_link"
    assert cameras["cam_left_wrist"].get("near_clip_m") is None
    assert cameras["cam_right_wrist"].get("near_clip_m") is None
    assert cameras["cam_head"]["stream_resolution"] == [640, 480]
    assert cameras["cam_left_wrist"]["stream_resolution"] == [640, 480]
    assert cameras["cam_right_wrist"]["stream_resolution"] == [640, 480]
    assert cameras["cam_head"]["cy"] == pytest.approx(240.0)
    assert cameras["cam_head"]["fx"] == pytest.approx(462.1386898729645)
    assert cameras["cam_left_wrist"]["mount_hardware_asset"].endswith("D405_proxy_moonlake_office.usd")

    with pytest.raises(ValueError, match="unknown cameras"):
        normalize_camera_rig(
            camera_config,
            mount_overrides={
                "missing": {
                    "kind": "world",
                    "position": [0, 0, 0],
                    "orientation": [1, 0, 0, 0],
                }
            },
        )


def test_camera_mount_registry_forwards_the_named_fixture_frame():
    calls = []

    class Scene:
        def resolve_camera_fixture_mount(self, env_id, target, frame):
            calls.append((env_id, target, frame))
            return "/World/envs/env_2/geometry/fixture/Mounts/D435OpticalFrame"

    registry = CameraMountRegistry(Scene(), SimpleNamespace())
    result = registry.resolve_parent_path(
        2,
        {
            "mount_kind": "scene_fixture",
            "mount_target": "moonlake_office_fixture",
            "mount_frame": "Mounts/D435OpticalFrame",
        },
    )
    assert result.endswith("/Mounts/D435OpticalFrame")
    assert calls == [(2, "moonlake_office_fixture", "Mounts/D435OpticalFrame")]


def test_named_setups_own_distinct_robot_roots_and_visual_assets():
    classic = load_environment_profile(PATHS, "bimanual_yam_molmoact2")
    office = load_environment_profile(PATHS, "bimanual_yam_moonlake_office")
    classic_robots = yaml.safe_load(classic.component_paths["robot"].read_text(encoding="utf-8"))["robots"]
    office_robots = yaml.safe_load(office.component_paths["robot"].read_text(encoding="utf-8"))["robots"]

    assert classic_robots[0]["default_root_pos"] == [-0.24, -0.45, 0.765]
    assert office_robots[0]["default_root_pos"] == [-0.24, -0.40, 0.75]
    assert {robot["usd_asset"] for robot in classic_robots} == {"YAM_molmoact2.usd"}
    assert {robot["usd_asset"] for robot in office_robots} == {"YAM_moonlake_office.usd"}


def test_moonlake_pickup_preserves_table_local_robot_and_target_offsets():
    classic_layout = json.loads((ROOT / "configs/layout/molmo_yam/0/general_pickup_0.json").read_text())
    office_layout = json.loads((ROOT / "configs/layout/moonlake_office/0/general_pickup_0.json").read_text())
    classic_profile = load_environment_profile(PATHS, "bimanual_yam_molmoact2")
    office_profile = load_environment_profile(PATHS, "bimanual_yam_moonlake_office")
    classic_robot = yaml.safe_load(classic_profile.component_paths["robot"].read_text())["robots"][1]
    office_robot = yaml.safe_load(office_profile.component_paths["robot"].read_text())["robots"][1]

    translation = [
        office_robot["default_root_pos"][axis] - classic_robot["default_root_pos"][axis] for axis in range(3)
    ]
    classic_target = classic_layout["Rigid"]["ball"][0]["default_pos"]
    office_target = office_layout["Rigid"]["ball"][0]["default_pos"]
    classic_table = classic_layout["Table"]["default_pos"]
    office_table = office_layout["Table"]["default_pos"]

    assert translation == pytest.approx([0.0, 0.05, -0.015])
    assert office_target == pytest.approx([classic_target[axis] + translation[axis] for axis in range(3)])
    assert office_table == pytest.approx([classic_table[axis] + translation[axis] for axis in range(3)])
