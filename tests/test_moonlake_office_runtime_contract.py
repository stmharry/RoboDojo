from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles import load_scene_profile
from robodojo.sim.environment.camera_manager.mount_registry import CameraMountRegistry
from robodojo.sim.environment.camera_manager.rig_spec import normalize_camera_rig
from robodojo.sim.environment.robot_manager.mount_spec import (
    apply_robot_mount_override,
    normalize_robot_mount_overrides,
)

ROOT = Path(__file__).resolve().parents[1]
PATHS = RepositoryPaths.resolve(ROOT)


def test_scene_camera_override_changes_only_the_head_mount():
    scene = load_scene_profile(PATHS, "moonlake_office")
    camera_config = yaml.safe_load((ROOT / "configs/camera/bimanual_yam.yml").read_text(encoding="utf-8"))
    overrides = scene.document.mounts.model_dump(mode="python", exclude_none=True)["cameras"]

    rig = normalize_camera_rig(camera_config, mount_overrides=overrides)
    cameras = {camera.observation_key: camera.runtime_camera() for camera in rig.cameras}
    assert cameras["cam_head"]["mount_kind"] == "scene_fixture"
    assert cameras["cam_head"]["mount_target"] == "moonlake_office_fixture"
    assert cameras["cam_head"]["mount_frame"] == "Mounts/D435OpticalFrame"
    assert cameras["cam_head"]["pos"] == [0.0, 0.0, 0.0]
    assert cameras["cam_head"]["ori"] == [1.0, 0.0, 0.0, 0.0]
    assert cameras["cam_left_wrist"]["mount_kind"] == "robot_link"
    assert cameras["cam_right_wrist"]["mount_kind"] == "robot_link"
    assert cameras["cam_head"]["stream_resolution"] == [640, 360]
    assert cameras["cam_head"]["fx"] == pytest.approx(462.1386898729645)

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


def test_robot_mount_override_updates_runtime_pose_without_mutating_config():
    original = [-0.24, -0.45, 0.765, 0.0, 0.0, 0.0, 1.0]
    robot = SimpleNamespace(
        default_root_pos=original[:3],
        default_root_rot=original[3:],
        entity_origin_pose=list(original),
    )
    override = {
        "position": [-0.2032, -0.32, 0.77],
        "orientation": [0.7071067811865476, 0.0, 0.0, 0.7071067811865476],
    }

    pose = apply_robot_mount_override(robot, override)

    assert pose == override["position"] + override["orientation"]
    assert robot.entity_origin_pose == pose
    assert original == [-0.24, -0.45, 0.765, 0.0, 0.0, 0.0, 1.0]

    with pytest.raises(ValueError, match="unknown slots"):
        normalize_robot_mount_overrides({"robot0": override}, robot_count=0)
