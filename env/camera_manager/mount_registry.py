"""Mount resolution and pose composition for normalized camera rigs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation


def orientation_quaternion(value) -> np.ndarray:
    """Return a scalar-first quaternion from XYZ degrees or scalar-first input."""
    value = np.asarray(value, dtype=np.float64)
    if value.shape == (4,):
        return value
    if value.shape != (3,):
        raise ValueError(f"orientation must have 3 or 4 values, got {value.shape}")
    xyzw = Rotation.from_euler("XYZ", value, degrees=True).as_quat()
    return xyzw[[3, 0, 1, 2]]


def apply_optical_roll(orientation, roll_deg: float) -> np.ndarray:
    """Compose a physical local optical-axis (+Z) roll onto a mount pose."""
    base = orientation_quaternion(orientation)
    base_rotation = Rotation.from_quat(base[[1, 2, 3, 0]])
    local_roll = Rotation.from_euler("Z", float(roll_deg), degrees=True)
    xyzw = (base_rotation * local_roll).as_quat()
    return xyzw[[3, 0, 1, 2]]


def compose_pose(parent_position, parent_orientation, local_position, local_orientation):
    """Compose parent and local poses, returning position and scalar-first quaternion."""
    parent_q = orientation_quaternion(parent_orientation)
    local_q = orientation_quaternion(local_orientation)
    parent_rotation = Rotation.from_quat(parent_q[[1, 2, 3, 0]])
    local_rotation = Rotation.from_quat(local_q[[1, 2, 3, 0]])
    position = np.asarray(parent_position, dtype=np.float64) + parent_rotation.apply(local_position)
    xyzw = (parent_rotation * local_rotation).as_quat()
    return position, xyzw[[3, 0, 1, 2]]


@dataclass
class CameraMountRegistry:
    scene_manager: Any
    robot_manager: Any

    def resolve_parent_path(self, env_id: int, camera: Mapping[str, Any]) -> str:
        env_root = f"/World/envs/env_{env_id}"
        kind = camera.get("mount_kind", "world")
        target = camera.get("mount_target")
        if kind == "world":
            return env_root
        if kind == "robot_link":
            if not target:
                raise ValueError("robot_link camera mount requires mount_target")
            return self.robot_manager.resolve_camera_link_mount(env_id, target)
        if kind != "scene_fixture":
            raise ValueError(f"unsupported camera mount kind: {kind}")
        if not target:
            raise ValueError("scene_fixture camera mount requires a fixture label")
        return self.scene_manager.resolve_camera_fixture_mount(env_id, target)
