"""Mount resolution and pose composition for normalized camera rigs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

VALID_MOUNT_POSE_CONVENTIONS = frozenset({"isaac_usd", "sapien_robotics"})
SAPIEN_ROBOTICS_TO_ISAAC_USD = np.asarray([0.5, 0.5, -0.5, -0.5], dtype=np.float64)


def robot_link_prim_path(env_id: int, robot_mount_name: str, link: str) -> str:
    """Build an environment prim path while allowing a nested logical link."""
    if not link or link.startswith("/") or ".." in link.split("/"):
        raise ValueError(f"invalid robot camera mount link: {link!r}")
    return f"/World/envs/env_{int(env_id)}/{robot_mount_name}/{link.strip('/')}"


def require_camera_mount_prim(parent_path: str, is_valid) -> None:
    """Fail early when a logical mount resolves to an absent asset prim."""
    if not is_valid(parent_path):
        raise ValueError(
            f"resolved camera mount prim {parent_path} does not exist; "
            "rebuild the embodiment asset if it publishes a generated camera frame"
        )


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


def convert_mount_orientation(orientation, pose_convention: str = "isaac_usd") -> np.ndarray:
    """Convert a scalar-first mount quaternion to Isaac/USD camera axes.

    SAPIEN robotics cameras use +X forward, +Y left, and +Z up. Isaac/USD
    cameras use -Z forward and +Y up. The fixed local rotation is composed on
    the right so the configured pose remains the upstream mount pose.
    """
    if pose_convention not in VALID_MOUNT_POSE_CONVENTIONS:
        raise ValueError(f"unsupported camera mount pose convention: {pose_convention!r}")
    base = orientation_quaternion(orientation)
    if pose_convention == "isaac_usd":
        return base
    base_rotation = Rotation.from_quat(base[[1, 2, 3, 0]])
    axes_rotation = Rotation.from_quat(SAPIEN_ROBOTICS_TO_ISAAC_USD[[1, 2, 3, 0]])
    xyzw = (base_rotation * axes_rotation).as_quat()
    return xyzw[[3, 0, 1, 2]]


def mount_orientation(orientation, pose_convention: str = "isaac_usd", optical_roll_deg: float = 0.0) -> np.ndarray:
    """Convert camera axes, then apply the configured local optical roll."""
    converted = convert_mount_orientation(orientation, pose_convention)
    return apply_optical_roll(converted, optical_roll_deg)


def apply_mount_calibration(position, orientation, translation_m, rotation_rotvec_deg):
    """Apply a parent-frame extrinsic correction after axis/roll conversion."""
    position = np.asarray(position, dtype=np.float64)
    translation = np.asarray(translation_m, dtype=np.float64)
    rotation_vector = np.asarray(rotation_rotvec_deg, dtype=np.float64)
    if position.shape != (3,) or translation.shape != (3,) or rotation_vector.shape != (3,):
        raise ValueError("camera mount calibration requires three translation and rotation values")
    if not np.all(np.isfinite(np.concatenate((position, translation, rotation_vector)))):
        raise ValueError("camera mount calibration must be finite")
    base = orientation_quaternion(orientation)
    base_rotation = Rotation.from_quat(base[[1, 2, 3, 0]])
    delta = Rotation.from_rotvec(np.deg2rad(rotation_vector))
    xyzw = (delta * base_rotation).as_quat()
    return position + translation, xyzw[[3, 0, 1, 2]]


def compose_pose(parent_position, parent_orientation, local_position, local_orientation):
    """Compose parent and local poses, returning position and scalar-first quaternion."""
    parent_q = orientation_quaternion(parent_orientation)
    local_q = orientation_quaternion(local_orientation)
    parent_rotation = Rotation.from_quat(parent_q[[1, 2, 3, 0]])
    local_rotation = Rotation.from_quat(local_q[[1, 2, 3, 0]])
    position = np.asarray(parent_position, dtype=np.float64) + parent_rotation.apply(local_position)
    xyzw = (parent_rotation * local_rotation).as_quat()
    return position, xyzw[[3, 0, 1, 2]]


def pose_matrix(position, orientation) -> np.ndarray:
    """Return a conventional column-vector homogeneous pose matrix."""
    quaternion = orientation_quaternion(orientation)
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = np.asarray(position, dtype=np.float64)
    return matrix


def pose_from_matrix(matrix) -> tuple[np.ndarray, np.ndarray]:
    """Return position and scalar-first quaternion from a pose matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"pose matrix must be 4x4, got {matrix.shape}")
    rotation = Rotation.from_matrix(matrix[:3, :3])
    xyzw = rotation.as_quat()
    return matrix[:3, 3].copy(), xyzw[[3, 0, 1, 2]]


def align_hardware_frame_pose(target_position, target_orientation, frame_matrix):
    """Derive parent->hardware so parent->named-frame equals the target pose."""
    target = pose_matrix(target_position, target_orientation)
    frame = np.asarray(frame_matrix, dtype=np.float64)
    if frame.shape != (4, 4):
        raise ValueError(f"hardware frame matrix must be 4x4, got {frame.shape}")
    if not np.allclose(frame[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("hardware frame matrix is not homogeneous")
    hardware = target @ np.linalg.inv(frame)
    return pose_from_matrix(hardware)


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
