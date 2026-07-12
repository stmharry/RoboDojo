"""Validation for hardware-backed embodiment calibration manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_MEASURED_SECTIONS = ("identity", "sources", "robot", "cameras", "controller")
REQUIRED_ROBOT_FIELDS = ("link_geometry", "joint_zeros", "joint_limits", "gripper", "asset")
REQUIRED_CONTROLLER_FIELDS = (
    "latency_seconds",
    "interpolation",
    "target_clip_degrees",
    "damping",
    "tracking_error_degrees",
    "settled_pose_degrees",
)
REQUIRED_CAMERA_FIELDS = (
    "serial",
    "resolution",
    "intrinsics",
    "distortion_coefficients",
    "mount_extrinsics",
    "exposure",
    "white_balance",
    "focus",
)


def load_hardware_calibration(config_root: Path, name: str) -> dict[str, Any]:
    path = config_root / "calibration" / f"{name}.yml"
    if not path.is_file():
        raise ValueError(f"hardware calibration not found: {path}")
    payload: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if payload.get("schema_version") != 1:
        raise ValueError(f"hardware calibration schema_version must be 1: {path}")
    if payload.get("profile_id") != name:
        raise ValueError(f"hardware calibration profile_id must be {name}: {path}")
    if payload.get("status") != "measured":
        raise ValueError(
            f"hardware calibration is not release-ready: {name} "
            f"(status={payload.get('status', 'missing')})"
        )
    missing = [section for section in REQUIRED_MEASURED_SECTIONS if not payload.get(section)]
    if missing:
        raise ValueError(f"hardware calibration {name} is missing: {', '.join(missing)}")
    identity = payload["identity"]
    for field in ("vendor", "hardware_revision", "robot_serial", "calibration_date"):
        if not identity.get(field):
            raise ValueError(f"hardware calibration {name} identity.{field} is required")
    sources = payload["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"hardware calibration {name} requires source measurements")
    for source in sources:
        checksum = str(source.get("sha256", ""))
        if not source.get("path") or len(checksum) != 64:
            raise ValueError(f"hardware calibration {name} sources require path and sha256")
    for field in REQUIRED_ROBOT_FIELDS:
        if payload["robot"].get(field) is None:
            raise ValueError(f"hardware calibration {name} robot.{field} is required")
    cameras = payload["cameras"]
    if set(cameras) != {"base", "left_wrist", "right_wrist"}:
        raise ValueError(f"hardware calibration {name} requires base and two wrist cameras")
    for role, camera in cameras.items():
        for field in REQUIRED_CAMERA_FIELDS:
            if camera.get(field) is None:
                raise ValueError(f"hardware calibration {name} cameras.{role}.{field} is required")
    for field in REQUIRED_CONTROLLER_FIELDS:
        if payload["controller"].get(field) is None:
            raise ValueError(f"hardware calibration {name} controller.{field} is required")
    return payload


def calibration_name(environment_payload: dict[str, Any]) -> str | None:
    value = environment_payload.get("hardware_calibration")
    return str(value) if value else None
