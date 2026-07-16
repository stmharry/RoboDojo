"""Build a provenance-preserving Isaac USD for the I2RT YAM arm."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

ARM_MESH_NAMES = ("base.stl", "link1.stl", "link2.stl", "link3.stl", "link4.stl", "link5.stl")
GRIPPER_MESH_NAMES = ("gripper.stl", "tip_left.stl", "tip_right.stl")
ARM_JOINT_NAMES = tuple(f"dof_joint{index}" for index in range(1, 7))
GRIPPER_JOINT_NAMES = ("dof_joint7", "dof_joint8")
FINGER_LOWER_LIMIT_M = -0.0475
PREVIEW_MATERIAL_KEYS = ("diffuse_color", "roughness", "metallic", "opacity")

logger = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
