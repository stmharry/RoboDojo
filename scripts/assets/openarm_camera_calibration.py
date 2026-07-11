"""Pinned CAD measurements for the LeRobot OpenARM camera holders.

Coordinates are millimetres in the source STL frames.  The values are derived
from planar facet boundaries and circular mounting/lens apertures; no episode
pixels participate in these transforms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BLOG_SPACE_REVISION = "170e1d479579e0b4be1afe0c99ebf868b24803db"
HARDWARE_REVISION = "ffe34b93c070343042eb9412fbfeffce16139947"
HEAD_FIXTURE_SENSOR_POSITION_M = (0.0, -0.31855376, 0.05106626)
HEAD_FIXTURE_SENSOR_ORIENTATION_DEG = (120.0, 0.0, 0.0)
WRIST_LINK_SENSOR_POSES = {
    "left": {"position_m": (0.05, 0.0, 0.12), "orientation_deg": (180.0, 0.0, 90.0)},
    "right": {"position_m": (0.035, 0.0, 0.12), "orientation_deg": (180.0, 0.0, 90.0)},
}


@dataclass(frozen=True)
class HolderCalibration:
    filename: str
    mount_origin_mm: tuple[float, float, float]
    optical_origin_mm: tuple[float, float, float]
    optical_direction_cad: tuple[float, float, float]
    cad_to_mount: tuple[tuple[float, float, float], ...]

    def transformed_points_m(self, points) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        rotation = np.asarray(self.cad_to_mount, dtype=np.float64)
        origin = np.asarray(self.mount_origin_mm, dtype=np.float64)
        return (points - origin) @ rotation.T * 0.001

    def optical_position_m(self) -> np.ndarray:
        return self.transformed_points_m([self.optical_origin_mm])[0]

    def optical_direction_mount(self) -> np.ndarray:
        direction = np.asarray(self.cad_to_mount) @ np.asarray(self.optical_direction_cad)
        return direction / np.linalg.norm(direction)


# The bracket's 45 mm top plate mounts to the camera-stand tip.  CAD +X points
# behind the stand, +Y spans the extrusion, and +Z points upward.  The mapping
# below expresses the holder in the fixture frame: CAD X->fixture -Z,
# CAD Y->fixture +X, CAD Z->fixture -Y.
HEAD_CAD_TO_FIXTURE = np.asarray(((0.0, 1.0, 0.0), (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0)))
HEAD_HOLDER = HolderCalibration(
    filename="head camera holder v4.stl",
    mount_origin_mm=(-17.56719649, 41.0, 262.79797363),
    optical_origin_mm=(49.40184321, 41.0, 19.09895447),
    optical_direction_cad=(-0.3420201433, 0.0, -0.9396926208),
    cad_to_mount=tuple(tuple(float(value) for value in row) for row in HEAD_CAD_TO_FIXTURE),
)

# The wrist anchor is the mean of the four 3.4 mm mounting-hole centers.  Its
# X axis spans the gripper; Y follows the bracket arms; Z is the mounting-plane
# normal.  This normalized frame is mirrored at attachment time for the right
# hand, not inferred from camera pixels.
WRIST_MOUNT_ORIGIN_MM = (12.75, -42.62586212, -42.55709187)
WRIST_X_CAD = np.asarray((1.0, 0.0, 0.0))
WRIST_Z_CAD = np.asarray((0.0, -0.7660444431, 0.6427876097))
WRIST_Y_CAD = np.cross(WRIST_Z_CAD, WRIST_X_CAD)
WRIST_CAD_TO_ANCHOR = np.vstack((WRIST_X_CAD, WRIST_Y_CAD, WRIST_Z_CAD))
WRIST_HOLDER = HolderCalibration(
    filename="arducam_holder.stl",
    mount_origin_mm=WRIST_MOUNT_ORIGIN_MM,
    optical_origin_mm=(12.75, -6.21393824, 14.26977779),
    # The camera board sits on the +Y side of the mounting plane, while its
    # lens looks back through the holder aperture toward CAD -Y.
    optical_direction_cad=(0.0, -1.0, 0.0),
    cad_to_mount=tuple(tuple(float(value) for value in row) for row in WRIST_CAD_TO_ANCHOR),
)


def head_points_m(points) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return (points - np.asarray(HEAD_HOLDER.mount_origin_mm)) @ HEAD_CAD_TO_FIXTURE.T * 0.001


def head_optical_position_m() -> np.ndarray:
    return head_points_m([HEAD_HOLDER.optical_origin_mm])[0]


def head_optical_direction_fixture() -> np.ndarray:
    direction = HEAD_CAD_TO_FIXTURE @ np.asarray(HEAD_HOLDER.optical_direction_cad)
    return direction / np.linalg.norm(direction)


def wrist_points_m(points, side: str) -> np.ndarray:
    normalized = WRIST_HOLDER.transformed_points_m(points)
    if side == "right":
        normalized[:, 0] *= -1.0
    elif side != "left":
        raise ValueError(f"unknown OpenARM side: {side}")
    return normalized


def calibration_manifest() -> dict:
    return {
        "blog_space_revision": BLOG_SPACE_REVISION,
        "hardware_revision": HARDWARE_REVISION,
        "head": {
            "mount_origin_mm": list(HEAD_HOLDER.mount_origin_mm),
            "optical_origin_mm": list(HEAD_HOLDER.optical_origin_mm),
            "optical_position_m": head_optical_position_m().tolist(),
            "optical_direction_fixture": head_optical_direction_fixture().tolist(),
            "runtime_sensor_position_m": list(HEAD_FIXTURE_SENSOR_POSITION_M),
            "runtime_sensor_orientation_deg": list(HEAD_FIXTURE_SENSOR_ORIENTATION_DEG),
        },
        "wrist": {
            "mount_origin_mm": list(WRIST_HOLDER.mount_origin_mm),
            "optical_origin_mm": list(WRIST_HOLDER.optical_origin_mm),
            "optical_position_m": WRIST_HOLDER.optical_position_m().tolist(),
            "optical_direction_anchor": WRIST_HOLDER.optical_direction_mount().tolist(),
            "right_is_mirrored": True,
            "runtime_link_sensor_poses": {
                side: {key: list(value) for key, value in pose.items()}
                for side, pose in WRIST_LINK_SENSOR_POSES.items()
            },
        },
    }
