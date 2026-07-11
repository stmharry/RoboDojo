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


@dataclass(frozen=True)
class HolderCalibration:
    filename: str
    mount_origin_mm: tuple[float, float, float]
    optical_origin_mm: tuple[float, float, float]
    optical_direction_cad: tuple[float, float, float]
    optical_up_cad: tuple[float, float, float]
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

    def optical_up_mount(self) -> np.ndarray:
        up = np.asarray(self.cad_to_mount) @ np.asarray(self.optical_up_cad)
        return up / np.linalg.norm(up)

    def optical_frame_matrix(self) -> np.ndarray:
        """Return a right-handed camera frame (+X right, +Y up, -Z look)."""
        look = self.optical_direction_mount()
        up_hint = self.optical_up_mount()
        right = np.cross(look, up_hint)
        right /= np.linalg.norm(right)
        up = np.cross(right, look)
        up /= np.linalg.norm(up)
        rotation = np.column_stack((right, up, -look))
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9):
            raise ValueError("camera optical basis is not orthonormal")
        if np.linalg.det(rotation) < 0.999999:
            raise ValueError("camera optical basis is not right-handed")
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = self.optical_position_m()
        return matrix


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
    # The orthogonal long edge of the holder fixes image-up.  Together with
    # the aperture normal this also captures that the board is installed
    # upside-down in the bracket, without an image-space roll override.
    optical_up_cad=(0.9396926208, 0.0, -0.3420201433),
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
    # The lens aperture opens toward CAD +Y. The previous -Y interpretation
    # looked back into the bracket and hand rather than toward the contact
    # region below the jaw.
    optical_direction_cad=(0.0, 1.0, 0.0),
    # The asymmetric mounting-hole/connector direction fixes landscape up.
    optical_up_cad=(1.0, 0.0, 0.0),
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


def holder_optical_frame(side: str) -> np.ndarray:
    if side == "head":
        return HEAD_HOLDER.optical_frame_matrix()
    if side not in ("left", "right"):
        raise ValueError(f"unknown holder side: {side}")
    frame = WRIST_HOLDER.optical_frame_matrix()
    if side == "right":
        # A reflection cannot be authored as a rigid transform. Mirror the
        # physical aperture position, then rotate the image basis 180 degrees
        # about the optical axis to preserve a right-handed physical frame.
        frame[0, 3] *= -1.0
        frame[:3, 0] *= -1.0
        frame[:3, 1] *= -1.0
    return frame


def calibration_manifest() -> dict:
    return {
        "blog_space_revision": BLOG_SPACE_REVISION,
        "hardware_revision": HARDWARE_REVISION,
        "head": {
            "mount_origin_mm": list(HEAD_HOLDER.mount_origin_mm),
            "optical_origin_mm": list(HEAD_HOLDER.optical_origin_mm),
            "optical_position_m": head_optical_position_m().tolist(),
            "optical_direction_fixture": head_optical_direction_fixture().tolist(),
            "optical_up_fixture": HEAD_HOLDER.optical_up_mount().tolist(),
            "optical_frame_matrix": holder_optical_frame("head").tolist(),
        },
        "wrist": {
            "mount_origin_mm": list(WRIST_HOLDER.mount_origin_mm),
            "optical_origin_mm": list(WRIST_HOLDER.optical_origin_mm),
            "optical_position_m": WRIST_HOLDER.optical_position_m().tolist(),
            "optical_direction_anchor": WRIST_HOLDER.optical_direction_mount().tolist(),
            "optical_up_anchor": WRIST_HOLDER.optical_up_mount().tolist(),
            "left_optical_frame_matrix": holder_optical_frame("left").tolist(),
            "right_optical_frame_matrix": holder_optical_frame("right").tolist(),
            "right_is_mirrored": True,
        },
    }
