"""Validation helpers for active saved-layout object poses."""

from __future__ import annotations

import numpy as np


def validate_active_pose(inst_name: str, position, *, offscreen_threshold: float = 50000.0) -> None:
    if position is None:
        raise RuntimeError(f"active layout object {inst_name!r} has no physics wrapper")
    pose = np.asarray(position, dtype=float).reshape(-1)
    if pose.size < 3 or not np.all(np.isfinite(pose[:3])):
        raise RuntimeError(f"active layout object {inst_name!r} has a non-finite pose")
    if np.max(np.abs(pose[:3])) >= offscreen_threshold:
        raise RuntimeError(f"active layout object {inst_name!r} resolved to the offscreen pool")
