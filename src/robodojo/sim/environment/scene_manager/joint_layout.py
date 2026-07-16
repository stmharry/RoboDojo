"""Validation for articulation joint values stored in saved layouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def resolve_initial_joint_positions(
    current_positions: Sequence[float],
    joint_names: Sequence[str],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    configured_positions: Mapping[str, float] | None,
) -> np.ndarray:
    """Overlay validated name-keyed layout joint values on an asset pose."""
    positions = np.asarray(current_positions, dtype=np.float64).reshape(-1).copy()
    names = list(joint_names)
    lower = np.asarray(lower_limits, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper_limits, dtype=np.float64).reshape(-1)
    if not (positions.size == len(names) == lower.size == upper.size):
        raise ValueError("articulation joint names, positions, and limits must have equal lengths")
    if configured_positions is None:
        return positions
    if not isinstance(configured_positions, Mapping):
        raise ValueError("initial_joint_positions must be a joint-name-to-radians mapping")

    index_by_name = {name: index for index, name in enumerate(names)}
    for name, raw_value in configured_positions.items():
        if name not in index_by_name:
            raise ValueError(f"initial joint {name!r} is not present in articulation joints {names}")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"initial joint {name!r} must be a finite radian value")
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(f"initial joint {name!r} must be a finite radian value")
        index = index_by_name[name]
        if value < lower[index] or value > upper[index]:
            raise ValueError(
                f"initial joint {name!r}={value} is outside limits [{lower[index]}, {upper[index]}] radians"
            )
        positions[index] = value
    return positions
