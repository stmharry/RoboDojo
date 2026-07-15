from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np

FIXTURE_APPEARANCE_KEYS = ("visual_color", "replay_material_override")


def normalize_rgb_color(value: Any, *, field: str = "visual color") -> np.ndarray:
    """Return a finite float32 RGB color with one validation contract."""

    color = np.asarray(value, dtype=np.float32)
    if color.shape != (3,) or not np.isfinite(color).all() or ((color < 0.0) | (color > 1.0)).any():
        raise ValueError(f"{field} must contain three finite values in [0, 1]")
    return color


def merge_fixture_appearance(saved_fixture: dict[str, Any], active_fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay profile-owned appearance onto a replayed fixture contract.

    Evaluation layouts persist fixture geometry and physics so they can be
    replayed exactly. Appearance, however, belongs to the active scene
    profile. Keeping this overlay deliberately narrow prevents a scene skin
    from changing a saved table or room transform.
    """
    merged = deepcopy(saved_fixture)
    for key in FIXTURE_APPEARANCE_KEYS:
        if key in active_fixture:
            merged[key] = deepcopy(active_fixture[key])
        else:
            merged.pop(key, None)
    return merged
