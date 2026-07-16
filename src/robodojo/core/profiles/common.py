"""Shared profile loading helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def profile_path(config_root: Path, relative: str, *, field: str) -> Path:
    root = config_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{field} must stay below {root}: {relative}")
    if not path.is_file():
        raise ValueError(f"{field} not found: {path}")
    return path


def merge_profile_payload(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_profile_payload(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged
