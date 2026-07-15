"""Canonical runtime identity fields for scene results, resumes, and exports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SCENE_IDENTITY_FIELDS = (
    "environment_profile_hash",
    "policy_contract",
    "scene_config",
    "scene_component",
    "scene_profile_hash",
    "layout_config_name",
    "layout_source",
    "layout_set_hash",
    "scene_asset_hash",
)


def scene_identity(values: Mapping[str, Any]) -> dict[str, Any]:
    return {field: values.get(field) for field in SCENE_IDENTITY_FIELDS}


def require_matching_scene_identity(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    context: str,
) -> None:
    for field, expected_value in scene_identity(expected).items():
        actual_value = actual.get(field)
        if actual_value != expected_value:
            raise ValueError(f"{context} {field} mismatch: expected {expected_value!r}, found {actual_value!r}")
