"""Canonical runtime identity fields for scene results, resumes, and exports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SCENE_IDENTITY_FIELDS = (
    "recipe_name",
    "contract_hash",
    "protocol_name",
    "task_name",
    "layout_name",
    "episode_horizon",
    "native_eval_num",
    "environment_profile_hash",
    "policy_contract",
    "scene_config",
    "scene_component",
    "scene_profile_hash",
    "layout_config_name",
    "layout_source",
    "layout_set_hash",
    "scene_asset_hash",
    "scene_asset_builds",
    "scene_asset_identities",
)


def _plain_value(value: Any) -> Any:
    """Detach identity data from OmegaConf and other runtime containers."""

    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_value(item) for item in value]
    return value


def scene_identity(values: Mapping[str, Any]) -> dict[str, Any]:
    return {field: _plain_value(values.get(field)) for field in SCENE_IDENTITY_FIELDS}


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
