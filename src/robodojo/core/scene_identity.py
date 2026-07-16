"""Canonical runtime identity fields for scene results, resumes, and exports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

ARTIFACT_SCHEMA_VERSION = 3


class ArtifactSchemaError(ValueError):
    """Raised when a result or resume belongs to an unsupported contract."""


SCENE_IDENTITY_FIELDS = (
    "artifact_schema_version",
    "recipe_name",
    "contract_hash",
    "protocol_name",
    "task_name",
    "episode_horizon",
    "native_eval_num",
    "robodojo_revision",
    "xpolicylab_revision",
    "policy_name",
    "policy_profile",
    "policy_checkpoint",
    "policy_descriptor_hash",
    "policy_reference_match",
    "policy_execution",
    "policy_training",
    "policy_adapter",
    "environment_profile",
    "environment_profile_hash",
    "environment_variant",
    "environment_asset_hash",
    "environment_asset_builds",
    "environment_asset_identities",
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


def require_current_artifact_schema(values: Mapping[str, Any], *, context: str) -> None:
    if "layout_name" in values:
        raise ArtifactSchemaError(f"{context} contains removed layout_name selector")
    actual = values.get("artifact_schema_version")
    if actual != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactSchemaError(
            f"{context} artifact_schema_version mismatch: expected {ARTIFACT_SCHEMA_VERSION}, found {actual!r}"
        )


def require_current_result_artifact(values: Mapping[str, Any], *, context: str) -> None:
    require_current_artifact_schema(values, context=context)
    required = (
        "task_name",
        "protocol_name",
        "episode_horizon",
        "native_eval_num",
        "robodojo_revision",
        "xpolicylab_revision",
        "policy_profile",
        "environment_profile",
        "environment_profile_hash",
        "environment_asset_hash",
        "policy_contract",
        "scene_config",
        "layout_config_name",
        "layout_source",
        "layout_set_hash",
    )
    missing = [field for field in required if values.get(field) in (None, "")]
    if missing:
        raise ArtifactSchemaError(f"{context} is missing required fields: {', '.join(missing)}")
    if values.get("policy_profile") != "manual" and not values.get("policy_descriptor_hash"):
        raise ArtifactSchemaError(f"{context} is missing policy_descriptor_hash for a tracked policy profile")
    try:
        eval_time = int(values.get("eval_time", 0))
    except (TypeError, ValueError) as exc:
        raise ArtifactSchemaError(f"{context} has invalid eval_time") from exc
    if eval_time < 1:
        raise ArtifactSchemaError(f"{context} is incomplete: eval_time={eval_time}")
    details = values.get("details")
    if not isinstance(details, Mapping) or len(details) < eval_time:
        raise ArtifactSchemaError(f"{context} has incomplete episode details")
    task_name = str(values["task_name"])
    for index, detail in details.items():
        if not isinstance(detail, Mapping):
            raise ArtifactSchemaError(f"{context} detail {index!r} is not an object")
        layout_id = detail.get("layout_id")
        layout_file = detail.get("layout_file")
        layout_hash = detail.get("layout_sha256")
        if not isinstance(layout_id, int) or layout_file != f"{task_name}_{layout_id}.json":
            raise ArtifactSchemaError(f"{context} detail {index!r} has invalid task-keyed layout identity")
        if not isinstance(layout_hash, str) or len(layout_hash) != 64:
            raise ArtifactSchemaError(f"{context} detail {index!r} has invalid layout_sha256")


def require_matching_scene_identity(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    context: str,
) -> None:
    require_current_artifact_schema(expected, context=f"expected {context}")
    require_current_artifact_schema(actual, context=context)
    for field, expected_value in scene_identity(expected).items():
        actual_value = actual.get(field)
        if actual_value != expected_value:
            raise ValueError(f"{context} {field} mismatch: expected {expected_value!r}, found {actual_value!r}")
