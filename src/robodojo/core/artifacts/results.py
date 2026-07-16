"""Evaluation result identity and historical result normalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

ARTIFACT_SCHEMA_VERSION = 4
LEGACY_ARTIFACT_SCHEMA_VERSION = 3


class ArtifactSchemaError(ValueError):
    """Raised when a result or resume belongs to an unsupported schema."""


LEGACY_FIELD_NAMES = {
    "recipe_name": "recipe",
    "contract_hash": "experiment_hash",
    "protocol_name": "task_protocol",
    "task_name": "task",
    "native_eval_num": "evaluation_episodes",
    "environment_profile": "environment",
    "policy_contract": "embodiment",
    "scene_config": "scene",
    "layout_config_name": "layout_set",
}

SCENE_IDENTITY_FIELDS = (
    "artifact_schema_version",
    "recipe",
    "experiment_hash",
    "task_protocol",
    "task",
    "episode_horizon",
    "evaluation_episodes",
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
    "environment",
    "environment_profile_hash",
    "environment_variant",
    "environment_asset_hash",
    "environment_asset_builds",
    "environment_asset_identities",
    "embodiment",
    "scene",
    "scene_component",
    "scene_profile_hash",
    "layout_set",
    "layout_source",
    "layout_set_hash",
    "scene_asset_hash",
    "scene_asset_builds",
    "scene_asset_identities",
)


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_value(item) for item in value]
    return value


def normalize_artifact(values: Mapping[str, Any], *, context: str = "artifact") -> dict[str, Any]:
    """Return a current in-memory view of a v3 or v4 artifact."""

    if "layout_name" in values:
        raise ArtifactSchemaError(f"{context} contains removed layout_name selector")
    version = values.get("artifact_schema_version")
    if version == ARTIFACT_SCHEMA_VERSION:
        return {str(key): _plain_value(value) for key, value in values.items()}
    if version == LEGACY_ARTIFACT_SCHEMA_VERSION:
        normalized = {LEGACY_FIELD_NAMES.get(str(key), str(key)): _plain_value(value) for key, value in values.items()}
        normalized["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
        return normalized
    raise ArtifactSchemaError(
        f"{context} artifact_schema_version mismatch: supported "
        f"{LEGACY_ARTIFACT_SCHEMA_VERSION}/{ARTIFACT_SCHEMA_VERSION}, found {version!r}"
    )


def scene_identity(values: Mapping[str, Any]) -> dict[str, Any]:
    normalized = normalize_artifact(values, context="scene identity")
    return {field: _plain_value(normalized.get(field)) for field in SCENE_IDENTITY_FIELDS}


def require_current_artifact_schema(values: Mapping[str, Any], *, context: str) -> None:
    normalize_artifact(values, context=context)


def require_current_result_artifact(values: Mapping[str, Any], *, context: str) -> None:
    normalized = normalize_artifact(values, context=context)
    required = (
        "task",
        "task_protocol",
        "episode_horizon",
        "evaluation_episodes",
        "robodojo_revision",
        "xpolicylab_revision",
        "policy_profile",
        "environment",
        "environment_profile_hash",
        "environment_asset_hash",
        "embodiment",
        "scene",
        "layout_set",
        "layout_source",
        "layout_set_hash",
    )
    missing = [field for field in required if normalized.get(field) in (None, "")]
    if missing:
        raise ArtifactSchemaError(f"{context} is missing required fields: {', '.join(missing)}")
    if normalized.get("policy_profile") != "manual" and not normalized.get("policy_descriptor_hash"):
        raise ArtifactSchemaError(f"{context} is missing policy_descriptor_hash for a tracked policy profile")
    try:
        eval_time = int(normalized.get("eval_time", 0))
    except (TypeError, ValueError) as exc:
        raise ArtifactSchemaError(f"{context} has invalid eval_time") from exc
    if eval_time < 1:
        raise ArtifactSchemaError(f"{context} is incomplete: eval_time={eval_time}")
    details = normalized.get("details")
    if not isinstance(details, Mapping) or len(details) < eval_time:
        raise ArtifactSchemaError(f"{context} has incomplete episode details")
    task = str(normalized["task"])
    for index, detail in details.items():
        if not isinstance(detail, Mapping):
            raise ArtifactSchemaError(f"{context} detail {index!r} is not an object")
        layout_id = detail.get("layout_id")
        layout_file = detail.get("layout_file")
        layout_hash = detail.get("layout_sha256")
        if not isinstance(layout_id, int) or layout_file != f"{task}_{layout_id}.json":
            raise ArtifactSchemaError(f"{context} detail {index!r} has invalid task-keyed layout identity")
        if not isinstance(layout_hash, str) or len(layout_hash) != 64:
            raise ArtifactSchemaError(f"{context} detail {index!r} has invalid layout_sha256")


def require_matching_scene_identity(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    context: str,
) -> None:
    expected_identity = scene_identity(expected)
    actual_identity = scene_identity(actual)
    for field, expected_value in expected_identity.items():
        actual_value = actual_identity[field]
        if actual_value != expected_value:
            raise ValueError(f"{context} {field} mismatch: expected {expected_value!r}, found {actual_value!r}")
