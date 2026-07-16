"""Snapshot schema normalization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SNAPSHOT_SCHEMA_VERSION = 2
LEGACY_SNAPSHOT_SCHEMA_VERSION = 1


def normalize_snapshot_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    version = payload.get("format_version", LEGACY_SNAPSHOT_SCHEMA_VERSION)
    normalized = dict(payload)
    if version == SNAPSHOT_SCHEMA_VERSION:
        return normalized
    if version != LEGACY_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported snapshot schema {version!r}; expected "
            f"{LEGACY_SNAPSHOT_SCHEMA_VERSION} or {SNAPSHOT_SCHEMA_VERSION}"
        )
    results = []
    for raw_record in normalized.get("results") or ():
        record = dict(raw_record)
        if "protocol" in record:
            record["task_protocol"] = record.pop("protocol")
        if "contract_hash" in record:
            record["experiment_hash"] = record.pop("contract_hash")
        results.append(record)
    normalized["results"] = results
    normalized["format_version"] = SNAPSHOT_SCHEMA_VERSION
    return normalized


def normalize_recipe_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    version = payload.get("format_version")
    normalized = dict(payload)
    if version == SNAPSHOT_SCHEMA_VERSION:
        return normalized
    if version != LEGACY_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported snapshot recipe format {version!r}; expected "
            f"{LEGACY_SNAPSHOT_SCHEMA_VERSION} or {SNAPSHOT_SCHEMA_VERSION}"
        )
    identity = dict(normalized.get("identity") or {})
    if "contract_hash" in identity:
        identity["experiment_hash"] = identity.pop("contract_hash")
    if "protocol" in identity:
        identity["task_protocol"] = identity.pop("protocol")
    normalized["identity"] = identity
    normalized["format_version"] = SNAPSHOT_SCHEMA_VERSION
    return normalized
