"""Read-only identity for embodiment-owned generated assets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from robodojo.core.profiles import EnvironmentProfile
from robodojo.core.storage import assets_root


@dataclass(frozen=True)
class EnvironmentAssetIdentity:
    identity_hash: str
    artifacts: tuple[dict[str, Any], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_environment_assets(profile: EnvironmentProfile) -> EnvironmentAssetIdentity:
    """Resolve the generated robot manifests selected by one environment profile."""

    root = assets_root()
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256(b"robodojo-environment-assets-v1\0")
    for build in profile.document.asset_builds:
        manifest_path = root / "Robots" / build / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"environment asset manifest is missing or invalid: {manifest_path}: {exc}") from exc
        manifest_hash = _sha256(manifest_path)
        derivation_hash = str((manifest.get("provenance") or {}).get("build_manifest_sha256") or manifest_hash)
        record = {
            "build": build,
            "destination": (Path("Robots") / build).as_posix(),
            "derivation_hash": derivation_hash,
            "manifest_hash": manifest_hash,
        }
        records.append(record)
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\0")
    return EnvironmentAssetIdentity(identity_hash=digest.hexdigest(), artifacts=tuple(records))
