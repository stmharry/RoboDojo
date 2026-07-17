"""Scene export manifest normalization and completion validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

SCENE_EXPORT_FORMAT_VERSION = 8
LEGACY_SCENE_EXPORT_FORMAT_VERSION = 7
REQUIRED_EXPORT_ARTIFACTS = (
    "scene_referenced.usda",
    "scene_flattened.usdc",
    "scene_preview.usdz",
)
REQUIRED_MANIFEST_ARTIFACTS = {
    "referenced_usda": "scene_referenced.usda",
    "flattened_usdc": "scene_flattened.usdc",
    "preview_usdz": "scene_preview.usdz",
}
REQUIRED_PREVIEW_DIAGNOSTICS = (
    "preserved_materials",
    "translated_materials",
    "fallback_materials",
    "missing_textures",
    "unsupported_inputs",
    "excluded_guide_meshes",
    "approximation",
)


class SceneExportArtifactError(ValueError):
    """Raised when a scene export is unsupported or incomplete."""


@dataclass(frozen=True)
class ExportIdentity:
    task: str
    task_protocol: str
    episode_horizon: int
    evaluation_episodes: int
    recipe: str | None
    experiment_hash: str | None
    environment: str
    scene: str
    seed: int
    layout_id: int
    repository_revision: str
    environment_profile_hash: str
    embodiment: str
    scene_profile_hash: str
    layout_set_hash: str
    scene_asset_hash: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_export_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a format-v8 in-memory view of a v7 or v8 scene manifest."""

    version = manifest.get("format_version")
    normalized = dict(manifest)
    if version == SCENE_EXPORT_FORMAT_VERSION:
        return normalized
    if version != LEGACY_SCENE_EXPORT_FORMAT_VERSION:
        raise SceneExportArtifactError(
            f"unsupported scene export format {version!r}; expected "
            f"{LEGACY_SCENE_EXPORT_FORMAT_VERSION} or {SCENE_EXPORT_FORMAT_VERSION}"
        )
    identity = dict(normalized.get("identity") or {})
    for old, new in {
        "protocol": "task_protocol",
        "native_eval_num": "evaluation_episodes",
        "contract_hash": "experiment_hash",
        "profile": "environment",
        "scene_config": "scene",
        "policy_contract": "embodiment",
    }.items():
        if old in identity:
            identity[new] = identity.pop(old)
    normalized["identity"] = identity
    normalized["format_version"] = SCENE_EXPORT_FORMAT_VERSION
    return normalized


def require_completed_scene_export(
    output_dir: str | Path,
    *,
    require_scene_export_only: bool = False,
    context: str = "scene export",
) -> dict[str, Any]:
    """Return a normalized manifest after validating its completed payload."""

    output = Path(output_dir)
    manifest_path = output / "scene_manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise SceneExportArtifactError(f"{context} manifest is missing or invalid: {manifest_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SceneExportArtifactError(f"{context} manifest is not an object: {manifest_path}")
    try:
        manifest = normalize_export_manifest(payload)
    except (TypeError, ValueError) as exc:
        raise SceneExportArtifactError(f"{context} {exc}") from exc
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise SceneExportArtifactError(f"{context} manifest has invalid identity: {manifest_path}")
    try:
        ExportIdentity(**identity)
    except TypeError as exc:
        raise SceneExportArtifactError(f"{context} manifest has invalid identity: {manifest_path}: {exc}") from exc
    if not manifest.get("complete"):
        raise SceneExportArtifactError(f"{context} manifest is incomplete: {manifest_path}")
    if require_scene_export_only and manifest.get("scene_export_only") is not True:
        raise SceneExportArtifactError(f"{context} manifest is not marked scene_export_only: {manifest_path}")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise SceneExportArtifactError(f"{context} manifest has invalid artifacts: {manifest_path}")
    for key, filename in REQUIRED_MANIFEST_ARTIFACTS.items():
        artifact = artifacts.get(key)
        if (
            not isinstance(artifact, Mapping)
            or artifact.get("path") != filename
            or not isinstance(artifact.get("sha256"), str)
            or len(artifact["sha256"]) != 64
            or not (output / filename).is_file()
        ):
            raise SceneExportArtifactError(f"{context} is missing completed artifact {filename}: {manifest_path}")

    preview = manifest.get("preview")
    if not isinstance(preview, Mapping) or any(key not in preview for key in REQUIRED_PREVIEW_DIAGNOSTICS):
        raise SceneExportArtifactError(f"{context} manifest has incomplete preview diagnostics: {manifest_path}")
    return manifest
