"""Export the composed, post-reset Isaac stage without editing the live stage."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import shutil
from typing import Any

from pxr import Ar, Sdf, UsdUtils

from robodojo.sim.scene_export.contracts import (
    package_member_exists,
    split_package_asset_path,
)
from robodojo.sim.scene_export.services.manifest import _sha256_file

logger = logging.getLogger(__name__)

USD_EXTENSIONS = frozenset({".usd", ".usda", ".usdc", ".usdz"})
MANIFEST_NAME = "scene_manifest.json"
REFERENCED_NAME = "scene_referenced.usda"
FLATTENED_NAME = "scene_flattened.usdc"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_asset(path: str, repo_root: Path) -> Path | None:
    if not path or path.startswith("anon:") or "://" in path:
        return None
    candidate = Path(path)
    if candidate.is_absolute() and candidate.is_file():
        return candidate.resolve()
    try:
        resolved = str(Ar.GetResolver().Resolve(path))
    except Exception:
        resolved = ""
    if resolved and Path(resolved).is_file():
        return Path(resolved).resolve()
    for base in (repo_root, Path.cwd()):
        candidate = base / path
        if candidate.is_file():
            return candidate.resolve()
    return None


def _modify_asset_paths(layer: Sdf.Layer, callback) -> None:
    modifier = getattr(UsdUtils, "ModifyAssetPaths", None)
    if modifier is None:
        return
    modifier(layer, callback)


def _make_referenced_paths_portable(layer: Sdf.Layer, output_dir: Path, repo_root: Path) -> list[dict[str, Any]]:
    dependencies: dict[str, dict[str, Any]] = {}

    def rewrite(path: str) -> str:
        resolved = _resolve_asset(path, repo_root)
        if resolved is None:
            if path:
                dependencies.setdefault(path, {"authored_path": path, "status": "unresolved"})
            return path
        relative = os.path.relpath(resolved, output_dir)
        dependencies[str(resolved)] = {
            "authored_path": path,
            "resolved_path": str(resolved),
            "export_path": relative,
            "sha256": _sha256_file(resolved),
            "status": "external",
        }
        return relative

    _modify_asset_paths(layer, rewrite)
    return sorted(dependencies.values(), key=lambda item: item.get("resolved_path", item["authored_path"]))


def _bundle_flattened_assets(
    layer: Sdf.Layer, output_dir: Path, repo_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dependency_dir = output_dir / "dependencies"
    bundled: dict[str, dict[str, Any]] = {}
    unresolved: dict[str, dict[str, Any]] = {}

    def rewrite(path: str) -> str:
        outer_path, package_member = split_package_asset_path(path)
        resolved = _resolve_asset(outer_path, repo_root)
        if resolved is None:
            if path:
                unresolved.setdefault(path, {"authored_path": path, "status": "unresolved"})
            return path
        if package_member and not package_member_exists(resolved, package_member):
            unresolved.setdefault(
                path,
                {
                    "authored_path": path,
                    "resolved_package": str(resolved),
                    "missing_member": package_member,
                    "status": "missing-package-member",
                },
            )
            return path
        source = str(resolved)
        if source not in bundled:
            digest = _sha256_file(resolved)
            destination_name = f"{digest[:12]}_{resolved.name}"
            destination = dependency_dir / destination_name
            dependency_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved, destination)
            bundled[source] = {
                "authored_path": path,
                "resolved_path": source,
                "bundled_path": f"dependencies/{destination_name}",
                "sha256": digest,
                "kind": "usd" if resolved.suffix.lower() in USD_EXTENSIONS else "asset",
                "status": "bundled",
            }
        suffix = f"[{package_member}]" if package_member else ""
        return bundled[source]["bundled_path"] + suffix

    _modify_asset_paths(layer, rewrite)
    return list(bundled.values()), list(unresolved.values())
