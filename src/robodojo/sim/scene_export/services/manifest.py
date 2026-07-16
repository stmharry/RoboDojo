"""Export the composed, post-reset Isaac stage without editing the live stage."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
import logging
from pathlib import Path
import subprocess
from typing import Any

import yaml

from robodojo.core.storage import assets_root
from robodojo.sim.scene_export.contracts import (
    scene_input_paths,
)

logger = logging.getLogger(__name__)

USD_EXTENSIONS = frozenset({".usd", ".usda", ".usdc", ".usdz"})
MANIFEST_NAME = "scene_manifest.json"
REFERENCED_NAME = "scene_referenced.usda"
FLATTENED_NAME = "scene_flattened.usdc"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(repo_root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return revision, dirty


def _runtime_versions() -> dict[str, str | None]:
    versions = {}
    for package in ("isaacsim", "isaaclab", "usd-core", "torch", "numpy"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _source_revisions(repo_root: Path) -> dict[str, Any]:
    tracked_manifest = repo_root / "configs/tooling/openarm.yml"
    lerobot_reference = repo_root / "configs/reference/openarm_lerobot.yml"
    generated_manifest = assets_root() / "Robots/openarm/manifest.json"
    result = {
        "tracked_openarm_manifest": None,
        "generated_openarm_manifest": None,
        "openarm_lerobot_reference": None,
    }
    try:
        result["tracked_openarm_manifest"] = yaml.safe_load(tracked_manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):
        pass
    try:
        result["openarm_lerobot_reference"] = yaml.safe_load(lerobot_reference.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):
        pass
    try:
        result["generated_openarm_manifest"] = json.loads(generated_manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return result


def _config_hashes(repo_root: Path, env) -> dict[str, str]:
    paths = scene_input_paths(
        repo_root,
        env.environment,
        env.scene,
        env.task_name,
        env.eval_cfg["config"],
    )
    return {str(path.relative_to(repo_root)): _sha256_file(path) for path in paths if path.is_file()}
