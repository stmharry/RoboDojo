"""Typed scene profile loading and compatibility."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import yaml

from robodojo.core.models.scene import SceneConfigDocument
from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles.common import profile_path
from robodojo.core.profiles.environment import EnvironmentProfile


@dataclass(frozen=True)
class SceneProfile:
    name: str
    path: Path
    payload: dict[str, Any]
    document: SceneConfigDocument
    component_path: Path
    component: dict[str, Any]
    identity_hash: str


def validate_scene_environment_compatibility(scene: SceneProfile, environment: EnvironmentProfile) -> None:
    compatible = scene.document.compatible_environments
    if compatible and environment.name not in compatible:
        raise ValueError(
            f"scene profile {scene.name!r} is compatible only with environment profiles {compatible}; "
            f"received {environment.name!r}"
        )


def load_scene_profile(paths: RepositoryPaths, name: str) -> SceneProfile:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    if not name or any(character not in allowed for character in name):
        raise ValueError("scene profile names must contain only letters, digits, and underscores")
    path = paths.scene_profiles / f"{name}.yml"
    if not path.is_file():
        raise ValueError(f"scene profile not found: {path}")
    payload: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    document = SceneConfigDocument.model_validate(payload)
    if document.config_name != name:
        raise ValueError(f"scene config_name must be {name}: {path}")
    component_path = profile_path(
        paths.environment_configs,
        f"scene/components/{document.component}.yml",
        field="referenced scene component",
    )
    component: dict[str, Any] = yaml.safe_load(component_path.read_text(encoding="utf-8")) or {}
    digest = hashlib.sha256()
    digest.update(b"robodojo-scene-profile-v2\0")
    for input_path in (path, component_path):
        digest.update(input_path.relative_to(paths.environment_configs).as_posix().encode())
        digest.update(b"\0")
        digest.update(input_path.read_bytes())
        digest.update(b"\0")
    return SceneProfile(
        name=name,
        path=path,
        payload=payload,
        document=document,
        component_path=component_path,
        component=component,
        identity_hash=digest.hexdigest(),
    )
