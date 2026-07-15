"""Environment-profile loading shared by lightweight launch workflows."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import yaml

from robodojo.core.calibration import load_hardware_calibration
from robodojo.core.models import EnvironmentConfigDocument, SceneConfigDocument
from robodojo.core.paths import RepositoryPaths


@dataclass(frozen=True)
class EnvironmentProfile:
    name: str
    path: Path
    payload: dict[str, Any]
    document: EnvironmentConfigDocument
    component_paths: dict[str, Path]
    sim: dict[str, Any]
    calibration: dict[str, Any] | None
    matched_replay_manifest: Path | None
    source_paths: tuple[Path, ...]
    identity_hash: str

    @property
    def num_envs(self) -> int:
        value = int(self.sim.get("scene", {}).get("num_envs", 1))
        if value < 1:
            raise ValueError(f"environment profile {self.name} must configure at least one environment")
        return value

    @property
    def policy_contract(self) -> str:
        return self.document.policy_contract or self.name


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
    """Reject scene/embodiment combinations that have no declared mount contract."""
    compatible = scene.document.compatible_environments
    if compatible and environment.name not in compatible:
        raise ValueError(
            f"scene profile {scene.name!r} is compatible only with environment profiles {compatible}; "
            f"received {environment.name!r}"
        )


def _profile_path(config_root: Path, relative: str, *, field: str) -> Path:
    root = config_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{field} must stay below {root}: {relative}")
    if not path.is_file():
        raise ValueError(f"{field} not found: {path}")
    return path


def _merge_profile_payload(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_profile_payload(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _environment_payload(
    paths: RepositoryPaths,
    name: str,
    stack: tuple[str, ...] = (),
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    if name in stack:
        raise ValueError(f"environment profile inheritance cycle: {' -> '.join((*stack, name))}")
    path = paths.environment_profiles / f"{name}.yml"
    if not path.is_file():
        raise ValueError(f"environment config not found: {path}")
    payload: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = payload.get("extends")
    if parent is None:
        return payload, (path,)
    if not isinstance(parent, str) or not parent:
        raise ValueError(f"environment profile extends must be a non-empty profile name: {path}")
    base, source_paths = _environment_payload(paths, parent, (*stack, name))
    return _merge_profile_payload(base, payload), (*source_paths, path)


def load_environment_profile(
    paths: RepositoryPaths,
    name: str,
    *,
    validate_calibration: bool = True,
    require_selectable: bool = True,
) -> EnvironmentProfile:
    """Load and validate an additive RoboDojo environment profile."""
    config_root = paths.environment_configs
    path = paths.environment_profiles / f"{name}.yml"
    payload, source_paths = _environment_payload(paths, name)
    document = EnvironmentConfigDocument.model_validate(payload)
    if document.config_name != name:
        raise ValueError(f"environment config_name must be {name}: {path}")
    if require_selectable and not document.selectable:
        raise ValueError(
            f"environment profile {name!r} is an internal contract and cannot be selected; "
            "choose a named selectable setup profile"
        )

    component_paths: dict[str, Path] = {}
    for section, component_name in document.config.model_dump().items():
        component_paths[section] = _profile_path(
            config_root,
            f"{section}/{component_name}.yml",
            field=f"referenced {section} config",
        )

    sim: dict[str, Any] = yaml.safe_load(component_paths["sim"].read_text(encoding="utf-8")) or {}
    calibration = None
    if document.hardware_calibration and validate_calibration:
        calibration = load_hardware_calibration(config_root, document.hardware_calibration)

    matched_replay_manifest = None
    if document.diagnostics and document.diagnostics.matched_replay_manifest:
        matched_replay_manifest = _profile_path(
            config_root,
            document.diagnostics.matched_replay_manifest,
            field="matched replay manifest",
        )

    digest = hashlib.sha256()
    digest.update(b"robodojo-environment-profile-v1\0")
    identity_inputs = (*source_paths, *(component_paths[key] for key in sorted(component_paths)))
    for input_path in identity_inputs:
        digest.update(input_path.relative_to(paths.environment_configs).as_posix().encode())
        digest.update(b"\0")
        digest.update(input_path.read_bytes())
        digest.update(b"\0")
    profile = EnvironmentProfile(
        name=name,
        path=path,
        payload=payload,
        document=document,
        component_paths=component_paths,
        sim=sim,
        calibration=calibration,
        matched_replay_manifest=matched_replay_manifest,
        source_paths=source_paths,
        identity_hash=digest.hexdigest(),
    )
    profile.num_envs
    return profile


def bind_policy_contract(paths: RepositoryPaths, request):
    """Return a request whose adapter environment names the shared policy contract."""

    if request.env_config is None:
        return request
    if not (paths.environment_profiles / f"{request.env_config}.yml").is_file():
        return request
    profile = load_environment_profile(paths, request.env_config)
    return request.model_copy(update={"policy_contract": profile.policy_contract})


def load_scene_profile(paths: RepositoryPaths, name: str) -> SceneProfile:
    """Load the typed scene profile selected by the existing ``--scene`` surface."""

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
    component_path = _profile_path(
        paths.environment_configs,
        f"scene/components/{document.component}.yml",
        field="referenced scene component",
    )
    component: dict[str, Any] = yaml.safe_load(component_path.read_text(encoding="utf-8")) or {}
    digest = hashlib.sha256()
    digest.update(b"robodojo-scene-profile-v1\0")
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
