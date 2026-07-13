"""Environment-profile loading shared by lightweight launch workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robodojo.core.calibration import load_hardware_calibration
from robodojo.core.models import EnvironmentConfigDocument
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

    @property
    def num_envs(self) -> int:
        value = int(self.sim.get("scene", {}).get("num_envs", 1))
        if value < 1:
            raise ValueError(f"environment profile {self.name} must configure at least one environment")
        return value

    @property
    def xpolicylab_env_cfg_type(self) -> str:
        if self.document.xpolicylab is None:
            return self.name
        return self.document.xpolicylab.env_cfg_type


def _profile_path(config_root: Path, relative: str, *, field: str) -> Path:
    root = config_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{field} must stay below {root}: {relative}")
    if not path.is_file():
        raise ValueError(f"{field} not found: {path}")
    return path


def load_environment_profile(
    paths: RepositoryPaths,
    name: str,
    *,
    validate_calibration: bool = True,
) -> EnvironmentProfile:
    """Load and validate an additive RoboDojo environment profile."""
    config_root = paths.environment_configs
    path = config_root / f"{name}.yml"
    if not path.is_file():
        raise ValueError(f"environment config not found: {path}")
    payload: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    document = EnvironmentConfigDocument.model_validate(payload)
    if document.config_name != name:
        raise ValueError(f"environment config_name must be {name}: {path}")

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

    profile = EnvironmentProfile(
        name=name,
        path=path,
        payload=payload,
        document=document,
        component_paths=component_paths,
        sim=sim,
        calibration=calibration,
        matched_replay_manifest=matched_replay_manifest,
    )
    profile.num_envs
    return profile
