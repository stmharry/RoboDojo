"""Typed environment profile loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from robodojo.core.calibration import load_hardware_calibration
from robodojo.core.models.environment import EnvironmentConfigDocument
from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles.common import merge_profile_payload, profile_path


@dataclass(frozen=True)
class EnvironmentCameraInterface:
    role: str
    resolutions: tuple[tuple[int, int], ...]
    rate_hz: int | None


@dataclass(frozen=True)
class EnvironmentPolicyInterface:
    state_dimension: int
    action_dimension: int
    action_rate_hz: int
    cameras: tuple[EnvironmentCameraInterface, ...]

    def camera(self, role: str) -> EnvironmentCameraInterface | None:
        return next((camera for camera in self.cameras if camera.role == role), None)


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
    policy_interface: EnvironmentPolicyInterface

    @property
    def num_envs(self) -> int:
        value = int(self.sim.get("scene", {}).get("num_envs", 1))
        if value < 1:
            raise ValueError(f"environment profile {self.name} must configure at least one environment")
        return value

    @property
    def embodiment(self) -> str:
        return self.document.embodiment or self.name


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
    return merge_profile_payload(base, payload), (*source_paths, path)


def _policy_interface(
    config_root: Path,
    document: EnvironmentConfigDocument,
    component_paths: dict[str, Path],
) -> EnvironmentPolicyInterface:
    robot_info_path = config_root / "robot" / "_robot_info.json"
    dimension = 0
    if robot_info_path.is_file():
        robot_info = json.loads(robot_info_path.read_text(encoding="utf-8"))
        dimensions = robot_info.get(document.config.robot)
        if isinstance(dimensions, dict):
            arm_dimensions = dimensions.get("arm_dim")
            end_effector_dimensions = dimensions.get("ee_dim")
            if isinstance(arm_dimensions, list) and isinstance(end_effector_dimensions, list):
                dimension = sum(int(value) for value in (*arm_dimensions, *end_effector_dimensions))

    rate = document.observation.get("collect_freq")
    if isinstance(rate, bool) or not isinstance(rate, int) or rate < 1:
        rate = 0

    camera_payload: dict[str, Any] = yaml.safe_load(component_paths["camera"].read_text(encoding="utf-8")) or {}
    camera_interfaces: list[EnvironmentCameraInterface] = []
    layered = camera_payload.get("camera_rig", {}).get("cameras")
    if isinstance(layered, dict):
        for camera in layered.values():
            if not isinstance(camera, dict):
                continue
            role = camera.get("role")
            sensor = camera.get("sensor") or {}
            if not isinstance(role, str) or not role:
                continue
            resolution = sensor.get("stream_resolution")
            resolutions = ()
            if isinstance(resolution, list) and len(resolution) == 2:
                resolutions = ((int(resolution[0]), int(resolution[1])),)
            fps = sensor.get("fps")
            camera_interfaces.append(
                EnvironmentCameraInterface(
                    role=role,
                    resolutions=resolutions,
                    rate_hz=int(fps) if isinstance(fps, int) and not isinstance(fps, bool) else None,
                )
            )
    else:
        roles = {
            "cam_head": "top",
            "cam_left_wrist": "left_wrist",
            "cam_right_wrist": "right_wrist",
        }
        for key in camera_payload.get("annotator") or {}:
            if key in roles:
                camera_interfaces.append(EnvironmentCameraInterface(role=roles[key], resolutions=(), rate_hz=None))

    role_names = [camera.role for camera in camera_interfaces]
    if len(role_names) != len(set(role_names)):
        raise ValueError(f"environment {document.config_name!r} camera roles must be unique")
    return EnvironmentPolicyInterface(
        state_dimension=dimension,
        action_dimension=dimension,
        action_rate_hz=rate,
        cameras=tuple(camera_interfaces),
    )


def load_environment_profile(
    paths: RepositoryPaths,
    name: str,
    *,
    validate_calibration: bool = True,
    require_selectable: bool = True,
) -> EnvironmentProfile:
    config_root = paths.environment_configs
    path = paths.environment_profiles / f"{name}.yml"
    payload, source_paths = _environment_payload(paths, name)
    document = EnvironmentConfigDocument.model_validate(payload)
    if document.config_name != name:
        raise ValueError(f"environment config_name must be {name}: {path}")
    if require_selectable and not document.selectable:
        raise ValueError(f"environment profile {name!r} is internal and cannot be selected")

    component_paths = {
        section: profile_path(config_root, f"{section}/{component}.yml", field=f"referenced {section} config")
        for section, component in document.config.model_dump().items()
    }
    sim: dict[str, Any] = yaml.safe_load(component_paths["sim"].read_text(encoding="utf-8")) or {}
    calibration = (
        load_hardware_calibration(config_root, document.hardware_calibration)
        if document.hardware_calibration and validate_calibration
        else None
    )
    matched_replay_manifest = (
        profile_path(config_root, document.diagnostics.matched_replay_manifest, field="matched replay manifest")
        if document.diagnostics and document.diagnostics.matched_replay_manifest
        else None
    )

    policy_interface = _policy_interface(config_root, document, component_paths)
    digest = hashlib.sha256()
    digest.update(b"robodojo-environment-profile-v3\0")
    for input_path in (*source_paths, *(component_paths[key] for key in sorted(component_paths))):
        digest.update(input_path.relative_to(paths.environment_configs).as_posix().encode())
        digest.update(b"\0")
        digest.update(input_path.read_bytes())
        digest.update(b"\0")
    digest.update(json.dumps(asdict(policy_interface), sort_keys=True, separators=(",", ":")).encode())
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
        policy_interface=policy_interface,
    )
    profile.num_envs
    return profile
