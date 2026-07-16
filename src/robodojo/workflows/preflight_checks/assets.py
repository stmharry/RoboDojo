"""Read-only experiment validation."""

from __future__ import annotations

import shlex

from robodojo.core.models.reports import (
    PreflightCheck,
)
from robodojo.core.models.requests import (
    PreflightRequest,
)
from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles.environment import EnvironmentProfile
from robodojo.core.profiles.scene import SceneProfile
from robodojo.sim.scene_assets import inspect_scene_assets
from robodojo.workflows.assets import (
    generated_fixture_error,
    generated_robot_error,
    required_fixture_builds,
    required_robot_builds,
)
from robodojo.workflows.preflight_checks.reporting import _check, _setup_remediation

HOOK_WARNING_EXIT = 3
ROOT_SETUP_REMEDIATION = "make setup; or " + shlex.join(
    ["uv", "run", "--locked", "robodojo", "setup", "--only", "root"]
)


def _robot_asset_check(
    profile: EnvironmentProfile | None,
    request: PreflightRequest | None = None,
) -> PreflightCheck:
    remediation = _setup_remediation(request, "assets") if request else "make setup"
    if profile is None:
        return _check("robot_assets", "FAIL", "environment did not resolve", remediation)
    generated = required_robot_builds(profile)
    if not generated:
        return _check("robot_assets", "PASS", "environment requires no generated robot manifests")
    for name in generated:
        if error := generated_robot_error(name):
            return _check("robot_assets", "FAIL", error, remediation)
    return _check("robot_assets", "PASS", f"verified generated manifest(s): {', '.join(generated)}")


def _scene_asset_check(paths: RepositoryPaths, request: PreflightRequest, scene: SceneProfile | None) -> PreflightCheck:
    remediation = _setup_remediation(request, "assets")
    if scene is None:
        return _check("scene_assets", "FAIL", "scene did not resolve", remediation)
    required_builds = required_fixture_builds(scene, request.experiment.task)
    for name in required_builds:
        if error := generated_fixture_error(paths, name):
            return _check("scene_assets", "FAIL", error, remediation)
    try:
        prepared = inspect_scene_assets(scene, request.experiment.task)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return _check("scene_assets", "FAIL", str(exc), remediation)
    identities = [
        f"{artifact.destination_root.name}:{artifact.derivation_hash[:12]}:{artifact.manifest_hash[:12]}"
        for artifact in prepared.artifacts
    ]
    resolved = ", ".join((*required_builds, *identities)) or "none"
    return _check(
        "scene_assets",
        "PASS",
        f"verified {len(prepared.artifacts)} task-derived asset(s) and "
        f"{len(required_builds)} generated scene asset build(s); identities={resolved}",
    )
