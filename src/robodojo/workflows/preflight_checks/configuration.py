"""Read-only experiment validation."""

from __future__ import annotations

import json
import shlex

import yaml

from robodojo.core.experiments.catalogs import load_protocol_catalog
from robodojo.core.experiments.selection import compose_experiment
from robodojo.core.layouts import resolve_layout_set
from robodojo.core.models.reports import (
    PreflightCheck,
)
from robodojo.core.models.requests import (
    PreflightRequest,
    SimulatorLaunchRequest,
)
from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles.environment import EnvironmentProfile, load_environment_profile
from robodojo.core.profiles.scene import SceneProfile, validate_scene_environment_compatibility
from robodojo.core.storage import assets_root
from robodojo.core.workspace import validate_layout_contract
from robodojo.sim.launcher import resolve_scene_profile
from robodojo.workflows.preflight_checks.reporting import _check, _setup_remediation
from robodojo.workflows.setup import root_environment_error
from robodojo.workflows.task_inventory import build_inventory

HOOK_WARNING_EXIT = 3
ROOT_SETUP_REMEDIATION = "make setup; or " + shlex.join(
    ["uv", "run", "--locked", "robodojo", "setup", "--only", "root"]
)


def _root_runtime_check(paths: RepositoryPaths) -> PreflightCheck:
    if error := root_environment_error(paths):
        return _check("root_sim_environment", "FAIL", error, ROOT_SETUP_REMEDIATION)
    python = paths.root / ".venv" / "bin" / "python"
    return _check("root_sim_environment", "PASS", f"uv.lock and simulator packages are ready in {python.parent.parent}")


def _configuration_checks(
    paths: RepositoryPaths,
    request: PreflightRequest,
) -> tuple[list[PreflightCheck], EnvironmentProfile | None, SceneProfile | None]:
    checks: list[PreflightCheck] = []
    inventory = build_inventory()
    record = next((item for item in inventory["tasks"] if item["name"] == request.experiment.task), None)
    if record is None:
        checks.append(
            _check(
                "task",
                "FAIL",
                f"unknown task: {request.experiment.task}",
                "run make tasks and select a valid TASK",
            )
        )
    elif not record["runnable"]:
        checks.append(
            _check(
                "task",
                "FAIL",
                f"task is not runnable: {request.experiment.task}",
                "run make tasks and select a runnable TASK",
            )
        )
    else:
        checks.append(_check("task", "PASS", f"{request.experiment.task} code and YAML are runnable"))

    try:
        profile = load_environment_profile(paths, request.experiment.environment)
        checks.append(_check("environment", "PASS", str(profile.path)))
    except Exception as exc:
        checks.append(_check("environment", "FAIL", str(exc), "select a valid ENV_CFG"))
        return checks, None, None

    if request.experiment.embodiment != profile.embodiment:
        checks.append(
            _check(
                "policy_environment",
                "FAIL",
                f"policy embodiment {request.experiment.embodiment!r} does not match "
                f"environment embodiment {profile.embodiment!r}",
                "select a compatible policy profile and environment recipe",
            )
        )
    else:
        checks.append(_check("policy_environment", "PASS", profile.embodiment))

    try:
        protocol = load_protocol_catalog(paths).protocols[request.experiment.task_protocol]
        actual = (request.experiment.task, request.experiment.episode_horizon, request.experiment.evaluation_episodes)
        expected = (protocol.task, protocol.episode_horizon, protocol.evaluation_episodes)
        if actual != expected:
            raise ValueError(f"resolved fields {actual} do not match catalog {expected}")
        if protocol.compatible_scenes and request.experiment.scene not in protocol.compatible_scenes:
            raise ValueError(
                f"compatible scenes are {protocol.compatible_scenes}, received {request.experiment.scene!r}"
            )
        checks.append(
            _check("protocol", "PASS", f"{request.experiment.task_protocol} -> task={request.experiment.task}")
        )
    except (KeyError, TypeError, ValueError) as exc:
        checks.append(_check("protocol", "FAIL", str(exc), "select a valid recipe or complete manual contract"))

    simulator_request = SimulatorLaunchRequest(
        experiment=request.experiment,
        policy_name=request.experiment.policy_dir.name,
        port=1,
        seed=request.seed,
        additional_info="preflight",
    )
    try:
        scene = resolve_scene_profile(paths, simulator_request)
        validate_scene_environment_compatibility(scene, profile)
        checks.append(_check("scene", "PASS", f"{scene.name} -> {scene.component_path}"))
    except Exception as exc:
        checks.append(_check("scene", "FAIL", str(exc), "select compatible SCENE and ENV_CFG values"))
        return checks, profile, None

    if request.experiment.policy_profile == "manual":
        checks.append(
            _check(
                "policy_descriptor",
                "WARN",
                "direct request has no tracked policy profile identity",
                "resolve the request through --policy-profile or --recipe",
            )
        )
    else:
        try:
            resolved = compose_experiment(
                paths,
                policy_name=request.experiment.policy_profile,
                environment_name=request.experiment.environment,
                scene_name=request.experiment.scene,
                task_protocol=request.experiment.task_protocol,
                recipe_name=request.experiment.recipe,
            )
            actual = (
                request.experiment.checkpoint,
                request.experiment.embodiment,
                request.experiment.action_type,
                request.experiment.dataset,
                request.experiment.policy_descriptor_hash,
                request.experiment.experiment_hash,
                request.experiment.policy_reference_match,
            )
            expected = (
                resolved.policy.checkpoint,
                resolved.policy_descriptor.interface.embodiment,
                resolved.policy_descriptor.launch.action_type,
                resolved.policy_descriptor.launch.dataset,
                resolved.policy_descriptor_hash,
                resolved.identity_hash,
                resolved.policy_reference_match,
            )
            if actual != expected:
                raise ValueError(f"resolved policy descriptor fields {actual} do not match catalog {expected}")
            if resolved.policy_reference_match == "domain_shift":
                checks.append(
                    _check(
                        "policy_descriptor",
                        "WARN",
                        f"{request.experiment.policy_profile} is running outside its declared reference setup",
                    )
                )
            else:
                checks.append(
                    _check(
                        "policy_descriptor",
                        "PASS",
                        f"{request.experiment.policy_profile} descriptor={resolved.policy_descriptor_hash}",
                    )
                )
        except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            checks.append(
                _check(
                    "policy_descriptor",
                    "FAIL",
                    str(exc),
                    "select a tracked policy profile with a valid XPolicyLab eval_contracts.yml",
                )
            )
    return checks, profile, scene


def _layout_check(
    paths: RepositoryPaths,
    request: PreflightRequest,
    scene: SceneProfile | None,
    profile: EnvironmentProfile | None = None,
) -> PreflightCheck:
    if scene is None:
        return _check(
            "layout",
            "FAIL",
            "scene did not resolve",
            "select compatible SCENE and ENV_CFG values",
        )
    try:
        resolved = resolve_layout_set(
            config_root=paths.environment_configs,
            assets_root=assets_root(),
            benchmark=request.experiment.dataset,
            layout_set=scene.document.layout_set,
            layout_source=scene.document.layout_source,
            task=request.experiment.task,
            seed=request.seed,
        )
        task_config = (
            yaml.safe_load((paths.task_configs / f"{request.experiment.task}.yml").read_text(encoding="utf-8")) or {}
        )
        robot_config = (
            yaml.safe_load(profile.component_paths["robot"].read_text(encoding="utf-8")) or {}
            if profile is not None
            else None
        )
        for selected in resolved.layouts:
            layout = json.loads(selected.path.read_text(encoding="utf-8"))
            validate_layout_contract(
                layout,
                task_config,
                workspace=profile.document.workspace if profile is not None else None,
                robot_config=robot_config,
                context=str(selected.path),
            )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return _check(
            "layout",
            "FAIL",
            f"invalid task-keyed layout for {request.experiment.task}: {exc}",
            _setup_remediation(request, "assets"),
        )
    return _check(
        "layout",
        "PASS",
        f"{len(resolved.layouts)} {resolved.directory} layout(s) keyed by task {request.experiment.task}",
    )
