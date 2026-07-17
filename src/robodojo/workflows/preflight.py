"""Read-only experiment validation."""

from __future__ import annotations

import shlex

from robodojo.core.models.reports import (
    PreflightCheck,
    PreflightReport,
)
from robodojo.core.models.requests import (
    ExperimentRequest,
    PreflightRequest,
)
from robodojo.core.paths import RepositoryPaths
from robodojo.core.processes import free_port, start, terminate_process_group, wait_for_port
from robodojo.policy.adapter import policy_launch_environment, policy_server_command
from robodojo.workflows.preflight_checks.assets import _robot_asset_check, _scene_asset_check
from robodojo.workflows.preflight_checks.configuration import _configuration_checks, _layout_check, _root_runtime_check
from robodojo.workflows.preflight_checks.policy import (
    _adapter_files_check,
    _checkpoint_check,
    _policy_hook_check,
    _policy_runtime_checks,
)
from robodojo.workflows.preflight_checks.reporting import _check, _setup_remediation, build_report, emit_report
from robodojo.workflows.preflight_checks.runtime import _resolve_preflight_gpus
from robodojo.workflows.preflight_checks.storage import _publication_check

HOOK_WARNING_EXIT = 3
ROOT_SETUP_REMEDIATION = "make setup; or " + shlex.join(
    ["uv", "run", "--locked", "robodojo", "workspace", "setup", "--only", "root"]
)


def request_from_evaluation(request: ExperimentRequest, *, task: str | None = None) -> PreflightRequest:
    """Project an evaluation request onto the shared fast-preflight request."""
    experiment = request.experiment
    if task is not None:
        experiment = experiment.model_copy(update={"task": task})
    return PreflightRequest(
        experiment=experiment,
        seed=request.seed,
        policy_gpu=request.policy_gpu,
        environment_gpu=request.environment_gpu,
        publish=getattr(request, "publish", False),
    )


def _run_fast_preflight_resolved(
    paths: RepositoryPaths,
    request: PreflightRequest,
    gpu_check: PreflightCheck,
    *,
    simulator_only: bool = False,
) -> PreflightReport:
    checks: list[PreflightCheck] = [_root_runtime_check(paths)]
    config_checks, profile, scene = _configuration_checks(paths, request)
    checks.extend(config_checks)
    checks.append(_layout_check(paths, request, scene, profile))
    checks.append(_robot_asset_check(profile, request))
    checks.append(_scene_asset_check(paths, request, scene))
    checks.append(gpu_check)
    if simulator_only:
        return build_report(checks)
    checks.append(_publication_check(request))
    checks.append(_adapter_files_check(request))
    checks.extend(_policy_runtime_checks(paths, request))
    checks.append(_checkpoint_check(request))
    checks.append(_policy_hook_check(paths, request))
    return build_report(checks)


def run_fast_preflight(paths: RepositoryPaths, request: PreflightRequest) -> PreflightReport:
    """Run every read-only experiment check without starting a process."""
    resolved, gpu_check = _resolve_preflight_gpus(request)
    if resolved is None:
        return build_report([_root_runtime_check(paths), gpu_check])
    return _run_fast_preflight_resolved(paths, resolved, gpu_check)


def run_simulator_preflight(paths: RepositoryPaths, request: PreflightRequest) -> PreflightReport:
    """Validate only the simulator-side contract for policy-free workflows."""
    resolved, gpu_check = _resolve_preflight_gpus(request, simulator_only=True)
    if resolved is None:
        return build_report([_root_runtime_check(paths), gpu_check])
    return _run_fast_preflight_resolved(paths, resolved, gpu_check, simulator_only=True)


def run_sweep_preflight(
    paths: RepositoryPaths,
    request: PreflightRequest,
    tasks: list[str],
) -> PreflightReport:
    """Run one shared gate while validating every selected task layout."""
    report = run_fast_preflight(paths, request)
    checks = list(report.checks)
    for task in tasks:
        if task == request.experiment.task:
            continue
        experiment = request.experiment.model_copy(update={"task": task})
        task_request = request.model_copy(update={"experiment": experiment})
        config_checks, profile, scene = _configuration_checks(paths, task_request)
        checks.extend(
            item.model_copy(update={"name": f"{item.name}[{task}]"})
            for item in config_checks
            if item.name in {"task", "scene"}
        )
        layout = _layout_check(paths, task_request, scene, profile)
        checks.append(layout.model_copy(update={"name": f"layout[{task}]"}))
        scene_assets = _scene_asset_check(paths, task_request, scene)
        checks.append(scene_assets.model_copy(update={"name": f"scene_assets[{task}]"}))
    return build_report(checks)


def run_deep_preflight(paths: RepositoryPaths, request: PreflightRequest) -> PreflightReport:
    """Run fast checks, then start and always stop the normal policy server."""
    resolved, gpu_check = _resolve_preflight_gpus(request)
    if resolved is None:
        return build_report([_root_runtime_check(paths), gpu_check])
    report = run_fast_preflight(paths, resolved)
    if report.status == "FAIL":
        return report
    process = None
    port = free_port()
    policy_request = resolved.policy_request(port=port)
    command = policy_server_command(policy_request, port)
    try:
        process = start(
            command,
            cwd=resolved.experiment.policy_dir.expanduser().resolve(),
            env=policy_launch_environment(resolved.experiment.checkpoint),
        )
        wait_for_port(process, "127.0.0.1", port, timeout=request.timeout)
        check = _check("deep_policy_server", "PASS", f"normal policy server became ready on temporary port {port}")
    except (OSError, RuntimeError, TimeoutError) as exc:
        check = _check("deep_policy_server", "FAIL", str(exc), _setup_remediation(resolved, "policy"))
    finally:
        if process is not None:
            terminate_process_group(process)
    return build_report([*report.checks, check])


def run_preflight(
    paths: RepositoryPaths,
    request: PreflightRequest,
    *,
    output_format: str = "human",
) -> int:
    report = run_deep_preflight(paths, request) if request.deep else run_fast_preflight(paths, request)
    emit_report(report, output_format)
    return 1 if report.status == "FAIL" else 0
