"""Idempotent repository-local experiment setup."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable

import yaml

from robodojo.core.gpu import GpuSelectionError, resolve_gpus
from robodojo.core.models import SetupReport, SetupRequest, SetupStage, SetupStageResult, SimulatorLaunchRequest
from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles import (
    bind_policy_contract,
    load_environment_profile,
    validate_scene_environment_compatibility,
)
from robodojo.core.storage import assets_root
from robodojo.policy.adapter import policy_hook_command, policy_launch_environment
from robodojo.sim.launcher import resolve_scene_profile
from robodojo.sim.scene_assets import inspect_scene_assets, prepare_scene_assets
from robodojo.workflows.assets import (
    ensure_generated_fixture,
    ensure_generated_robot,
    required_fixture_builds,
    required_robot_builds,
)
from robodojo.workflows.downloads import assets_ready, download_assets
from robodojo.workflows.task_inventory import build_inventory


def _stage(
    name: str,
    status: str,
    detail: str,
    remediation: str | None = None,
) -> SetupStageResult:
    return SetupStageResult(name=name, status=status, detail=detail, remediation=remediation)


def build_report(stages: list[SetupStageResult]) -> SetupReport:
    if any(stage.status == "FAIL" for stage in stages):
        status = "FAIL"
    elif any(stage.status == "WARN" for stage in stages):
        status = "WARN"
    else:
        status = "PASS"
    return SetupReport(status=status, stages=stages)


def emit_report(report: SetupReport, output_format: str = "human") -> None:
    if output_format == "json":
        sys.stdout.write(report.model_dump_json(indent=2, exclude_none=True) + "\n")
        return
    if output_format != "human":
        raise ValueError(f"unsupported report format: {output_format}")
    for stage in report.stages:
        sys.stdout.write(f"{stage.status} {stage.name}: {stage.detail}\n")
        if stage.remediation:
            sys.stdout.write(f"  remediation: {stage.remediation}\n")
    sys.stdout.write(f"{report.status} overall: {len(report.stages)} setup stages\n")


def _command_detail(result: subprocess.CompletedProcess[str]) -> str:
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part and part.strip())
    return output[-4000:] if output else f"command exited {result.returncode}"


def _prerequisite_stage(selected: set[SetupStage]) -> SetupStageResult:
    tools = {"git": "install Git", "uv": "install uv"}
    if SetupStage.ROOT in selected or SetupStage.ASSETS in selected:
        tools.update(
            {
                "cmake": "install CMake",
                "ffmpeg": "install FFmpeg",
                "nvidia-smi": "install the NVIDIA driver",
            }
        )
        if shutil.which("cc") is None and shutil.which("gcc") is None:
            return _stage("prerequisites", "FAIL", "a C/C++ compiler is unavailable", "install build-essential")
    missing = [(name, remediation) for name, remediation in tools.items() if shutil.which(name) is None]
    if missing:
        names = ", ".join(name for name, _ in missing)
        return _stage("prerequisites", "FAIL", f"required tool(s) are unavailable: {names}", missing[0][1])
    if SetupStage.ASSETS in selected:
        lfs = subprocess.run(["git", "lfs", "version"], capture_output=True, text=True, check=False)
        if lfs.returncode != 0:
            return _stage("prerequisites", "FAIL", "git-lfs is unavailable", "install Git LFS")
    return _stage("prerequisites", "READY", "required host tools are available")


def _submodule_paths(paths: RepositoryPaths) -> list[Path]:
    result = subprocess.run(
        ["git", "config", "--file", ".gitmodules", "--get-regexp", "path"],
        cwd=paths.root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(_command_detail(result))
    return [paths.root / line.split(maxsplit=1)[1] for line in result.stdout.splitlines() if line.strip()]


def _submodules_stage(paths: RepositoryPaths) -> SetupStageResult:
    for submodule in _submodule_paths(paths):
        if not (submodule / ".git").exists():
            continue
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=submodule,
            capture_output=True,
            text=True,
            check=False,
        )
        if status.returncode != 0:
            return _stage("submodules", "FAIL", _command_detail(status), "inspect the submodule checkout")
        if status.stdout.strip():
            return _stage(
                "submodules",
                "FAIL",
                f"submodule has tracked or untracked changes: {submodule.relative_to(paths.root)}",
                "commit, move, or remove the submodule changes before make setup",
            )
    before = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        cwd=paths.root,
        capture_output=True,
        text=True,
        check=False,
    )
    if before.returncode != 0:
        return _stage("submodules", "FAIL", _command_detail(before), "inspect .gitmodules and pinned gitlinks")
    commands = (
        ["git", "submodule", "sync", "--recursive"],
        ["git", "submodule", "update", "--init", "--recursive", "--progress"],
    )
    for command in commands:
        result = subprocess.run(command, cwd=paths.root, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return _stage("submodules", "FAIL", _command_detail(result), "retry make setup after fixing Git access")
    changed = any(line.startswith(("-", "+", "U")) for line in before.stdout.splitlines())
    return _stage("submodules", "CHANGED" if changed else "READY", "pinned submodules are initialized")


def root_environment_error(paths: RepositoryPaths) -> str | None:
    uv = shutil.which("uv")
    python = paths.root / ".venv" / "bin" / "python"
    if uv is None:
        return "uv is unavailable"
    if not python.is_file():
        return f"locked simulator Python is missing: {python}"
    lock = subprocess.run(
        [uv, "lock", "--check", "--offline"],
        cwd=paths.root,
        capture_output=True,
        text=True,
        check=False,
    )
    if lock.returncode != 0:
        return f"uv.lock is stale: {_command_detail(lock)}"
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "from importlib.metadata import version; [version(n) for n in ('robodojo','torch','isaaclab')]",
        ],
        cwd=paths.root,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return f"locked simulator packages are incomplete: {_command_detail(probe)}"
    return None


def _root_environment_stage(paths: RepositoryPaths) -> SetupStageResult:
    if root_environment_error(paths) is None:
        return _stage("root_environment", "READY", f"locked simulator environment is ready: {paths.root / '.venv'}")
    environment = os.environ.copy()
    environment["OMNI_KIT_ACCEPT_EULA"] = environment.get("OMNI_KIT_ACCEPT_EULA", "YES")
    for command in (
        ["uv", "python", "install", "3.11"],
        ["uv", "sync", "--extra", "sim", "--locked"],
        ["uv", "lock", "--check"],
    ):
        result = subprocess.run(command, cwd=paths.root, env=environment, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return _stage(
                "root_environment",
                "FAIL",
                _command_detail(result),
                "fix the lockfile or uv access and retry",
            )
    if error := root_environment_error(paths):
        return _stage("root_environment", "FAIL", error, "inspect uv sync output")
    return _stage("root_environment", "CHANGED", "installed Python 3.11 and synchronized the locked simulator env")


def _asset_context(paths: RepositoryPaths, request: SetupRequest):
    profile = load_environment_profile(paths, request.env_config)
    simulator = SimulatorLaunchRequest(
        task=request.task,
        policy_name=request.policy_dir.name if request.policy_dir else "setup",
        port=1,
        env_config=request.env_config,
        scene_config=request.scene_config,
        additional_info="setup",
    )
    scene = resolve_scene_profile(paths, simulator)
    validate_scene_environment_compatibility(scene, profile)
    return profile, scene


def _asset_selection_stage(paths: RepositoryPaths, request: SetupRequest) -> SetupStageResult:
    tasks = {item["name"] for item in build_inventory()["tasks"] if item["runnable"]}
    if request.task not in tasks:
        return _stage(
            "experiment_selection",
            "FAIL",
            f"unknown or unrunnable task: {request.task}",
            "run make tasks and select a valid TASK",
        )
    try:
        profile, scene = _asset_context(paths, request)
    except (OSError, RuntimeError, ValueError, yaml.YAMLError) as exc:
        return _stage(
            "experiment_selection",
            "FAIL",
            str(exc),
            "select valid TASK, ENV_CFG, and SCENE values",
        )
    return _stage(
        "experiment_selection",
        "READY",
        f"{request.task} uses environment {profile.name} and scene {scene.name}",
    )


def _base_assets_stage(paths: RepositoryPaths) -> SetupStageResult:
    if assets_ready():
        return _stage("base_assets", "READY", f"base asset bundle is ready: {assets_root()}")
    download_assets(paths)
    if not assets_ready():
        return _stage("base_assets", "FAIL", "base asset download is incomplete", "inspect Git LFS access")
    return _stage("base_assets", "CHANGED", f"downloaded the base asset bundle to {assets_root()}")


def _generated_assets_stages(paths: RepositoryPaths, request: SetupRequest) -> list[SetupStageResult]:
    profile, scene = _asset_context(paths, request)
    stages: list[SetupStageResult] = []
    for name in required_robot_builds(profile):
        changed = ensure_generated_robot(paths, name)
        stages.append(_stage(f"robot_asset[{name}]", "CHANGED" if changed else "READY", "manifest verified"))
    for name in required_fixture_builds(scene, request.task):
        changed = ensure_generated_fixture(paths, name)
        stages.append(_stage(f"scene_asset[{name}]", "CHANGED" if changed else "READY", "manifest verified"))
    try:
        inspect_scene_assets(scene, request.task)
    except (FileNotFoundError, RuntimeError, ValueError):
        prepare_scene_assets(scene, request.task)
        stages.append(_stage("task_scene_assets", "CHANGED", f"prepared assets for {scene.name}/{request.task}"))
    else:
        stages.append(_stage("task_scene_assets", "READY", f"prepared assets verified for {scene.name}/{request.task}"))
    return stages


def _policy_stage(paths: RepositoryPaths, request: SetupRequest) -> SetupStageResult:
    try:
        assignment = resolve_gpus(policy_gpu=request.policy_gpu)
    except GpuSelectionError as exc:
        return _stage(
            "policy",
            "FAIL",
            f"GPU selection failed: {exc}",
            "set POLICY_GPU to 'auto' or an available nonnegative GPU index",
        )
    request = bind_policy_contract(paths, request.model_copy(update={"policy_gpu": assignment.policy_gpu}))
    policy = request.policy_request()
    prepare = policy_hook_command(policy, "prepare_eval_policy.sh")
    if prepare is None:
        return _stage(
            "policy",
            "WARN",
            "adapter has no prepare_eval_policy.sh; no policy state was changed",
            "follow the adapter README for legacy setup",
        )
    environment = {**os.environ, "ROBODOJO_ROOT": str(paths.root)}
    check = policy_hook_command(policy, "check_eval_policy.sh")
    if check is not None:
        ready = subprocess.run(
            check,
            cwd=policy.policy_dir.expanduser().resolve(),
            env={**environment, **policy_launch_environment(policy.checkpoint)},
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if ready.returncode == 0:
            return _stage("policy", "READY", _command_detail(ready))
    result = subprocess.run(
        prepare,
        cwd=policy.policy_dir.expanduser().resolve(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return _stage("policy", "FAIL", _command_detail(result), "fix the policy hook and retry make setup")
    if check is not None:
        verified = subprocess.run(
            check,
            cwd=policy.policy_dir.expanduser().resolve(),
            env={**environment, **policy_launch_environment(policy.checkpoint)},
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if verified.returncode not in {0, 3}:
            return _stage("policy", "FAIL", _command_detail(verified), "inspect the policy setup output")
        if verified.returncode == 3:
            return _stage("policy", "WARN", _command_detail(verified), "review the policy adapter warning")
    detail = "policy preparation hook completed"
    if check is not None:
        detail += " and validation passed"
    return _stage("policy", "CHANGED", detail)


def run_setup(paths: RepositoryPaths, request: SetupRequest) -> SetupReport:
    """Prepare selected setup stages, stopping at the first failed dependency."""

    selected = set(request.selected_stages())
    records = [_prerequisite_stage(selected)]
    if records[-1].status == "FAIL":
        return build_report(records)
    if SetupStage.ASSETS in selected:
        records.append(_asset_selection_stage(paths, request))
        if records[-1].status == "FAIL":
            return build_report(records)

    actions: list[Callable[[], SetupStageResult | list[SetupStageResult]]] = []
    if SetupStage.ROOT in selected:
        actions.extend((lambda: _submodules_stage(paths), lambda: _root_environment_stage(paths)))
    elif SetupStage.ASSETS in selected and (error := root_environment_error(paths)):
        records.append(_stage("root_environment", "FAIL", error, "robodojo setup --only root"))
        return build_report(records)
    if SetupStage.ASSETS in selected:
        actions.extend((lambda: _base_assets_stage(paths), lambda: _generated_assets_stages(paths, request)))
    if SetupStage.POLICY in selected:
        actions.append(lambda: _policy_stage(paths, request))

    for action in actions:
        try:
            result = action()
        except (
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            ValueError,
            yaml.YAMLError,
            json.JSONDecodeError,
        ) as exc:
            records.append(_stage("setup", "FAIL", str(exc), "fix the reported prerequisite and retry make setup"))
            break
        records.extend(result if isinstance(result, list) else [result])
        if records[-1].status == "FAIL":
            break
    return build_report(records)


def setup(paths: RepositoryPaths, request: SetupRequest, *, output_format: str = "human") -> int:
    report = run_setup(paths, request)
    emit_report(report, output_format)
    return 1 if report.status == "FAIL" else 0
