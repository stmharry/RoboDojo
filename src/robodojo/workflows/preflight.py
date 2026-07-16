"""Read-only experiment validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Iterable

import yaml

from robodojo.core.contracts import load_protocol_catalog, resolve_contract
from robodojo.core.gpu import GpuSelectionError, resolve_gpus, validate_gpu_assignment
from robodojo.core.layouts import resolve_layout_set
from robodojo.core.models import (
    ExperimentRequest,
    PreflightCheck,
    PreflightReport,
    PreflightRequest,
    SimulatorLaunchRequest,
)
from robodojo.core.paths import RepositoryPaths
from robodojo.core.processes import free_port, start, terminate_process_group, wait_for_port
from robodojo.core.profiles import (
    EnvironmentProfile,
    SceneProfile,
    load_environment_profile,
    validate_scene_environment_compatibility,
)
from robodojo.core.storage import assets_root, s3_uri
from robodojo.core.workspace import validate_layout_contract
from robodojo.policy.adapter import policy_hook_command, policy_launch_environment, policy_server_command
from robodojo.sim.launcher import resolve_scene_profile
from robodojo.sim.scene_assets import inspect_scene_assets
from robodojo.workflows.assets import (
    generated_fixture_error,
    generated_robot_error,
    required_fixture_builds,
    required_robot_builds,
)
from robodojo.workflows.setup import root_environment_error
from robodojo.workflows.task_inventory import build_inventory

HOOK_WARNING_EXIT = 3
ROOT_SETUP_REMEDIATION = "make setup; or " + shlex.join(
    ["uv", "run", "--locked", "robodojo", "setup", "--only", "root"]
)


def _setup_remediation(request: PreflightRequest, stage: str) -> str:
    arguments = ["uv", "run", "--locked", "robodojo", "setup", "--only", stage]
    if request.recipe:
        arguments += ["--recipe", request.recipe, "--seed", str(request.seed)]
        if stage == "policy":
            arguments += ["--policy-gpu", str(request.policy_gpu)]
        return f"make setup RECIPE={shlex.quote(request.recipe)}; or {shlex.join(arguments)}"
    return "make setup with the same complete manual contract"


def request_from_evaluation(request: ExperimentRequest, *, task: str | None = None) -> PreflightRequest:
    """Project an evaluation request onto the shared fast-preflight contract."""
    return PreflightRequest(
        policy_dir=request.policy_dir,
        task=task or request.task,
        checkpoint=request.checkpoint,
        policy_profile=request.policy_profile,
        policy_descriptor_hash=request.policy_descriptor_hash,
        policy_reference_match=request.policy_reference_match,
        policy_env=request.policy_env,
        dataset=request.dataset,
        env_config=request.env_config,
        policy_contract=request.policy_contract,
        recipe=request.recipe,
        contract_hash=request.contract_hash,
        protocol=request.protocol,
        episode_horizon=request.episode_horizon,
        native_eval_num=request.native_eval_num,
        scene_config=request.scene_config,
        action_type=request.action_type,
        seed=request.seed,
        policy_gpu=request.policy_gpu,
        env_gpu=request.env_gpu,
        publish=getattr(request, "publish", False),
    )


def _check(
    name: str,
    status: str,
    detail: str,
    remediation: str | None = None,
) -> PreflightCheck:
    return PreflightCheck(name=name, status=status, detail=detail, remediation=remediation)


def build_report(checks: Iterable[PreflightCheck]) -> PreflightReport:
    records = list(checks)
    if any(item.status == "FAIL" for item in records):
        status = "FAIL"
    elif any(item.status == "WARN" for item in records):
        status = "WARN"
    else:
        status = "PASS"
    return PreflightReport(status=status, checks=records)


def emit_report(report: PreflightReport, output_format: str = "human") -> None:
    if output_format == "json":
        sys.stdout.write(report.model_dump_json(indent=2, exclude_none=True) + "\n")
        return
    if output_format != "human":
        raise ValueError(f"unsupported report format: {output_format}")
    for item in report.checks:
        sys.stdout.write(f"{item.status} {item.name}: {item.detail}\n")
        if item.remediation:
            sys.stdout.write(f"  remediation: {item.remediation}\n")
    sys.stdout.write(f"{report.status} overall: {len(report.checks)} checks\n")


def _command_detail(result: subprocess.CompletedProcess[str]) -> str:
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part and part.strip())
    return output[-4000:] if output else f"command exited {result.returncode}"


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
    record = next((item for item in inventory["tasks"] if item["name"] == request.task), None)
    if record is None:
        checks.append(
            _check(
                "task",
                "FAIL",
                f"unknown task: {request.task}",
                "run make tasks and select a valid TASK",
            )
        )
    elif not record["runnable"]:
        checks.append(
            _check(
                "task",
                "FAIL",
                f"task is not runnable: {request.task}",
                "run make tasks and select a runnable TASK",
            )
        )
    else:
        checks.append(_check("task", "PASS", f"{request.task} code and YAML are runnable"))

    try:
        profile = load_environment_profile(paths, request.env_config)
        checks.append(_check("environment", "PASS", str(profile.path)))
    except Exception as exc:
        checks.append(_check("environment", "FAIL", str(exc), "select a valid ENV_CFG"))
        return checks, None, None

    if request.policy_contract != profile.policy_contract:
        checks.append(
            _check(
                "policy_environment",
                "FAIL",
                f"policy embodiment {request.policy_contract!r} does not match "
                f"environment contract {profile.policy_contract!r}",
                "select a compatible policy profile and environment recipe",
            )
        )
    else:
        checks.append(_check("policy_environment", "PASS", profile.policy_contract))

    try:
        protocol = load_protocol_catalog(paths).protocols[request.protocol]
        actual = (request.task, request.episode_horizon, request.native_eval_num)
        expected = (protocol.task, protocol.episode_horizon, protocol.evaluation_episodes)
        if actual != expected:
            raise ValueError(f"resolved fields {actual} do not match catalog {expected}")
        if protocol.compatible_scenes and request.scene_config not in protocol.compatible_scenes:
            raise ValueError(f"compatible scenes are {protocol.compatible_scenes}, received {request.scene_config!r}")
        checks.append(_check("protocol", "PASS", f"{request.protocol} -> task={request.task}"))
    except (KeyError, TypeError, ValueError) as exc:
        checks.append(_check("protocol", "FAIL", str(exc), "select a valid recipe or complete manual contract"))

    simulator_request = SimulatorLaunchRequest(
        task=request.task,
        protocol_name=request.protocol,
        episode_horizon=request.episode_horizon,
        native_eval_num=request.native_eval_num,
        recipe=request.recipe,
        contract_hash=request.contract_hash,
        policy_name=request.policy_dir.name,
        port=1,
        env_config=request.env_config,
        scene_config=request.scene_config,
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

    if request.policy_profile == "manual":
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
            resolved = resolve_contract(
                paths,
                policy_name=request.policy_profile,
                environment_name=request.env_config,
                scene_name=request.scene_config,
                protocol_name=request.protocol,
                recipe_name=request.recipe,
            )
            actual = (
                request.checkpoint,
                request.policy_contract,
                request.action_type,
                request.dataset,
                request.policy_descriptor_hash,
                request.contract_hash,
                request.policy_reference_match,
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
                        f"{request.policy_profile} is running outside its declared reference setup",
                    )
                )
            else:
                checks.append(
                    _check(
                        "policy_descriptor",
                        "PASS",
                        f"{request.policy_profile} descriptor={resolved.policy_descriptor_hash}",
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
            benchmark=request.dataset,
            layout_set=scene.document.layout_set,
            layout_source=scene.document.layout_source,
            task=request.task,
            seed=request.seed,
        )
        task_config = yaml.safe_load((paths.task_configs / f"{request.task}.yml").read_text(encoding="utf-8")) or {}
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
            f"invalid task-keyed layout for {request.task}: {exc}",
            _setup_remediation(request, "assets"),
        )
    return _check(
        "layout",
        "PASS",
        f"{len(resolved.layouts)} {resolved.directory} layout(s) keyed by task {request.task}",
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
    required_builds = required_fixture_builds(scene, request.task)
    for name in required_builds:
        if error := generated_fixture_error(paths, name):
            return _check("scene_assets", "FAIL", error, remediation)
    try:
        prepared = inspect_scene_assets(scene, request.task)
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


def _resolve_preflight_gpus(
    request: PreflightRequest,
    *,
    simulator_only: bool = False,
) -> tuple[PreflightRequest | None, PreflightCheck]:
    policy_selector = None if simulator_only else request.policy_gpu
    try:
        assignment = resolve_gpus(policy_gpu=policy_selector, env_gpu=request.env_gpu)
        if policy_selector != "auto" and request.env_gpu != "auto":
            validate_gpu_assignment(policy_gpu=assignment.policy_gpu, env_gpu=assignment.env_gpu)
    except GpuSelectionError as exc:
        variables = "ENV_GPU" if simulator_only else "POLICY_GPU and ENV_GPU"
        return None, _check(
            "gpu_indices",
            "FAIL",
            str(exc),
            f"set {variables} to 'auto' or available nonnegative GPU indices",
        )

    assert assignment.env_gpu is not None
    updates: dict[str, int] = {"env_gpu": assignment.env_gpu}
    if not simulator_only:
        assert assignment.policy_gpu is not None
        updates["policy_gpu"] = assignment.policy_gpu
        detail = (
            f"policy GPU {assignment.policy_gpu} ({assignment.policy_source}) and "
            f"simulator GPU {assignment.env_gpu} ({assignment.env_source}) are available"
        )
    else:
        detail = f"simulator GPU {assignment.env_gpu} ({assignment.env_source}) is available"
    return request.model_copy(update=updates), _check("gpu_indices", "PASS", detail)


def _gpu_check(request: PreflightRequest) -> PreflightCheck:
    """Resolve and validate the paired GPU contract for focused checks."""
    _, check = _resolve_preflight_gpus(request)
    return check


def _publication_check(request: PreflightRequest) -> PreflightCheck:
    if not request.publish:
        return _check("publication", "PASS", "publication disabled")
    remote = s3_uri()
    if remote is None or not remote.startswith("s3://"):
        return _check(
            "publication",
            "FAIL",
            "ROBODOJO_S3_URI is not a dedicated s3:// prefix",
            "export ROBODOJO_S3_URI in the process environment",
        )
    aws = shutil.which("aws")
    if aws is None:
        return _check("publication", "FAIL", "AWS CLI is unavailable", "install and configure the AWS CLI")
    return _check("publication", "PASS", f"AWS CLI and destination {remote} are configured")


def _adapter_files_check(request: PreflightRequest) -> PreflightCheck:
    policy_dir = request.policy_dir.expanduser().resolve()
    missing = [name for name in ("setup_eval_policy_server.sh", "deploy.yml") if not (policy_dir / name).is_file()]
    if missing:
        return _check(
            "policy_adapter",
            "FAIL",
            f"missing adapter file(s) in {policy_dir}: {', '.join(missing)}",
            "select a valid POLICY_DIR",
        )
    return _check("policy_adapter", "PASS", str(policy_dir))


def _resolve_policy_python(request: PreflightRequest) -> tuple[Path | None, Path | None, str | None]:
    policy_dir = request.policy_dir.expanduser().resolve()
    raw = request.policy_env.strip()
    project: Path | None = None
    if raw == "uv":
        deploy_path = policy_dir / "deploy.yml"
        try:
            deploy = yaml.safe_load(deploy_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            return None, None, f"could not read {deploy_path}: {exc}"
        raw = str(deploy.get("policy_uv_env_path", "")).strip()
        if not raw:
            return None, None, f"policy_uv_env_path is missing from {deploy_path}"
        project = Path(os.path.expanduser(raw))
    elif "/" in raw or raw.startswith(".") or Path(os.path.expanduser(raw)).is_absolute():
        project = Path(os.path.expanduser(raw))

    if project is not None:
        if not project.is_absolute():
            project = policy_dir / project
        project = project.resolve()
        direct_python = project / "bin" / "python"
        python = direct_python if direct_python.is_file() else project / ".venv" / "bin" / "python"
        if not python.is_file():
            return None, project, f"policy environment Python is missing: {python}"
        return python, project, None

    conda = shutil.which("conda")
    if conda is None:
        return None, None, f"Conda environment {raw!r} cannot be resolved because conda is unavailable"
    result = subprocess.run([conda, "env", "list", "--json"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None, None, f"could not inspect Conda environments: {_command_detail(result)}"
    try:
        prefixes = [Path(value) for value in json.loads(result.stdout).get("envs", [])]
    except (json.JSONDecodeError, AttributeError) as exc:
        return None, None, f"invalid conda env list: {exc}"
    prefix = next((value for value in prefixes if value.name == raw), None)
    if prefix is None:
        return None, None, f"Conda environment does not exist: {raw}"
    python = prefix / "bin" / "python"
    if not python.is_file():
        return None, prefix, f"Conda environment Python is missing: {python}"
    return python, prefix, None


def _policy_runtime_checks(paths: RepositoryPaths, request: PreflightRequest) -> list[PreflightCheck]:
    remediation = _setup_remediation(request, "policy")
    if policy_hook_command(request.policy_request(), "check_eval_policy.sh") is not None:
        return [
            _check(
                "policy_runtime",
                "PASS",
                "runtime resolution and XPolicyLab imports delegated to the policy-specific read-only hook",
            )
        ]
    python, environment_root, error = _resolve_policy_python(request)
    if error or python is None:
        return [_check("policy_runtime", "FAIL", error or "policy runtime did not resolve", remediation)]
    checks: list[PreflightCheck] = []
    if environment_root and (environment_root / "pyproject.toml").is_file():
        lock_path = environment_root / "uv.lock"
        uv = shutil.which("uv")
        if not lock_path.is_file():
            checks.append(_check("policy_runtime", "FAIL", f"uv.lock is missing: {lock_path}", remediation))
            return checks
        if uv is None:
            checks.append(_check("policy_runtime", "FAIL", "uv is unavailable", remediation))
            return checks
        locked = subprocess.run(
            [uv, "lock", "--check", "--offline"],
            cwd=environment_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if locked.returncode != 0:
            checks.append(
                _check(
                    "policy_runtime",
                    "FAIL",
                    f"policy uv.lock is stale: {_command_detail(locked)}",
                    remediation,
                )
            )
            return checks
    checks.append(_check("policy_runtime", "PASS", str(python)))
    import_env = {**os.environ, **policy_launch_environment(request.checkpoint)}
    import_env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(paths.root), import_env.get("PYTHONPATH", ""))))
    probe = subprocess.run(
        [str(python), "-c", "import XPolicyLab, client_server, utils"],
        cwd=request.policy_dir.expanduser().resolve(),
        env=import_env,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        checks.append(_check("xpolicylab_import", "FAIL", _command_detail(probe), remediation))
    else:
        checks.append(_check("xpolicylab_import", "PASS", "XPolicyLab server imports succeeded in the policy runtime"))
    return checks


def _checkpoint_check(request: PreflightRequest) -> PreflightCheck:
    remediation = _setup_remediation(request, "policy")
    raw = request.checkpoint.strip()
    policy_dir = request.policy_dir.expanduser().resolve()
    expanded = Path(os.path.expanduser(raw))
    explicit = expanded.is_absolute() or "/" in raw or raw.startswith(".") or (policy_dir / expanded).exists()
    if explicit:
        path = expanded if expanded.is_absolute() else policy_dir / expanded
        path = path.resolve()
        if not path.exists():
            return _check("checkpoint", "FAIL", f"explicit checkpoint does not exist: {path}", remediation)
        return _check("checkpoint", "PASS", f"explicit checkpoint exists: {path}")
    if policy_hook_command(request.policy_request(), "check_eval_policy.sh") is not None:
        return _check("checkpoint", "PASS", f"opaque alias {raw!r} delegated to the policy-specific hook")
    return _check(
        "checkpoint",
        "WARN",
        f"opaque checkpoint alias {raw!r} requires a policy-specific hook for integrity validation",
        remediation,
    )


def _policy_hook_check(paths: RepositoryPaths, request: PreflightRequest) -> PreflightCheck:
    command = policy_hook_command(request.policy_request(), "check_eval_policy.sh")
    if command is None:
        return _check(
            "policy_specific",
            "WARN",
            "adapter has no read-only check_eval_policy.sh; generic checks only",
            "add check_eval_policy.sh using the documented eight-argument contract",
        )
    environment = {**os.environ, **policy_launch_environment(request.checkpoint)}
    environment["ROBODOJO_ROOT"] = str(paths.root)
    try:
        result = subprocess.run(
            command,
            cwd=request.policy_dir.expanduser().resolve(),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return _check(
            "policy_specific",
            "FAIL",
            "policy check timed out after 120s",
            _setup_remediation(request, "policy"),
        )
    detail = _command_detail(result)
    if result.returncode == 0:
        return _check("policy_specific", "PASS", detail)
    if result.returncode == HOOK_WARNING_EXIT:
        return _check("policy_specific", "WARN", detail, _setup_remediation(request, "policy"))
    return _check("policy_specific", "FAIL", detail, _setup_remediation(request, "policy"))


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
        if task == request.task:
            continue
        task_request = request.model_copy(update={"task": task})
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
            cwd=resolved.policy_dir.expanduser().resolve(),
            env=policy_launch_environment(resolved.checkpoint),
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
