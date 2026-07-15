"""Read-only experiment validation and explicit policy setup orchestration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable

import yaml

from robodojo.core.models import (
    EvaluationRequest,
    PolicyServerLaunchRequest,
    PreflightCheck,
    PreflightReport,
    PreflightRequest,
    SimulatorLaunchRequest,
)
from robodojo.core.paths import RepositoryPaths
from robodojo.core.processes import free_port, start, terminate_process_group, wait_for_port
from robodojo.core.profiles import EnvironmentProfile, SceneProfile, load_environment_profile
from robodojo.core.storage import assets_root, s3_uri
from robodojo.policy.adapter import policy_hook_command, policy_launch_environment, policy_server_command
from robodojo.sim.launcher import resolve_scene_profile
from robodojo.workflows.task_inventory import build_inventory

HOOK_WARNING_EXIT = 3


def request_from_evaluation(request: EvaluationRequest, *, task: str | None = None) -> PreflightRequest:
    """Project an evaluation request onto the shared fast-preflight contract."""
    return PreflightRequest(
        policy_dir=request.policy_dir,
        task=task or request.task,
        checkpoint=request.checkpoint,
        policy_env=request.policy_env,
        dataset=request.dataset,
        env_config=request.env_config,
        scene_config=request.scene_config,
        action_type=request.action_type,
        seed=request.seed,
        policy_gpu=request.policy_gpu,
        env_gpu=request.env_gpu,
        publish=request.publish,
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
    uv = shutil.which("uv")
    python = paths.root / ".venv" / "bin" / "python"
    if uv is None:
        return _check("root_sim_environment", "FAIL", "uv is not installed", "make sync")
    if not python.is_file():
        return _check("root_sim_environment", "FAIL", f"locked environment is missing: {python}", "make sync")
    if not (paths.root / "uv.lock").is_file():
        return _check("root_sim_environment", "FAIL", "uv.lock is missing", "make sync")
    lock = subprocess.run(
        [uv, "lock", "--check", "--offline"],
        cwd=paths.root,
        capture_output=True,
        text=True,
        check=False,
    )
    if lock.returncode != 0:
        return _check("root_sim_environment", "FAIL", f"uv.lock is stale: {_command_detail(lock)}", "make sync")
    probe = subprocess.run(
        [
            str(python),
            "-c",
            ("from importlib.metadata import version; [version(name) for name in ('robodojo','torch','isaaclab')]"),
        ],
        cwd=paths.root,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return _check(
            "root_sim_environment",
            "FAIL",
            f"locked simulator packages are incomplete: {_command_detail(probe)}",
            "make sync",
        )
    return _check("root_sim_environment", "PASS", f"uv.lock and simulator packages are ready in {python.parent.parent}")


def _configuration_checks(
    paths: RepositoryPaths,
    request: PreflightRequest,
) -> tuple[list[PreflightCheck], EnvironmentProfile | None, SceneProfile | None]:
    checks: list[PreflightCheck] = []
    inventory = build_inventory()
    record = next((item for item in inventory["tasks"] if item["name"] == request.task), None)
    if record is None:
        checks.append(_check("task", "FAIL", f"unknown task: {request.task}", "make tasks-check"))
    elif not record["runnable"]:
        checks.append(_check("task", "FAIL", f"task is not runnable: {request.task}", "make tasks-check"))
    else:
        checks.append(_check("task", "PASS", f"{request.task} code and YAML are runnable"))

    try:
        profile = load_environment_profile(paths, request.env_config)
        checks.append(_check("environment", "PASS", str(profile.path)))
    except Exception as exc:
        checks.append(_check("environment", "FAIL", str(exc), "make tasks-check"))
        return checks, None, None

    simulator_request = SimulatorLaunchRequest(
        task=request.task,
        policy_name=request.policy_dir.name,
        port=1,
        env_config=request.env_config,
        scene_config=request.scene_config,
        env_gpu=request.env_gpu,
        seed=request.seed,
        additional_info="preflight",
    )
    try:
        scene = resolve_scene_profile(paths, simulator_request)
        checks.append(_check("scene", "PASS", f"{scene.name} -> {scene.component_path}"))
    except Exception as exc:
        checks.append(_check("scene", "FAIL", str(exc), "make tasks-check"))
        return checks, profile, None
    return checks, profile, scene


def _layout_check(paths: RepositoryPaths, request: PreflightRequest, scene: SceneProfile | None) -> PreflightCheck:
    if scene is None:
        return _check("layout", "FAIL", "scene did not resolve", "make tasks-check")
    relative = Path(scene.document.layout_set) / str(request.seed)
    candidates = (
        assets_root() / "Eval_Layout" / request.dataset / relative,
        paths.environment_configs / "layout" / relative,
    )
    for directory in candidates:
        matches = sorted(directory.glob(f"{request.task}_*.json")) if directory.is_dir() else []
        if not matches:
            continue
        try:
            for path in matches:
                json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return _check("layout", "FAIL", f"invalid layout {path}: {exc}", "make assets")
        return _check("layout", "PASS", f"{len(matches)} layout(s) in {directory}")
    searched = ", ".join(str(path) for path in candidates)
    return _check(
        "layout",
        "FAIL",
        f"no {request.task}_*.json layout for seed {request.seed}; searched {searched}",
        "make assets",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _robot_asset_check(profile: EnvironmentProfile | None) -> PreflightCheck:
    if profile is None:
        return _check("robot_assets", "FAIL", "environment did not resolve", "make sync")
    robot_config = yaml.safe_load(profile.component_paths["robot"].read_text(encoding="utf-8")) or {}
    robot_names = sorted({str(item.get("robot_name", "")) for item in robot_config.get("robots", [])})
    generated = [name for name in robot_names if name in {"yam", "openarm"}]
    if not generated:
        return _check("robot_assets", "PASS", f"no generated manifest required for {', '.join(robot_names)}")
    verified: list[str] = []
    for name in generated:
        root = assets_root() / "Robots" / name
        manifest_path = root / "manifest.json"
        remediation = (
            "make assets-yam" if name == "yam" else "uv run --extra sim --locked robodojo assets build-openarm"
        )
        if not manifest_path.is_file():
            return _check("robot_assets", "FAIL", f"generated {name} manifest is missing: {manifest_path}", remediation)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return _check("robot_assets", "FAIL", f"invalid {name} manifest: {exc}", remediation)
        outputs = manifest.get("outputs")
        if isinstance(outputs, dict):
            for relative, expected in outputs.items():
                output = root / relative
                if not output.is_file():
                    return _check("robot_assets", "FAIL", f"generated asset is missing: {output}", remediation)
                if _sha256(output) != expected:
                    return _check("robot_assets", "FAIL", f"generated asset checksum mismatch: {output}", remediation)
        else:
            output_name = manifest.get("output")
            if output_name and not (root / str(output_name)).is_file():
                return _check(
                    "robot_assets",
                    "FAIL",
                    f"generated asset is missing: {root / str(output_name)}",
                    remediation,
                )
        verified.append(name)
    return _check("robot_assets", "PASS", f"verified generated manifest(s): {', '.join(verified)}")


def _gpu_check(request: PreflightRequest) -> PreflightCheck:
    tool = shutil.which("nvidia-smi")
    if tool is None:
        return _check("gpu_indices", "FAIL", "nvidia-smi is unavailable", "install the NVIDIA driver")
    result = subprocess.run(
        [tool, "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return _check("gpu_indices", "FAIL", _command_detail(result), "verify the NVIDIA driver with nvidia-smi")
    available = {int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()}
    invalid = [value for value in (request.policy_gpu, request.env_gpu) if value not in available]
    if invalid:
        return _check(
            "gpu_indices",
            "FAIL",
            f"invalid GPU index/indices {invalid}; available: {sorted(available)}",
            "set POLICY_GPU and ENV_GPU to indices shown by nvidia-smi",
        )
    return _check(
        "gpu_indices",
        "PASS",
        f"policy GPU {request.policy_gpu} and simulator GPU {request.env_gpu} are available",
    )


def _publication_check(request: PreflightRequest) -> PreflightCheck:
    if not request.publish:
        return _check("publication", "PASS", "publication disabled")
    remote = s3_uri()
    if remote is None or not remote.startswith("s3://"):
        return _check(
            "publication",
            "FAIL",
            "ROBODOJO_S3_URI is not a dedicated s3:// prefix",
            "configure ROBODOJO_S3_URI in .env",
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
        return [_check("policy_runtime", "FAIL", error or "policy runtime did not resolve", "make policy-setup")]
    checks: list[PreflightCheck] = []
    if environment_root and (environment_root / "pyproject.toml").is_file():
        lock_path = environment_root / "uv.lock"
        uv = shutil.which("uv")
        if not lock_path.is_file():
            checks.append(_check("policy_runtime", "FAIL", f"uv.lock is missing: {lock_path}", "make policy-setup"))
            return checks
        if uv is None:
            checks.append(_check("policy_runtime", "FAIL", "uv is unavailable", "make policy-setup"))
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
                    "make policy-setup",
                )
            )
            return checks
    checks.append(_check("policy_runtime", "PASS", str(python)))
    import_env = os.environ.copy()
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
        checks.append(_check("xpolicylab_import", "FAIL", _command_detail(probe), "make policy-setup"))
    else:
        checks.append(_check("xpolicylab_import", "PASS", "XPolicyLab server imports succeeded in the policy runtime"))
    return checks


def _checkpoint_check(request: PreflightRequest) -> PreflightCheck:
    raw = request.checkpoint.strip()
    policy_dir = request.policy_dir.expanduser().resolve()
    expanded = Path(os.path.expanduser(raw))
    explicit = expanded.is_absolute() or "/" in raw or raw.startswith(".") or (policy_dir / expanded).exists()
    if explicit:
        path = expanded if expanded.is_absolute() else policy_dir / expanded
        path = path.resolve()
        if not path.exists():
            return _check("checkpoint", "FAIL", f"explicit checkpoint does not exist: {path}", "make policy-setup")
        return _check("checkpoint", "PASS", f"explicit checkpoint exists: {path}")
    if policy_hook_command(request.policy_request(), "check_eval_policy.sh") is not None:
        return _check("checkpoint", "PASS", f"opaque alias {raw!r} delegated to the policy-specific hook")
    return _check(
        "checkpoint",
        "WARN",
        f"opaque checkpoint alias {raw!r} requires a policy-specific hook for integrity validation",
        "make policy-setup",
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
    environment = os.environ.copy()
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
        return _check("policy_specific", "FAIL", "policy check timed out after 120s", "make policy-setup")
    detail = _command_detail(result)
    if result.returncode == 0:
        return _check("policy_specific", "PASS", detail)
    if result.returncode == HOOK_WARNING_EXIT:
        return _check("policy_specific", "WARN", detail, "make policy-setup")
    return _check("policy_specific", "FAIL", detail, "make policy-setup")


def run_fast_preflight(paths: RepositoryPaths, request: PreflightRequest) -> PreflightReport:
    """Run every read-only check without allocating a port or starting a process."""
    checks: list[PreflightCheck] = [_root_runtime_check(paths)]
    config_checks, profile, scene = _configuration_checks(paths, request)
    checks.extend(config_checks)
    checks.append(_layout_check(paths, request, scene))
    checks.append(_robot_asset_check(profile))
    checks.append(_gpu_check(request))
    checks.append(_publication_check(request))
    checks.append(_adapter_files_check(request))
    checks.extend(_policy_runtime_checks(paths, request))
    checks.append(_checkpoint_check(request))
    checks.append(_policy_hook_check(paths, request))
    return build_report(checks)


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
        config_checks, _, scene = _configuration_checks(paths, task_request)
        checks.extend(
            item.model_copy(update={"name": f"{item.name}[{task}]"})
            for item in config_checks
            if item.name in {"task", "scene"}
        )
        layout = _layout_check(paths, task_request, scene)
        checks.append(layout.model_copy(update={"name": f"layout[{task}]"}))
    return build_report(checks)


def run_deep_preflight(paths: RepositoryPaths, request: PreflightRequest) -> PreflightReport:
    """Run fast checks, then start and always stop the normal policy server."""
    report = run_fast_preflight(paths, request)
    if report.status == "FAIL":
        return report
    process = None
    port = free_port()
    policy_request = request.policy_request(port=port)
    command = policy_server_command(policy_request, port)
    try:
        process = start(
            command,
            cwd=request.policy_dir.expanduser().resolve(),
            env=policy_launch_environment(request.checkpoint),
        )
        wait_for_port(process, "127.0.0.1", port, timeout=request.timeout)
        check = _check("deep_policy_server", "PASS", f"normal policy server became ready on temporary port {port}")
    except (OSError, RuntimeError, TimeoutError) as exc:
        check = _check("deep_policy_server", "FAIL", str(exc), "make policy-setup")
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


def run_policy_setup(
    paths: RepositoryPaths,
    request: PolicyServerLaunchRequest,
) -> tuple[PreflightReport, str]:
    """Run the one explicit policy-owned setup mutation hook."""
    command = policy_hook_command(request, "prepare_eval_policy.sh")
    if command is None:
        report = build_report(
            [
                _check(
                    "policy_setup",
                    "WARN",
                    "adapter has no prepare_eval_policy.sh; no changes were made",
                    "follow the adapter README for legacy setup",
                )
            ]
        )
        return report, ""
    environment = os.environ.copy()
    environment["ROBODOJO_ROOT"] = str(paths.root)
    result = subprocess.run(
        command,
        cwd=request.policy_dir.expanduser().resolve(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    transcript = "\n".join(part.rstrip() for part in (result.stdout, result.stderr) if part and part.strip())
    if result.returncode == 0:
        detail = transcript.splitlines()[-1] if transcript else "policy setup completed successfully"
        report = build_report([_check("policy_setup", "PASS", detail)])
    else:
        detail = transcript[-4000:] if transcript else f"prepare_eval_policy.sh exited {result.returncode}"
        report = build_report([_check("policy_setup", "FAIL", detail, "make policy-setup")])
    return report, transcript
