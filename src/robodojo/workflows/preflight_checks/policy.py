"""Read-only experiment validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import yaml

from robodojo.core.models.reports import (
    PreflightCheck,
)
from robodojo.core.models.requests import (
    PreflightRequest,
)
from robodojo.core.paths import RepositoryPaths
from robodojo.policy.adapter import policy_hook_command, policy_launch_environment
from robodojo.workflows.preflight_checks.reporting import _check, _command_detail, _setup_remediation

HOOK_WARNING_EXIT = 3
ROOT_SETUP_REMEDIATION = "make setup; or " + shlex.join(
    ["uv", "run", "--locked", "robodojo", "setup", "--only", "root"]
)


def _adapter_files_check(request: PreflightRequest) -> PreflightCheck:
    policy_dir = request.experiment.policy_dir.expanduser().resolve()
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
    policy_dir = request.experiment.policy_dir.expanduser().resolve()
    raw = request.experiment.policy_runtime.strip()
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
    import_env = {**os.environ, **policy_launch_environment(request.experiment.checkpoint)}
    import_env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(paths.root), import_env.get("PYTHONPATH", ""))))
    probe = subprocess.run(
        [str(python), "-c", "import XPolicyLab, client_server, utils"],
        cwd=request.experiment.policy_dir.expanduser().resolve(),
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
    raw = request.experiment.checkpoint.strip()
    policy_dir = request.experiment.policy_dir.expanduser().resolve()
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
    environment = {**os.environ, **policy_launch_environment(request.experiment.checkpoint)}
    environment["ROBODOJO_ROOT"] = str(paths.root)
    try:
        result = subprocess.run(
            command,
            cwd=request.experiment.policy_dir.expanduser().resolve(),
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
