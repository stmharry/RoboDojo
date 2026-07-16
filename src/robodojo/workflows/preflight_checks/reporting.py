"""Read-only experiment validation."""

from __future__ import annotations

import shlex
import subprocess
import sys
from typing import Iterable

from robodojo.core.models.reports import (
    PreflightCheck,
    PreflightReport,
)
from robodojo.core.models.requests import (
    PreflightRequest,
)

HOOK_WARNING_EXIT = 3
ROOT_SETUP_REMEDIATION = "make setup; or " + shlex.join(
    ["uv", "run", "--locked", "robodojo", "setup", "--only", "root"]
)


def _setup_remediation(request: PreflightRequest, stage: str) -> str:
    arguments = ["uv", "run", "--locked", "robodojo", "setup", "--only", stage]
    if request.experiment.recipe:
        arguments += ["--recipe", request.experiment.recipe, "--seed", str(request.seed)]
        if stage == "policy":
            arguments += ["--policy-gpu", str(request.policy_gpu)]
        return f"make setup RECIPE={shlex.quote(request.experiment.recipe)}; or {shlex.join(arguments)}"
    return "make setup with the same complete manual contract"


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
