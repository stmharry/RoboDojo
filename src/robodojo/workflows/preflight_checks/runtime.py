"""Read-only experiment validation."""

from __future__ import annotations

import shlex

from robodojo.core.gpu import GpuSelectionError, resolve_gpus, validate_gpu_assignment
from robodojo.core.models.reports import (
    PreflightCheck,
)
from robodojo.core.models.requests import (
    PreflightRequest,
)
from robodojo.workflows.preflight_checks.reporting import _check

HOOK_WARNING_EXIT = 3
ROOT_SETUP_REMEDIATION = "make setup; or " + shlex.join(
    ["uv", "run", "--locked", "robodojo", "setup", "--only", "root"]
)


def _resolve_preflight_gpus(
    request: PreflightRequest,
    *,
    simulator_only: bool = False,
) -> tuple[PreflightRequest | None, PreflightCheck]:
    policy_selector = None if simulator_only else request.policy_gpu
    try:
        assignment = resolve_gpus(policy_gpu=policy_selector, env_gpu=request.environment_gpu)
        if policy_selector != "auto" and request.environment_gpu != "auto":
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
    updates: dict[str, int] = {"environment_gpu": assignment.env_gpu}
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
