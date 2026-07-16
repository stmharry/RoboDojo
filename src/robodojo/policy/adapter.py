"""Validation and process launching for XPolicyLab policy adapters."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

from robodojo.core.models.requests import PolicyServerLaunchRequest
from robodojo.core.processes import format_command, free_port, run
from robodojo.core.storage import checkpoint_label

logger = logging.getLogger(__name__)

_NO_DOWNLOAD_ENV = {
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "WANDB_MODE": "offline",
}


def require_policy_adapter(policy_dir: Path) -> Path:
    """Return the upstream policy adapter or fail with an actionable error."""
    script = policy_dir.expanduser().resolve() / "setup_eval_policy_server.sh"
    if not script.is_file():
        raise ValueError(f"policy server adapter not found: {script}")
    return script


def policy_server_command(request: PolicyServerLaunchRequest, port: int) -> list[str]:
    """Build the official XPolicyLab setup adapter argument vector."""
    experiment = request.experiment
    script = require_policy_adapter(experiment.policy_dir)
    return [
        "bash",
        str(script),
        experiment.dataset,
        experiment.task,
        experiment.checkpoint,
        experiment.embodiment,
        experiment.action_type,
        str(request.seed),
        str(request.policy_gpu),
        experiment.policy_runtime,
        str(port),
        request.host,
    ]


def policy_hook_command(request: PolicyServerLaunchRequest, hook_name: str) -> list[str] | None:
    """Build an optional policy-owned hook using the standardized eight arguments."""
    allowed = {"prepare_eval_policy.sh", "check_eval_policy.sh"}
    if hook_name not in allowed:
        raise ValueError(f"unsupported policy hook: {hook_name}")
    experiment = request.experiment
    hook = experiment.policy_dir.expanduser().resolve() / hook_name
    if not hook.is_file():
        return None
    return [
        "bash",
        str(hook),
        experiment.dataset,
        experiment.task,
        experiment.checkpoint,
        experiment.embodiment,
        experiment.action_type,
        str(request.seed),
        str(request.policy_gpu),
        experiment.policy_runtime,
    ]


def policy_launch_environment(checkpoint: str) -> dict[str, str]:
    """Return launch guards that prevent implicit downloads in real and deep runs."""
    return {
        **_NO_DOWNLOAD_ENV,
        "ROBODOJO_CKPT_LABEL": checkpoint_label(checkpoint),
    }


def run_policy_server(request: PolicyServerLaunchRequest) -> int:
    """Launch one policy-owned adapter from its policy directory."""
    port = request.port or free_port()
    argv = policy_server_command(request, port)
    env = policy_launch_environment(request.experiment.checkpoint)
    logger.info("policy server: %s:%s", request.host, port)
    if request.dry_run:
        sys.stdout.write(f"{format_command(argv, env)}\n")
        return 0
    return run(argv, cwd=request.experiment.policy_dir.expanduser().resolve(), env=env)
