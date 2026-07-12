"""Validation and process launching for XPolicyLab policy adapters."""

from __future__ import annotations

import logging
from pathlib import Path

from robodojo.core.models import PolicyServerLaunchRequest
from robodojo.core.processes import format_command, free_port, run
from robodojo.core.storage import checkpoint_label

logger = logging.getLogger(__name__)

POLICY_ENVIRONMENT_NAMES = {
    ("LeRobot_Pi05_OpenArm", "openarm_wowrobo_v1_1"): "openarm_cloth_folding",
    ("LeRobot_Pi05_OpenArm", "openarm_anvil_v2"): "openarm_cloth_folding",
}


def require_policy_adapter(policy_dir: Path) -> Path:
    """Return the upstream policy adapter or fail with an actionable error."""
    script = policy_dir.expanduser().resolve() / "setup_eval_policy_server.sh"
    if not script.is_file():
        raise ValueError(f"policy server adapter not found: {script}")
    return script


def policy_server_command(request: PolicyServerLaunchRequest, port: int) -> list[str]:
    """Build the official XPolicyLab setup adapter argument vector."""
    script = require_policy_adapter(request.policy_dir)
    policy_environment = POLICY_ENVIRONMENT_NAMES.get(
        (request.policy_dir.expanduser().resolve().name, request.env_config), request.env_config
    )
    return [
        "bash",
        str(script),
        request.dataset,
        request.task,
        request.checkpoint,
        policy_environment,
        request.action_type,
        str(request.seed),
        str(request.policy_gpu),
        request.policy_env,
        str(port),
        request.host,
    ]


def run_policy_server(request: PolicyServerLaunchRequest) -> int:
    """Launch one policy-owned adapter from its policy directory."""
    port = request.port or free_port()
    argv = policy_server_command(request, port)
    env = {"ROBODOJO_CKPT_LABEL": checkpoint_label(request.checkpoint)}
    logger.info("policy server: %s:%s", request.host, port)
    if request.dry_run:
        print(format_command(argv, env))
        return 0
    return run(argv, cwd=request.policy_dir.expanduser().resolve(), env=env)
