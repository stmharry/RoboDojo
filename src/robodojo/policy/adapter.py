"""Validation and process launching for XPolicyLab policy adapters."""

from __future__ import annotations

import logging
from pathlib import Path

from robodojo.core.models import PolicyServerLaunchRequest
from robodojo.core.paths import RepositoryPaths
from robodojo.core.processes import format_command, free_port, run
from robodojo.core.profiles import load_environment_profile
from robodojo.core.storage import checkpoint_label

logger = logging.getLogger(__name__)


def require_policy_adapter(policy_dir: Path) -> Path:
    """Return the upstream policy adapter or fail with an actionable error."""
    script = policy_dir.expanduser().resolve() / "setup_eval_policy_server.sh"
    if not script.is_file():
        raise ValueError(f"policy server adapter not found: {script}")
    return script


def policy_server_command(paths: RepositoryPaths, request: PolicyServerLaunchRequest, port: int) -> list[str]:
    """Build the official XPolicyLab setup adapter argument vector."""
    script = require_policy_adapter(request.policy_dir)
    # Policy-server argument construction stays lightweight; the simulator
    # launcher is the release gate for hardware calibration readiness.
    profile = load_environment_profile(paths, request.env_config, validate_calibration=False)
    return [
        "bash",
        str(script),
        request.dataset,
        request.task,
        request.checkpoint,
        profile.xpolicylab_env_cfg_type,
        request.action_type,
        str(request.seed),
        str(request.policy_gpu),
        request.policy_env,
        str(port),
        request.host,
    ]


def run_policy_server(paths: RepositoryPaths, request: PolicyServerLaunchRequest) -> int:
    """Launch one policy-owned adapter from its policy directory."""
    port = request.port or free_port()
    argv = policy_server_command(paths, request, port)
    env = {"ROBODOJO_CKPT_LABEL": checkpoint_label(request.checkpoint)}
    logger.info("policy server: %s:%s", request.host, port)
    if request.dry_run:
        print(format_command(argv, env))
        return 0
    return run(argv, cwd=request.policy_dir.expanduser().resolve(), env=env)
