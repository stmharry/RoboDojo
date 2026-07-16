"""Split policy-server and simulator-client orchestration."""

from __future__ import annotations

import logging
import socket

from robodojo.core.gpu import GpuSelectionError, resolve_gpus
from robodojo.core.models.requests import (
    PreflightRequest,
    ServerRequest,
    SimulatorLaunchRequest,
)
from robodojo.core.paths import RepositoryPaths

logger = logging.getLogger(__name__)


def run_server(paths: RepositoryPaths, request: ServerRequest) -> int:
    """Preflight and launch a standalone policy server."""

    from robodojo.policy.adapter import run_policy_server
    from robodojo.workflows.preflight import emit_report, run_fast_preflight

    try:
        assignment = resolve_gpus(policy_gpu=request.policy_gpu, env_gpu=request.environment_gpu)
    except GpuSelectionError as exc:
        logger.error("GPU selection failed: %s", exc)
        return 2
    request = request.model_copy(update={"policy_gpu": assignment.policy_gpu, "environment_gpu": assignment.env_gpu})
    policy_request = request.policy_request(host=request.host, port=request.port, dry_run=request.dry_run)
    if not request.dry_run:
        report = run_fast_preflight(
            paths,
            PreflightRequest(
                **request.model_dump(exclude={"host", "port", "dry_run"}),
            ),
        )
        emit_report(report)
        if report.status == "FAIL":
            return 1
    return run_policy_server(policy_request)


def warn_if_server_unreachable(host: str, port: int, timeout: float) -> str | None:
    """Return a non-fatal simulator-client reachability warning."""

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return None
    except OSError:
        return f"warning: {host}:{port} is not reachable yet; the client will retry"


def run_client(paths: RepositoryPaths, request: SimulatorLaunchRequest, *, connect_timeout: float) -> int:
    """Check external policy reachability and launch the simulator client."""

    from robodojo.orchestration.evaluation import run_simulator_session

    if not request.dry_run:
        if warning := warn_if_server_unreachable(request.host, request.port, connect_timeout):
            logger.warning(warning.removeprefix("warning: "))
    return run_simulator_session(paths, request)
