"""Launch one policy-free simulator process for a first-frame snapshot."""

from __future__ import annotations

import logging

from robodojo.core.gpu import GpuSelectionError, resolve_gpus
from robodojo.core.models.requests import (
    SimulatorLaunchRequest,
    SnapshotCaptureRequest,
)
from robodojo.core.paths import RepositoryPaths
from robodojo.orchestration.evaluation import run_simulator_session

logger = logging.getLogger(__name__)


def run_snapshot_capture(paths: RepositoryPaths, request: SnapshotCaptureRequest) -> int:
    """Capture one recipe without resolving or starting its policy runtime."""
    try:
        assignment = resolve_gpus(env_gpu=request.environment_gpu)
    except GpuSelectionError as exc:
        logger.error("GPU selection failed: %s", exc)
        return 2
    request = request.model_copy(update={"environment_gpu": assignment.env_gpu})

    if not request.dry_run:
        from robodojo.workflows.preflight import emit_report, request_from_evaluation, run_simulator_preflight

        report = run_simulator_preflight(paths, request_from_evaluation(request))
        emit_report(report)
        if report.status == "FAIL":
            return 2

    simulator_request = SimulatorLaunchRequest(
        experiment=request.experiment,
        policy_name=request.experiment.policy_dir.expanduser().resolve().name,
        host="127.0.0.1",
        port=1,
        environment_gpu=request.environment_gpu,
        seed=request.seed,
        eval_num=1,
        additional_info=f"ckpt_name=snapshot,action_type={request.experiment.action_type}",
        dry_run=request.dry_run,
    )
    simulator_env = {
        "ROBODOJO_CAPTURE_FIRST_FRAME": "true",
        "ROBODOJO_FIRST_FRAME_DIR": str((request.output_dir / "first_frame").resolve()),
        "ROBODOJO_EXPORT_LAYOUT_ID": str(request.layout_id),
        "ROBODOJO_EXPORT_SCENE": str(request.export_scene).lower(),
        "ROBODOJO_EXPORT_SCENE_ONLY": "false",
        "ROBODOJO_RUN_ID": request.run_id,
        "ROBODOJO_SCENE_VISUAL_AUDIT": "false",
    }
    return run_simulator_session(paths, simulator_request, simulator_env)
