"""Coordinate policy and simulator lifecycles for local evaluation."""

from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

from robodojo.core.gpu import GpuSelectionError, resolve_gpus
from robodojo.core.models import EvaluationRequest, PolicyServerLaunchRequest, SimulatorLaunchRequest
from robodojo.core.paths import RepositoryPaths
from robodojo.core.processes import format_command, free_port, start, terminate_process_group, wait_for_port
from robodojo.core.storage import checkpoint_label, s3_uri
from robodojo.policy.adapter import policy_launch_environment, policy_server_command
from robodojo.sim.launcher import run_simulator, simulator_command

SCENE_VISUAL_AUDIT_ENV = "ROBODOJO_SCENE_VISUAL_AUDIT"
logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _policy_name(policy_dir: Path) -> str:
    return policy_dir.resolve().name


def run_simulator_session(
    paths: RepositoryPaths,
    request: SimulatorLaunchRequest,
    environment: dict[str, str] | None = None,
) -> int:
    """Run one simulator client without performing publication orchestration."""
    launch_env = dict(environment or {})
    if not request.dry_run:
        launch_env.setdefault("ROBODOJO_RUN_ID", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    return run_simulator(paths, request, launch_env)


def _publish_evaluation(run_id: str) -> int:
    from robodojo.workflows.storage import publish_evaluation_run

    try:
        publish_evaluation_run(run_id)
    except SystemExit as exc:
        logger.error("evaluation completed, but S3 publication failed: %s", exc)
        return exc.code if isinstance(exc.code, int) and exc.code != 0 else 1
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        logger.error("evaluation completed, but S3 publication failed: %s", detail)
        return exc.returncode or 1
    except OSError as exc:
        logger.error("evaluation completed, but S3 publication failed: %s", exc)
        return 1
    return 0


def run_evaluation(paths: RepositoryPaths, request: EvaluationRequest, *, preflight: bool = True) -> int:
    """Run the policy adapter and simulator as one deterministic lifecycle."""
    visual_audit = _env_flag(SCENE_VISUAL_AUDIT_ENV)
    if visual_audit and not request.export_scene_only:
        raise ValueError(f"{SCENE_VISUAL_AUDIT_ENV}=1 is valid only with --export-scene-only")
    try:
        if request.export_scene_only:
            assignment = resolve_gpus(env_gpu=request.env_gpu)
            request = request.model_copy(update={"env_gpu": assignment.env_gpu})
        else:
            assignment = resolve_gpus(policy_gpu=request.policy_gpu, env_gpu=request.env_gpu)
            request = request.model_copy(update={"policy_gpu": assignment.policy_gpu, "env_gpu": assignment.env_gpu})
    except GpuSelectionError as exc:
        logger.error("GPU selection failed: %s", exc)
        return 2
    if request.publish and not request.dry_run:
        remote = s3_uri()
        if remote is None or not remote.startswith("s3://"):
            logger.error("--publish requires ROBODOJO_S3_URI to name a dedicated s3:// prefix")
            return 2
        if shutil.which("aws") is None:
            logger.error("--publish requires the AWS CLI to be installed and available on PATH")
            return 2
    if preflight and not request.dry_run:
        from robodojo.workflows.preflight import (
            emit_report,
            request_from_evaluation,
            run_fast_preflight,
            run_simulator_preflight,
        )

        preflight_request = request_from_evaluation(request)
        report = (
            run_simulator_preflight(paths, preflight_request)
            if request.export_scene_only
            else run_fast_preflight(paths, preflight_request)
        )
        emit_report(report)
        if report.status == "FAIL":
            return 2
    policy_dir = request.policy_dir.expanduser().resolve()
    policy_name = _policy_name(policy_dir)
    label = checkpoint_label(request.checkpoint, request.checkpoint_label)
    eval_num: int | str = request.eval_num or "native"
    port = 1 if request.export_scene_only else free_port()
    simulator_request = SimulatorLaunchRequest(
        task=request.task,
        protocol_name=request.protocol,
        episode_horizon=request.episode_horizon,
        native_eval_num=request.native_eval_num,
        recipe=request.recipe,
        contract_hash=request.contract_hash,
        policy_name=policy_name,
        policy_profile=request.policy_profile,
        policy_descriptor_hash=request.policy_descriptor_hash,
        policy_reference_match=request.policy_reference_match,
        host="127.0.0.1",
        port=port,
        env_config=request.env_config,
        scene_config=request.scene_config,
        env_gpu=request.env_gpu,
        seed=request.seed,
        eval_num=eval_num,
        additional_info=f"ckpt_name={label},action_type={request.action_type}",
        dry_run=request.dry_run,
    )
    simulator_argv, simulator_env = simulator_command(paths, simulator_request)
    simulator_env.update(
        {
            "ROBODOJO_CKPT_LABEL": label,
            "ROBODOJO_EXPORT_SCENE": str(request.export_scene or request.export_scene_only).lower(),
            "ROBODOJO_EXPORT_SCENE_ONLY": str(request.export_scene_only).lower(),
            "ROBODOJO_EXPORT_LAYOUT_ID": str(request.layout_id),
        }
    )
    if visual_audit:
        simulator_env[SCENE_VISUAL_AUDIT_ENV] = "1"
    if request.export_scene_dir:
        simulator_env["ROBODOJO_EXPORT_SCENE_DIR"] = str(request.export_scene_dir.resolve())
    if not request.dry_run:
        simulator_env.setdefault("ROBODOJO_RUN_ID", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    if request.export_scene_only:
        if request.dry_run:
            sys.stdout.write(f"{format_command(simulator_argv, simulator_env)}\n")
            return 0
        return run_simulator_session(paths, simulator_request, simulator_env)

    policy_request = PolicyServerLaunchRequest(
        policy_dir=policy_dir,
        task=request.task,
        checkpoint=request.checkpoint,
        policy_profile=request.policy_profile,
        policy_descriptor_hash=request.policy_descriptor_hash,
        policy_reference_match=request.policy_reference_match,
        policy_env=request.policy_env,
        dataset=request.dataset,
        env_config=request.env_config,
        policy_contract=request.policy_contract,
        action_type=request.action_type,
        seed=request.seed,
        policy_gpu=request.policy_gpu,
        host="127.0.0.1",
        port=port,
        dry_run=request.dry_run,
    )
    policy_argv = policy_server_command(policy_request, port)
    policy_env = policy_launch_environment(request.checkpoint)
    policy_env["ROBODOJO_CKPT_LABEL"] = label
    if request.dry_run:
        sys.stdout.write(f"{format_command(policy_argv, policy_env)}\n")
        sys.stdout.write(f"{format_command(simulator_argv, simulator_env)}\n")
        return 0

    policy_process = start(policy_argv, cwd=policy_dir, env=policy_env)
    try:
        wait_for_port(policy_process, "127.0.0.1", port, timeout=600)
        code = run_simulator_session(paths, simulator_request, simulator_env)
        if code == 0 and request.publish:
            return _publish_evaluation(simulator_env["ROBODOJO_RUN_ID"])
        return code
    finally:
        terminate_process_group(policy_process)
