"""Coordinate policy and simulator lifecycles for local evaluation."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

from robodojo.core.models import EvaluationRequest, PolicyServerLaunchRequest, SimulatorLaunchRequest
from robodojo.core.paths import RepositoryPaths
from robodojo.core.processes import format_command, free_port, start, terminate_process_group, wait_for_port
from robodojo.core.storage import checkpoint_label, s3_uri
from robodojo.policy.adapter import policy_server_command
from robodojo.sim.launcher import run_simulator, simulator_command

SCENE_VISUAL_AUDIT_ENV = "ROBODOJO_SCENE_VISUAL_AUDIT"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _policy_name(policy_dir: Path) -> str:
    return policy_dir.resolve().name


def run_simulator_session(
    paths: RepositoryPaths,
    request: SimulatorLaunchRequest,
    environment: dict[str, str] | None = None,
) -> int:
    """Run a simulator client and publish its completed evaluation when configured."""
    launch_env = dict(environment or {})
    if not request.dry_run:
        launch_env.setdefault("ROBODOJO_RUN_ID", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    code = run_simulator(paths, request, launch_env)
    if (
        code == 0
        and not request.dry_run
        and launch_env.get("ROBODOJO_EXPORT_SCENE_ONLY", "").lower() != "true"
        and s3_uri() is not None
    ):
        from robodojo.workflows.storage import main as storage_main

        storage_main(["publish-eval", ".", "--run-id", launch_env["ROBODOJO_RUN_ID"]])
    return code


def run_evaluation(paths: RepositoryPaths, request: EvaluationRequest) -> int:
    """Run the policy adapter and simulator as one deterministic lifecycle."""
    visual_audit = _env_flag(SCENE_VISUAL_AUDIT_ENV)
    if visual_audit and not request.export_scene_only:
        raise ValueError(f"{SCENE_VISUAL_AUDIT_ENV}=1 is valid only with --export-scene-only")
    policy_dir = request.policy_dir.expanduser().resolve()
    policy_name = _policy_name(policy_dir)
    label = checkpoint_label(request.checkpoint, request.checkpoint_label)
    if request.eval_num is None:
        eval_num: int | str = int(os.environ.get("EVAL_NUM", "1"))
    else:
        eval_num = request.eval_num
    port = 1 if request.export_scene_only else free_port()
    simulator_request = SimulatorLaunchRequest(
        task=request.task,
        policy_name=policy_name,
        host="127.0.0.1",
        port=port,
        env_config=request.env_config,
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

    if request.export_scene_only:
        if request.dry_run:
            print(format_command(simulator_argv, simulator_env))
            return 0
        return run_simulator_session(paths, simulator_request, simulator_env)

    policy_request = PolicyServerLaunchRequest(
        policy_dir=policy_dir,
        task=request.task,
        checkpoint=request.checkpoint,
        policy_env=request.policy_env,
        dataset=request.dataset,
        env_config=request.env_config,
        action_type=request.action_type,
        seed=request.seed,
        policy_gpu=request.policy_gpu,
        host="127.0.0.1",
        port=port,
        dry_run=request.dry_run,
    )
    policy_argv = policy_server_command(policy_request, port)
    policy_env = {"ROBODOJO_CKPT_LABEL": label}
    if request.dry_run:
        print(format_command(policy_argv, policy_env))
        print(format_command(simulator_argv, simulator_env))
        return 0

    policy_process = start(policy_argv, cwd=policy_dir, env=policy_env)
    try:
        wait_for_port(policy_process, "127.0.0.1", port, timeout=600)
        return run_simulator_session(paths, simulator_request, simulator_env)
    finally:
        terminate_process_group(policy_process)
