"""Process orchestration around external XPolicyLab policy adapters."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sys
import time
from typing import Any

import yaml

from robodojo.core.models import ClientRequest, EnvironmentConfigDocument, EvaluationRequest, ServerRequest
from robodojo.core.paths import RepositoryPaths
from robodojo.core.processes import format_command, free_port, run, start, terminate_process_group, wait_for_port
from robodojo.core.storage import checkpoint_label


def _policy_name(policy_dir: Path) -> str:
    return policy_dir.resolve().name


def _require_policy_adapter(policy_dir: Path) -> Path:
    script = policy_dir.expanduser().resolve() / "setup_eval_policy_server.sh"
    if not script.is_file():
        raise ValueError(f"policy server adapter not found: {script}")
    return script


def server_command(request: ServerRequest, port: int) -> list[str]:
    script = _require_policy_adapter(request.policy_dir)
    return [
        "bash",
        str(script),
        request.dataset,
        request.task,
        request.checkpoint,
        request.env_config,
        request.action_type,
        str(request.seed),
        str(request.policy_gpu),
        request.policy_env,
        str(port),
        request.host,
    ]


def _load_client_config(paths: RepositoryPaths, request: ClientRequest) -> tuple[int, str]:
    config_path = paths.environment_configs / f"{request.env_config}.yml"
    if not config_path.is_file():
        raise ValueError(f"environment config not found: {config_path}")
    payload: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    document = EnvironmentConfigDocument.model_validate({"config": payload.get("config", {})})
    for section, name in document.config.model_dump().items():
        suffix = ".json" if section == "robot" and name == "_robot_info" else ".yml"
        referenced = paths.environment_configs / section / f"{name}{suffix}"
        if not referenced.is_file():
            raise ValueError(f"referenced {section} config not found: {referenced}")
    sim_path = paths.environment_configs / "sim" / f"{document.config.sim}.yml"
    sim: dict[str, Any] = yaml.safe_load(sim_path.read_text(encoding="utf-8")) or {}
    num_envs = int(sim.get("scene", {}).get("num_envs", 1))

    deploy = paths.xpolicy_root / "policy" / request.policy_name / "deploy.yml"
    protocol = request.protocol
    if deploy.is_file():
        deploy_payload = yaml.safe_load(deploy.read_text(encoding="utf-8")) or {}
        protocol = str(deploy_payload.get("protocol", protocol))
    if protocol != "ws":
        raise ValueError(f"unsupported policy protocol: {protocol}")
    return num_envs, protocol


def client_command(paths: RepositoryPaths, request: ClientRequest) -> tuple[list[str], dict[str, str]]:
    num_envs, protocol = _load_client_config(paths, request)
    server_url = request.policy_server_url or f"ws://{request.host}:{request.port}"
    kit_args = " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
    argv = [
        sys.executable,
        "-u",
        "-m",
        "robodojo.client.evaluation.main",
        "--task_name",
        request.task,
        "--env_cfg_type",
        request.env_config,
        "--num_envs",
        str(num_envs),
        "--enable_cameras",
        "--kit_args",
        kit_args,
        "--device_id",
        str(request.env_gpu),
        "--policy_name",
        request.policy_name,
        "--port",
        str(request.port),
        "--protocol",
        protocol,
        "--policy_server_url",
        server_url,
        "--additional_info",
        request.additional_info,
        "--seed",
        str(request.seed),
        "--host",
        request.host,
        "--headless",
    ]
    env = {
        "CUDA_VISIBLE_DEVICES": str(request.env_gpu),
        "OMNI_KIT_ACCEPT_EULA": os.environ.get("OMNI_KIT_ACCEPT_EULA", "YES"),
        "ACCEPT_EULA": os.environ.get("ACCEPT_EULA", "Y"),
        "PRIVACY_CONSENT": os.environ.get("PRIVACY_CONSENT", "Y"),
        "PYTHONPATH": os.pathsep.join(filter(None, (str(paths.xpolicy_root), os.environ.get("PYTHONPATH", "")))),
    }
    if request.eval_num != "native":
        env["EVAL_NUM"] = str(request.eval_num)
    return argv, env


def run_client(
    paths: RepositoryPaths,
    request: ClientRequest,
    environment: dict[str, str] | None = None,
) -> int:
    argv, env = client_command(paths, request)
    if environment:
        env.update(environment)
    if request.dry_run:
        print(format_command(argv, env))
        return 0
    env.setdefault("ROBODOJO_RUN_ID", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    retries = int(os.environ.get("ROBODOJO_MAX_BASH_RETRIES", "10"))
    for attempt in range(retries):
        code = run(argv, cwd=paths.root, env=env)
        if code not in {99, 134, 139}:
            if code == 0 and os.environ.get("ROBODOJO_EXPORT_SCENE_ONLY", "").lower() != "true":
                if os.environ.get("ROBODOJO_STORAGE_ROOT") or os.environ.get("ROBODOJO_S3_URI"):
                    from robodojo.workflows.storage import main as storage_main

                    storage_main(["publish-eval", ".", "--run-id", env["ROBODOJO_RUN_ID"]])
            return code
        if attempt + 1 >= retries:
            return code
        print(f"client exited with {code}; restarting ({attempt + 1}/{retries})", file=sys.stderr)
        time.sleep(5)
    return 1


def run_server(request: ServerRequest) -> int:
    port = request.port or free_port()
    argv = server_command(request, port)
    env = {"ROBODOJO_CKPT_LABEL": checkpoint_label(request.checkpoint)}
    print(f"policy server: {request.host}:{port}")
    if request.dry_run:
        print(format_command(argv, env))
        return 0
    return run(argv, cwd=request.policy_dir.resolve(), env=env)


def run_evaluation(paths: RepositoryPaths, request: EvaluationRequest) -> int:
    policy_dir = request.policy_dir.expanduser().resolve()
    policy_name = _policy_name(policy_dir)
    label = checkpoint_label(request.checkpoint, request.checkpoint_label)
    eval_num = request.eval_num
    if eval_num is None:
        eval_num_value: int | str = int(os.environ.get("EVAL_NUM", "1"))
    elif eval_num == "native":
        eval_num_value = "native"
    else:
        eval_num_value = eval_num
    port = 1 if request.export_scene_only else free_port()
    client_request = ClientRequest(
        task=request.task,
        policy_name=policy_name,
        host="127.0.0.1",
        port=port,
        env_config=request.env_config,
        env_gpu=request.env_gpu,
        seed=request.seed,
        eval_num=eval_num_value,
        additional_info=f"ckpt_name={label},action_type={request.action_type}",
        dry_run=request.dry_run,
    )
    client_argv, client_env = client_command(paths, client_request)
    client_env.update(
        {
            "ROBODOJO_CKPT_LABEL": label,
            "ROBODOJO_EXPORT_SCENE": str(request.export_scene or request.export_scene_only).lower(),
            "ROBODOJO_EXPORT_SCENE_ONLY": str(request.export_scene_only).lower(),
            "ROBODOJO_EXPORT_LAYOUT_ID": str(request.layout_id),
        }
    )
    if request.export_scene_dir:
        client_env["ROBODOJO_EXPORT_SCENE_DIR"] = str(request.export_scene_dir.resolve())

    if request.export_scene_only:
        if request.dry_run:
            print(format_command(client_argv, client_env))
            return 0
        return run_client(paths, client_request, client_env)

    server_request = ServerRequest(
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
    server_argv = server_command(server_request, port)
    server_env = {"ROBODOJO_CKPT_LABEL": label}
    if request.dry_run:
        print(format_command(server_argv, server_env))
        print(format_command(client_argv, client_env))
        return 0

    server_process = start(server_argv, cwd=policy_dir, env=server_env)
    try:
        wait_for_port(server_process, "127.0.0.1", port, timeout=600)
        return run_client(paths, client_request, client_env)
    finally:
        terminate_process_group(server_process)
