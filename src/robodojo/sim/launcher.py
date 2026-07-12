"""Lightweight command construction and launching for the simulator runtime."""

from __future__ import annotations

from datetime import datetime
import logging
import os
import sys
import time
from typing import Any

import yaml

from robodojo.core.models import EnvironmentConfigDocument, SimulatorLaunchRequest
from robodojo.core.paths import RepositoryPaths
from robodojo.core.processes import format_command, run

logger = logging.getLogger(__name__)


def load_simulator_config(paths: RepositoryPaths, request: SimulatorLaunchRequest) -> tuple[int, str]:
    """Validate the selected config graph and resolve launch-time values."""
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


def simulator_command(paths: RepositoryPaths, request: SimulatorLaunchRequest) -> tuple[list[str], dict[str, str]]:
    """Build the simulator command while preserving upstream option names."""
    num_envs, protocol = load_simulator_config(paths, request)
    server_url = request.policy_server_url or f"ws://{request.host}:{request.port}"
    kit_args = " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
    argv = [
        sys.executable,
        "-u",
        "-m",
        "robodojo.sim.evaluation.main",
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


def run_simulator(
    paths: RepositoryPaths,
    request: SimulatorLaunchRequest,
    environment: dict[str, str] | None = None,
) -> int:
    """Launch the simulator, applying the upstream-compatible retry policy."""
    argv, env = simulator_command(paths, request)
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
            return code
        if attempt + 1 >= retries:
            return code
        logger.warning("simulator exited with %s; restarting (%s/%s)", code, attempt + 1, retries)
        time.sleep(5)
    return 1
