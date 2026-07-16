"""Lightweight command construction and launching for the simulator runtime."""

from __future__ import annotations

from datetime import datetime
import logging
import os
import sys
import time

import yaml

from robodojo.core.models.requests import SimulatorLaunchRequest
from robodojo.core.paths import RepositoryPaths
from robodojo.core.processes import format_command, run
from robodojo.core.profiles.environment import EnvironmentProfile, load_environment_profile
from robodojo.core.profiles.scene import SceneProfile, load_scene_profile, validate_scene_environment_compatibility

logger = logging.getLogger(__name__)


def resolve_scene_profile(paths: RepositoryPaths, request: SimulatorLaunchRequest) -> SceneProfile:
    """Resolve the explicitly selected scene profile."""

    return load_scene_profile(paths, request.experiment.scene)


def resolve_scene_name(
    paths: RepositoryPaths,
    request: SimulatorLaunchRequest,
    *,
    profile: EnvironmentProfile | None = None,
) -> str:
    """Return the name of the explicit scene profile."""

    environment = profile or load_environment_profile(paths, request.experiment.environment)
    scene = resolve_scene_profile(paths, request)
    validate_scene_environment_compatibility(scene, environment)
    return scene.name


def _resolved_simulator_config(
    paths: RepositoryPaths,
    request: SimulatorLaunchRequest,
) -> tuple[int, str, str]:
    """Validate the selected config graph and resolve launch-time values."""
    profile = load_environment_profile(paths, request.experiment.environment)
    num_envs = profile.num_envs
    scene_name = resolve_scene_name(paths, request, profile=profile)

    deploy = paths.xpolicy_root / "policy" / request.policy_name / "deploy.yml"
    transport = request.transport
    if deploy.is_file():
        deploy_payload = yaml.safe_load(deploy.read_text(encoding="utf-8")) or {}
        transport = str(deploy_payload.get("protocol", transport))
    if transport != "ws":
        raise ValueError(f"unsupported policy transport: {transport}")
    return num_envs, transport, scene_name


def load_simulator_config(paths: RepositoryPaths, request: SimulatorLaunchRequest) -> tuple[int, str]:
    """Validate the selected config graph and resolve launch-time values."""
    num_envs, transport, _ = _resolved_simulator_config(paths, request)
    return num_envs, transport


def simulator_command(paths: RepositoryPaths, request: SimulatorLaunchRequest) -> tuple[list[str], dict[str, str]]:
    """Build the simulator command while preserving upstream option names."""
    num_envs, transport, scene_name = _resolved_simulator_config(paths, request)
    experiment = request.experiment
    server_url = request.policy_server_url or f"ws://{request.host}:{request.port}"
    # CUDA_VISIBLE_DEVICES preserves the upstream physical device_id while
    # exposing that selected GPU to Isaac Lab as the process-local cuda:0.
    logical_device = "cuda:0"
    # IsaacLab's headless rendering experience disables USD and Fabric
    # transform updates. RoboDojo's CPU/no-Fabric cloth tasks therefore need
    # the full Python experience so camera frames track simulated robots.
    experience = "isaaclab.python.kit"
    kit_args = (
        " --/app/extensions/registryEnabled=0 --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
    )
    argv = [
        sys.executable,
        "-u",
        "-m",
        "robodojo.sim.evaluation.main",
        "--task",
        experiment.task,
        "--task-protocol",
        experiment.task_protocol,
        "--episode-horizon",
        str(experiment.episode_horizon),
        "--evaluation-episodes",
        str(experiment.evaluation_episodes),
        "--environment",
        experiment.environment,
        "--scene",
        scene_name,
        "--num_envs",
        str(num_envs),
        "--enable_cameras",
        "--experience",
        experience,
        "--kit_args",
        kit_args,
        "--device",
        logical_device,
        "--device_id",
        str(request.environment_gpu),
        "--policy_name",
        request.policy_name,
        "--policy_profile",
        experiment.policy_profile,
        "--policy_descriptor_hash",
        experiment.policy_descriptor_hash or "manual",
        "--policy_reference_match",
        experiment.policy_reference_match,
        "--port",
        str(request.port),
        "--transport",
        transport,
        "--policy_server_url",
        server_url,
        "--additional_info",
        request.additional_info,
        "--recipe",
        experiment.recipe or "manual",
        "--experiment-hash",
        experiment.experiment_hash or "manual",
        "--seed",
        str(request.seed),
        "--host",
        request.host,
        "--headless",
    ]
    env = {
        "CUDA_VISIBLE_DEVICES": str(request.environment_gpu),
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
        sys.stdout.write(f"{format_command(argv, env)}\n")
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
