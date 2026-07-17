"""Docker installation and smoke-test workflows."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess

from robodojo.core.paths import RepositoryPaths
from robodojo.core.storage import s3_uri, storage_root

logger = logging.getLogger(__name__)


def build(paths: RepositoryPaths, image: str, extra_args: tuple[str, ...] = ()) -> int:
    return subprocess.run(["docker", "build", *extra_args, "-t", image, "."], cwd=paths.root).returncode


def install(paths: RepositoryPaths) -> int:
    if shutil.which("docker"):
        logger.info("docker is already installed")
        return 0
    prefix = [] if os.geteuid() == 0 else ["sudo"]
    commands = [
        [*prefix, "apt-get", "update"],
        [*prefix, "apt-get", "install", "-y", "docker.io", "nvidia-container-toolkit"],
        [*prefix, "systemctl", "enable", "--now", "docker"],
    ]
    for command in commands:
        code = subprocess.run(command, cwd=paths.root).returncode
        if code:
            return code
    return 0


def smoke(
    paths: RepositoryPaths,
    image: str,
    task_protocol: str,
    policy: str,
    port: int,
    environment: str,
    scene: str | None = None,
) -> int:
    local_storage = storage_root()
    local_storage.mkdir(parents=True, exist_ok=True)
    container_storage = Path("/workspace/RoboDojo/.robodojo")
    storage_args: list[str] = [
        "-v",
        f"{local_storage}:{container_storage}",
        "-e",
        f"ROBODOJO_STORAGE_ROOT={container_storage}",
    ]
    if remote := s3_uri():
        storage_args += ["-e", f"ROBODOJO_S3_URI={remote}"]
    if profile := os.environ.get("AWS_PROFILE"):
        storage_args += ["-e", f"AWS_PROFILE={profile}"]
        credentials = Path.home() / ".aws"
        if credentials.is_dir():
            storage_args += ["-v", f"{credentials}:{credentials}:ro"]
    if env_file := os.environ.get("ROBODOJO_AWS_ENV_FILE"):
        candidate = Path(env_file).expanduser()
        if not candidate.is_file():
            raise ValueError(f"ROBODOJO_AWS_ENV_FILE is not readable: {candidate}")
        storage_args += ["--env-file", str(candidate)]
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--network",
        "host",
        *storage_args,
        "--ipc",
        "host",
        image,
        "eval",
        "client",
        "--policy-profile",
        policy,
        "--environment",
        environment,
        "--scene",
        scene or "default",
        "--task-protocol",
        task_protocol,
        "--policy-host",
        "127.0.0.1",
        "--policy-port",
        str(port),
        "--eval-num",
        "1",
    ]
    return subprocess.run(command, cwd=paths.root).returncode


def monitor(paths: RepositoryPaths) -> int:
    log_dir = paths.root / "docker" / "smoke_logs"
    logs = sorted(log_dir.glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not logs:
        logger.warning("no Docker smoke logs found")
        return 1
    return subprocess.run(["tail", "-f", str(logs[0])]).returncode


def clean(paths: RepositoryPaths) -> int:
    marker = paths.root / "docker" / "smoke_logs" / ".pid"
    if marker.is_file():
        try:
            os.kill(int(marker.read_text(encoding="utf-8")), signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            pass
        marker.unlink(missing_ok=True)
    listed = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "label=robodojo.smoke=true"], capture_output=True, text=True
    )
    if listed.returncode:
        return listed.returncode
    containers = listed.stdout.split()
    if containers:
        return subprocess.run(["docker", "rm", "-f", *containers]).returncode
    return 0
