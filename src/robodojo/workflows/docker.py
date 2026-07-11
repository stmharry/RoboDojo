"""Docker installation and smoke-test workflows."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess

from robodojo.core.paths import RepositoryPaths


def build(paths: RepositoryPaths, image: str, extra_args: tuple[str, ...] = ()) -> int:
    return subprocess.run(["docker", "build", *extra_args, "-t", image, "."], cwd=paths.root).returncode


def install(paths: RepositoryPaths) -> int:
    if shutil.which("docker"):
        print("docker is already installed")
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


def smoke(paths: RepositoryPaths, image: str, task: str, policy: str, port: int, env_config: str) -> int:
    storage_args: list[str] = []
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
        "-v",
        f"{paths.assets}:{paths.root / 'Assets'}:ro",
        "-v",
        f"{paths.root / 'eval_result'}:{paths.root / 'eval_result'}",
        image,
        "client",
        "--task",
        task,
        "--policy-name",
        policy,
        "--policy-host",
        "127.0.0.1",
        "--policy-port",
        str(port),
        "--env-cfg",
        env_config,
        "--eval-num",
        "1",
    ]
    return subprocess.run(command, cwd=paths.root).returncode


def monitor(paths: RepositoryPaths) -> int:
    log_dir = paths.root / "docker" / "smoke_logs"
    logs = sorted(log_dir.glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not logs:
        print("no Docker smoke logs found")
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
