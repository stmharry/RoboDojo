"""Locked uv installation workflow."""

from __future__ import annotations

from enum import StrEnum
import os
from pathlib import Path
import shutil
import subprocess

from robodojo.core.paths import RepositoryPaths


class InstallStep(StrEnum):
    SYSTEM = "system"
    SUBMODULES = "submodules"
    SYNC = "sync"


def _command(argv: list[str], cwd: Path) -> None:
    subprocess.run(argv, cwd=cwd, check=True)


def install_system(paths: RepositoryPaths) -> None:
    missing = []
    for package in ("cmake", "build-essential", "ffmpeg"):
        if subprocess.run(["dpkg", "-s", package], capture_output=True).returncode != 0:
            missing.append(package)
    if not missing:
        print("system dependencies already installed")
        return
    prefix = [] if os.geteuid() == 0 else ["sudo"]
    _command([*prefix, "apt-get", "update"], paths.root)
    _command([*prefix, "apt-get", "install", "-y", *missing], paths.root)


def install_submodules(paths: RepositoryPaths) -> None:
    _command(["git", "submodule", "sync", "--recursive"], paths.root)
    _command(["git", "submodule", "update", "--init", "--recursive", "--progress"], paths.root)


def install_sync(paths: RepositoryPaths) -> None:
    if shutil.which("uv") is None:
        raise RuntimeError("uv is required; install it before running robodojo install")
    env = os.environ.copy()
    env["OMNI_KIT_ACCEPT_EULA"] = env.get("OMNI_KIT_ACCEPT_EULA", "YES")
    subprocess.run(["uv", "python", "install", "3.11"], cwd=paths.root, env=env, check=True)
    subprocess.run(["uv", "sync", "--extra", "sim", "--locked"], cwd=paths.root, env=env, check=True)
    subprocess.run(["uv", "lock", "--check"], cwd=paths.root, env=env, check=True)


def install(paths: RepositoryPaths, start: InstallStep = InstallStep.SYSTEM) -> None:
    steps = [InstallStep.SYSTEM, InstallStep.SUBMODULES, InstallStep.SYNC]
    for step in steps[steps.index(start) :]:
        print(f">>> {step.value}")
        if step is InstallStep.SYSTEM:
            install_system(paths)
        elif step is InstallStep.SUBMODULES:
            install_submodules(paths)
        else:
            install_sync(paths)
