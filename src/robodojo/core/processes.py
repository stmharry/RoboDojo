"""Safe subprocess and readiness helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
import signal
import socket
import subprocess
import time


def format_command(argv: Sequence[str], env: Mapping[str, str] | None = None) -> str:
    import shlex

    prefix = ""
    if env:
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items())) + " "
    return prefix + shlex.join(argv)


def run(argv: Sequence[str], *, cwd: str | os.PathLike[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(list(argv), cwd=cwd, env=merged, check=False).returncode


def start(
    argv: Sequence[str],
    *,
    cwd: str | os.PathLike[str],
    env: Mapping[str, str] | None = None,
) -> subprocess.Popen:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.Popen(list(argv), cwd=cwd, env=merged, start_new_session=True)


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def wait_for_port(process: subprocess.Popen, host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"policy server exited with status {code} before opening {host}:{port}")
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"timed out after {timeout:g}s waiting for {host}:{port}")


def terminate_process_group(process: subprocess.Popen, grace: float = 10.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
