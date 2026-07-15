"""GPU inventory inspection and deterministic role assignment."""

from __future__ import annotations

from dataclasses import dataclass
import re
import shutil
import subprocess
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

GpuSelector: TypeAlias = Annotated[int, Field(ge=0)] | Literal["auto"]
GpuSelectionSource: TypeAlias = Literal["auto", "explicit"]


class GpuSelectionError(RuntimeError):
    """Raised when requested GPU roles cannot be resolved safely."""


@dataclass(frozen=True)
class GpuDevice:
    index: int
    free_memory_mib: int


@dataclass(frozen=True)
class GpuAssignment:
    policy_gpu: int | None = None
    env_gpu: int | None = None
    policy_source: GpuSelectionSource | None = None
    env_source: GpuSelectionSource | None = None


def _command_detail(result: subprocess.CompletedProcess[str]) -> str:
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part and part.strip())
    return output[-4000:] if output else f"command exited {result.returncode}"


def discover_gpus() -> tuple[GpuDevice, ...]:
    """Return physical GPUs ordered by free memory, then stable device index."""
    tool = shutil.which("nvidia-smi")
    if tool is None:
        raise GpuSelectionError("nvidia-smi is unavailable; install or repair the NVIDIA driver")
    result = subprocess.run(
        [tool, "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise GpuSelectionError(f"nvidia-smi could not query GPU memory: {_command_detail(result)}")

    devices: dict[int, GpuDevice] = {}
    for raw_line in result.stdout.splitlines():
        if not raw_line.strip():
            continue
        match = re.fullmatch(r"\s*(\d+)\s*,\s*(\d+)\s*", raw_line)
        if match is None:
            raise GpuSelectionError(f"malformed nvidia-smi output: {raw_line}")
        index, free_memory = (int(value) for value in match.groups())
        if index in devices:
            raise GpuSelectionError(f"nvidia-smi reported GPU index {index} more than once")
        devices[index] = GpuDevice(index=index, free_memory_mib=free_memory)
    if not devices:
        raise GpuSelectionError("nvidia-smi reported no GPUs")
    return tuple(sorted(devices.values(), key=lambda device: (-device.free_memory_mib, device.index)))


def _validate_selector(name: str, value: GpuSelector | None) -> None:
    if value is None or value == "auto":
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GpuSelectionError(f"{name} must be 'auto' or a nonnegative integer, got {value!r}")


def parse_gpu_selector(value: str | int) -> GpuSelector:
    """Parse one public CLI/environment selector without accepting aliases for auto."""
    if value == "auto":
        return "auto"
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    raise GpuSelectionError(f"expected 'auto' or a nonnegative integer, got {value!r}")


def _validate_explicit_devices(
    devices: tuple[GpuDevice, ...],
    *,
    policy_gpu: GpuSelector | None,
    env_gpu: GpuSelector | None,
) -> None:
    available = {device.index for device in devices}
    invalid = [value for value in (policy_gpu, env_gpu) if isinstance(value, int) and value not in available]
    if invalid:
        raise GpuSelectionError(f"GPU index/indices {invalid} are unavailable; available: {sorted(available)}")


def resolve_gpus(
    *,
    policy_gpu: GpuSelector | None = None,
    env_gpu: GpuSelector | None = None,
) -> GpuAssignment:
    """Resolve active GPU roles, querying inventory only when a selector is auto."""
    _validate_selector("POLICY_GPU", policy_gpu)
    _validate_selector("ENV_GPU", env_gpu)
    if policy_gpu is None and env_gpu is None:
        raise GpuSelectionError("at least one GPU role must be selected")
    if isinstance(policy_gpu, int) and isinstance(env_gpu, int) and policy_gpu == env_gpu:
        raise GpuSelectionError(f"POLICY_GPU and ENV_GPU must be distinct; both resolve to {policy_gpu}")

    auto_requested = policy_gpu == "auto" or env_gpu == "auto"
    devices = discover_gpus() if auto_requested else ()
    if devices:
        _validate_explicit_devices(devices, policy_gpu=policy_gpu, env_gpu=env_gpu)

    resolved_policy = policy_gpu if isinstance(policy_gpu, int) else None
    resolved_env = env_gpu if isinstance(env_gpu, int) else None
    if env_gpu == "auto":
        resolved_env = next((device.index for device in devices if device.index != resolved_policy), None)
        if resolved_env is None:
            raise GpuSelectionError("no distinct GPU is available for the simulator")
    if policy_gpu == "auto":
        resolved_policy = next((device.index for device in devices if device.index != resolved_env), None)
        if resolved_policy is None:
            raise GpuSelectionError("no distinct GPU is available for the policy")

    return GpuAssignment(
        policy_gpu=resolved_policy,
        env_gpu=resolved_env,
        policy_source=None if policy_gpu is None else ("auto" if policy_gpu == "auto" else "explicit"),
        env_source=None if env_gpu is None else ("auto" if env_gpu == "auto" else "explicit"),
    )


def validate_gpu_assignment(*, policy_gpu: int | None = None, env_gpu: int | None = None) -> None:
    """Validate concrete active roles against the current physical inventory."""
    if policy_gpu is None and env_gpu is None:
        raise GpuSelectionError("at least one GPU role must be selected")
    if policy_gpu is not None and env_gpu is not None and policy_gpu == env_gpu:
        raise GpuSelectionError(f"POLICY_GPU and ENV_GPU must be distinct; both resolve to {policy_gpu}")
    devices = discover_gpus()
    _validate_explicit_devices(devices, policy_gpu=policy_gpu, env_gpu=env_gpu)
