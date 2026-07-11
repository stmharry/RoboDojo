"""Storage path contract for local work and durable RoboDojo artifacts.

Mountpoint for S3 is treated as a read-only consumption layer. Mutable work is
kept on a local POSIX filesystem and published separately with the AWS CLI.
"""

from __future__ import annotations

import os
from pathlib import Path

from robodojo.core.paths import discover_repository_root

try:
    REPO_ROOT: Path | None = discover_repository_root()
except RuntimeError:
    # Lightweight server installations may be imported outside a simulator checkout.
    REPO_ROOT = None


def _repo_root() -> Path:
    return REPO_ROOT or discover_repository_root()


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(os.path.expanduser(value)).resolve() if value else None


def storage_root() -> Path | None:
    return _env_path("ROBODOJO_STORAGE_ROOT")


def local_scratch_root() -> Path:
    return _env_path("ROBODOJO_LOCAL_SCRATCH_ROOT") or _repo_root() / ".cache" / "robodojo-runtime"


def assets_root() -> Path:
    root = storage_root()
    return _env_path("ROBODOJO_ASSETS_ROOT") or (root / "assets" if root else _repo_root() / "Assets")


def data_root() -> Path:
    root = storage_root()
    return (
        _env_path("ROBODOJO_DATA_ROOT")
        or _env_path("ROBO_DOJO_DATA_ROOT")
        or (root / "datasets" if root else _repo_root() / "data")
    )


def model_root() -> Path:
    root = storage_root()
    return _env_path("ROBODOJO_MODEL_ROOT") or (root / "model_weights" if root else _repo_root() / "model_weights")


def checkpoint_root() -> Path:
    root = storage_root()
    return _env_path("ROBODOJO_CHECKPOINT_ROOT") or (root / "model_weights" if root else _repo_root() / "checkpoints")


def eval_root() -> Path:
    """Durable/read root whose children are benchmark task names."""
    root = storage_root()
    return _env_path("ROBODOJO_EVAL_ROOT") or (
        root / "runs" / "eval_result" / "RoboDojo" if root else _repo_root() / "eval_result" / "RoboDojo"
    )


def eval_work_root() -> Path:
    """POSIX working root for active evaluations and resume manifests."""
    explicit = _env_path("ROBODOJO_EVAL_WORK_ROOT")
    if explicit:
        return explicit
    if storage_root():
        return local_scratch_root() / "eval_result" / "RoboDojo"
    # ROBODOJO_EVAL_ROOT historically selected the result tree. When no
    # durable storage root is configured it remains a directly writable root.
    return eval_root()


def run_root() -> Path:
    root = storage_root()
    return _env_path("ROBODOJO_RUN_ROOT") or (root / "runs" if root else _repo_root() / "smoke_results")


def run_work_root() -> Path:
    explicit = _env_path("ROBODOJO_RUN_WORK_ROOT")
    if explicit:
        return explicit
    if storage_root():
        return local_scratch_root() / "runs"
    return run_root()


def summary_path(override: os.PathLike[str] | str | None = None) -> Path:
    """Writable Markdown summary path, separate from durable inputs."""
    if override is not None:
        return Path(os.path.expanduser(os.fspath(override))).resolve()
    configured = _env_path("ROBODOJO_SUMMARY_PATH")
    if configured:
        return configured
    if storage_mode():
        return run_work_root() / "reports" / "_summary.md"
    return eval_root() / "_summary.md"


def s3_uri() -> str | None:
    value = os.environ.get("ROBODOJO_S3_URI", "").strip().rstrip("/")
    return value or None


def storage_mode() -> bool:
    return storage_root() is not None


def checkpoint_label(value: str, explicit: str | None = None) -> str:
    label = explicit.strip() if explicit is not None else value.strip()
    if explicit is None and (Path(os.path.expanduser(label)).is_absolute() or "/" in label or "\\" in label):
        label = Path(label.rstrip("/\\")).name
    if not label or label in {".", ".."} or any(ch in label for ch in "/\\"):
        raise ValueError(f"invalid checkpoint label: {label!r}")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in label):
        raise ValueError("checkpoint labels may not contain control characters")
    return label
