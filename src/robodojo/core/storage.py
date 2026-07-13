"""Canonical local storage paths and optional S3 publication settings."""

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


def _linked_worktree_primary(checkout: Path) -> Path | None:
    """Return the primary checkout for a Git linked worktree, if present."""
    git_file = checkout / ".git"
    if not git_file.is_file():
        return None

    try:
        marker = git_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    prefix = "gitdir:"
    if not marker.lower().startswith(prefix):
        return None

    gitdir_value = marker[len(prefix) :].strip()
    if not gitdir_value:
        return None
    gitdir = Path(gitdir_value).expanduser()
    if not gitdir.is_absolute():
        gitdir = git_file.parent / gitdir
    gitdir = gitdir.resolve()

    commondir_file = gitdir / "commondir"
    try:
        commondir_value = commondir_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not commondir_value:
        return None
    commondir = Path(commondir_value).expanduser()
    if not commondir.is_absolute():
        commondir = gitdir / commondir
    commondir = commondir.resolve()

    # In a regular primary checkout, Git's common metadata directory is
    # <checkout>/.git. Validate that shape before redirecting runtime data.
    if commondir.name != ".git" or not commondir.is_dir():
        return None
    primary = commondir.parent
    if not (primary / "pyproject.toml").is_file():
        return None
    return primary


def _default_storage_checkout(checkout: Path) -> Path:
    return _linked_worktree_primary(checkout) or checkout


def storage_root() -> Path:
    if explicit := _env_path("ROBODOJO_STORAGE_ROOT"):
        return explicit
    return _default_storage_checkout(_repo_root()) / ".robodojo"


def assets_root() -> Path:
    return storage_root() / "assets"


def data_root() -> Path:
    return storage_root() / "datasets"


def model_root() -> Path:
    return storage_root() / "model_weights"


def checkpoint_root() -> Path:
    return model_root()


def eval_root() -> Path:
    """Local result root whose children are benchmark task names."""
    return storage_root() / "runs" / "eval_result" / "RoboDojo"


def eval_work_root() -> Path:
    """Writable root for active evaluations and resume manifests."""
    return eval_root()


def run_root() -> Path:
    return storage_root() / "runs"


def run_work_root() -> Path:
    return run_root()


def summary_path(override: os.PathLike[str] | str | None = None) -> Path:
    """Writable Markdown summary path."""
    if override is not None:
        return Path(os.path.expanduser(os.fspath(override))).resolve()
    return run_root() / "reports" / "_summary.md"


def s3_uri() -> str | None:
    value = os.environ.get("ROBODOJO_S3_URI", "").strip().rstrip("/")
    return value or None


def checkpoint_label(value: str, explicit: str | None = None) -> str:
    label = explicit.strip() if explicit is not None else value.strip()
    if explicit is None and (Path(os.path.expanduser(label)).is_absolute() or "/" in label or "\\" in label):
        label = Path(label.rstrip("/\\")).name
    if not label or label in {".", ".."} or any(ch in label for ch in "/\\"):
        raise ValueError(f"invalid checkpoint label: {label!r}")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in label):
        raise ValueError("checkpoint labels may not contain control characters")
    return label
