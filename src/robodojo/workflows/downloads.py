"""Sparse Git LFS downloads for RoboDojo assets and datasets."""

from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path
import shutil
import subprocess

from robodojo.core.models import DataFormat
from robodojo.core.paths import RepositoryPaths
from robodojo.core.storage import assets_root, data_root, storage_root

HF_REPO_ID = "RoboDojo-Benchmark/RoboDojo"
DATASETS = {
    DataFormat.LEROBOT_V30: ("RoboDojo_lerobot_v30_video", "120GB", "LeRobot v3.0 joint-only"),
    DataFormat.LEROBOT_V21: ("RoboDojo_lerobot_v21_video", "64GB", "LeRobot v2.1 joint-only"),
    DataFormat.HDF5: ("RoboDojo", "523GB", "full HDF5"),
    DataFormat.DEMO: ("demo", "1.5GB", "demo dataset"),
    DataFormat.REAL: ("RoboDojo_real", "273GB", "real-world dataset"),
}

logger = logging.getLogger(__name__)


def assets_ready() -> bool:
    """Return whether the canonical base asset bundle is structurally complete."""

    target = assets_root()
    required = ("Robots", "Object", "Material", "Eval_Layout")
    return target.is_dir() and all((target / item).is_dir() for item in required)


def list_data() -> None:
    for kind, (directory, size, description) in DATASETS.items():
        print(f"{kind.value:14} {size:7} {directory}: {description}")


def _require_tools() -> None:
    if shutil.which("git") is None:
        raise RuntimeError("git is required")
    if subprocess.run(["git", "lfs", "version"], capture_output=True).returncode != 0:
        raise RuntimeError("git-lfs is required")


def _archive(path: Path) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path.rename(path.with_name(f"{path.name}.partial.{stamp}"))


def _sparse_payload(paths: RepositoryPaths, remote_dir: str, cache: Path, revision: str) -> Path:
    _require_tools()
    repo_url = os.environ.get("HF_REPO_URL", f"https://huggingface.co/datasets/{HF_REPO_ID}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not (cache / ".git").is_dir():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--sparse", repo_url, str(cache)],
            env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
            check=True,
        )
    else:
        fetch = subprocess.run(["git", "-C", str(cache), "fetch", "--depth", "1", "origin", revision])
        if fetch.returncode != 0:
            _archive(cache)
            return _sparse_payload(paths, remote_dir, cache, revision)
    subprocess.run(["git", "-C", str(cache), "sparse-checkout", "set", remote_dir], check=True)
    subprocess.run(["git", "-C", str(cache), "checkout", revision], check=True)
    subprocess.run(["git", "-C", str(cache), "lfs", "install", "--local"], check=True)
    subprocess.run(["git", "-C", str(cache), "lfs", "pull", f"--include={remote_dir}/**", "--exclude="], check=True)
    payload = cache / remote_dir
    if not payload.is_dir():
        raise RuntimeError(f"remote payload not found: {remote_dir}")
    return payload


def _install_payload(payload: Path, target: Path, cache: Path) -> None:
    staging_root = storage_root() / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staged = staging_root / f"{target.name}.installing"
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(payload, staged, symlinks=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, target)
    shutil.rmtree(cache, ignore_errors=True)


def download_assets(paths: RepositoryPaths, revision: str = "main") -> None:
    target = assets_root()
    if assets_ready():
        logger.info("assets already ready: %s", target)
        return
    if target.exists() or target.is_symlink():
        _archive(target)
    cache = storage_root() / ".cache" / "git" / "robodojo_assets_repo"
    payload = _sparse_payload(paths, "Assets", cache, revision)
    _install_payload(payload, target, cache)
    if not assets_ready():
        raise RuntimeError(f"asset download is incomplete: {target}")


def download_data(paths: RepositoryPaths, data_format: DataFormat, revision: str = "main") -> None:
    directory, size, _ = DATASETS[data_format]
    target = data_root() / directory
    if (target / ".download_complete").is_file():
        logger.info("dataset already ready: %s", target)
        return
    if target.exists() or target.is_symlink():
        _archive(target)
    cache = storage_root() / ".cache" / "git" / f"robodojo_data_{data_format.value}_repo"
    payload = _sparse_payload(paths, f"data/{directory}", cache, revision)
    marker = payload / ".download_complete"
    marker.write_text(
        f"repo_id={HF_REPO_ID}\nrevision={revision}\ndata_type={data_format.value}\nsize={size}\n",
        encoding="utf-8",
    )
    _install_payload(payload, target, cache)
