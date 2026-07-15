"""Deterministic scene-layout discovery shared by validation and simulation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Literal


@dataclass(frozen=True)
class ResolvedLayout:
    layout_id: int
    path: Path


@dataclass(frozen=True)
class ResolvedLayoutSet:
    layouts: tuple[ResolvedLayout, ...]
    identity_hash: str
    directory: Path


def _safe_name(value: str, *, field: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    if not value or any(character not in allowed for character in value):
        raise ValueError(f"{field} must contain only letters, digits, and underscores")
    return value


def resolve_layout_set(
    *,
    config_root: Path,
    assets_root: Path,
    benchmark: str,
    layout_set: str,
    layout_source: Literal["assets", "bundled"],
    task: str,
    seed: int,
) -> ResolvedLayoutSet:
    """Resolve and hash the exact ordered layouts selected for one evaluation."""

    benchmark = _safe_name(benchmark, field="benchmark")
    layout_set = _safe_name(layout_set, field="layout set")
    task = _safe_name(task, field="task")
    if seed < 0:
        raise ValueError("layout seed must be non-negative")
    if layout_source == "assets":
        directory = Path(assets_root, "Eval_Layout", benchmark, layout_set, str(seed))
    elif layout_source == "bundled":
        directory = Path(config_root, "layout", layout_set, str(seed))
    else:
        raise ValueError(f"unsupported layout source: {layout_source!r}")

    pattern = re.compile(rf"{re.escape(task)}_(\d+)\.json")
    by_id: dict[int, Path] = {}
    duplicate_ids: dict[int, list[Path]] = {}
    if directory.is_dir():
        for path in directory.iterdir():
            match = pattern.fullmatch(path.name)
            if not path.is_file() or match is None:
                continue
            layout_id = int(match.group(1))
            if previous := by_id.get(layout_id):
                duplicate_ids.setdefault(layout_id, [previous]).append(path)
            else:
                by_id[layout_id] = path
    if duplicate_ids:
        details = ", ".join(
            f"{layout_id}: {[path.name for path in paths]}" for layout_id, paths in sorted(duplicate_ids.items())
        )
        raise ValueError(f"duplicate layout ids for {task} in {directory}: {details}")
    if not by_id:
        raise ValueError(
            f"no {layout_source} layouts found for task {task!r}, set {layout_set!r}, seed {seed} in {directory}"
        )

    layouts = tuple(ResolvedLayout(layout_id, by_id[layout_id]) for layout_id in sorted(by_id))
    digest = hashlib.sha256()
    digest.update(f"layout-set-v1\0{layout_source}\0{benchmark}\0{layout_set}\0{task}\0{seed}\0".encode())
    for layout in layouts:
        digest.update(f"{layout.layout_id}\0{layout.path.name}\0".encode())
        digest.update(layout.path.read_bytes())
        digest.update(b"\0")
    return ResolvedLayoutSet(layouts=layouts, identity_hash=digest.hexdigest(), directory=directory)
