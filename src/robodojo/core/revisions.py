"""Repository revision identity for reproducible runtime artifacts."""

from __future__ import annotations

from pathlib import Path
import subprocess


def git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    revision = result.stdout.strip()
    if result.returncode != 0 or len(revision) != 40:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown git error"
        raise ValueError(f"could not resolve repository revision for {path}: {detail}")
    return revision
