"""Pure-Python contracts shared by the scene-export CLI and runtime exporter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ExportIdentity:
    task: str
    profile: str
    seed: int
    layout_id: int
    repository_revision: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def calculate_fov_degrees(width: int, height: int, fx: float, fy: float) -> dict[str, float]:
    """Calculate pinhole horizontal, vertical, and corner-to-corner FOV."""
    if width <= 0 or height <= 0 or fx <= 0 or fy <= 0:
        raise ValueError("resolution and focal lengths must be positive")
    horizontal = 2.0 * math.atan(width / (2.0 * fx))
    vertical = 2.0 * math.atan(height / (2.0 * fy))
    diagonal = 2.0 * math.atan(math.hypot(width / fx, height / fy) / 2.0)
    return {
        "horizontal": math.degrees(horizontal),
        "vertical": math.degrees(vertical),
        "diagonal": math.degrees(diagonal),
    }


def completed_export_matches(output_dir: str | Path, identity: ExportIdentity) -> bool:
    """Return whether an atomic export already completed for this exact scene."""
    manifest_path = Path(output_dir) / "scene_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return bool(manifest.get("complete")) and manifest.get("identity") == identity.to_dict()
