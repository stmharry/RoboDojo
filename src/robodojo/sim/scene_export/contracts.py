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


def calculate_fisheye_fov_degrees(
    width: int,
    height: int,
    fx: float,
    fy: float,
    coefficients: list[float] | tuple[float, ...],
) -> dict[str, float]:
    """Calculate OpenCV equidistant fisheye FOV from output intrinsics."""
    if width <= 0 or height <= 0 or fx <= 0 or fy <= 0:
        raise ValueError("resolution and focal lengths must be positive")
    if len(coefficients) != 4:
        raise ValueError("fisheye distortion requires four coefficients")

    def invert(theta_distorted: float) -> float:
        theta = theta_distorted
        for _ in range(10):
            theta2 = theta * theta
            powers = (theta2, theta2**2, theta2**3, theta2**4)
            scale = 1.0 + sum(k * power for k, power in zip(coefficients, powers, strict=True))
            derivative = 1.0 + sum(
                (2 * index + 1) * k * power
                for index, (k, power) in enumerate(zip(coefficients, powers, strict=True), start=1)
            )
            theta -= (theta * scale - theta_distorted) / derivative
        return theta

    half_x = width / (2.0 * fx)
    half_y = height / (2.0 * fy)
    return {
        "horizontal": math.degrees(2.0 * invert(half_x)),
        "vertical": math.degrees(2.0 * invert(half_y)),
        "diagonal": math.degrees(2.0 * invert(math.hypot(half_x, half_y))),
    }


def completed_export_matches(output_dir: str | Path, identity: ExportIdentity) -> bool:
    """Return whether an atomic export already completed for this exact scene."""
    manifest_path = Path(output_dir) / "scene_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return bool(manifest.get("complete")) and manifest.get("identity") == identity.to_dict()
