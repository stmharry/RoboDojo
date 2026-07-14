"""Pure-Python contracts shared by the scene-export CLI and runtime exporter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import zipfile

from robodojo.core.paths import RepositoryPaths

SCENE_EXPORT_FORMAT_VERSION = 2
REQUIRED_EXPORT_ARTIFACTS = (
    "scene_referenced.usda",
    "scene_flattened.usdc",
    "scene_preview.usdz",
)
REQUIRED_MANIFEST_ARTIFACTS = {
    "referenced_usda": "scene_referenced.usda",
    "flattened_usdc": "scene_flattened.usdc",
    "preview_usdz": "scene_preview.usdz",
}
REQUIRED_PREVIEW_DIAGNOSTICS = (
    "preserved_materials",
    "translated_materials",
    "fallback_materials",
    "missing_textures",
    "unsupported_inputs",
    "excluded_guide_meshes",
    "approximation",
)


@dataclass(frozen=True)
class ExportIdentity:
    task: str
    profile: str
    seed: int
    layout_id: int
    repository_revision: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def scene_config_paths(
    repo_root: Path,
    profile: str,
    task: str,
    components: Mapping[str, str],
) -> list[Path]:
    """Return the canonical configuration inputs for a scene export."""
    repository_paths = RepositoryPaths.resolve(repo_root)
    config_root = repository_paths.environment_configs
    return [
        repository_paths.environment_profiles / f"{profile}.yml",
        config_root / "camera" / f"{components['camera']}.yml",
        config_root / "scene" / f"{components['scene']}.yml",
        config_root / "robot" / f"{components['robot']}.yml",
        config_root / "sim" / f"{components['sim']}.yml",
        repository_paths.task_configs / f"{task}.yml",
    ]


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
    output = Path(output_dir)
    manifest_path = output / "scene_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    artifacts = manifest.get("artifacts")
    preview = manifest.get("preview")
    complete_artifacts = isinstance(artifacts, dict) and all(
        isinstance(artifacts.get(key), dict)
        and artifacts[key].get("path") == filename
        and isinstance(artifacts[key].get("sha256"), str)
        and len(artifacts[key]["sha256"]) == 64
        and (output / filename).is_file()
        for key, filename in REQUIRED_MANIFEST_ARTIFACTS.items()
    )
    complete_preview = isinstance(preview, dict) and all(key in preview for key in REQUIRED_PREVIEW_DIAGNOSTICS)
    return (
        bool(manifest.get("complete"))
        and manifest.get("format_version") == SCENE_EXPORT_FORMAT_VERSION
        and manifest.get("identity") == identity.to_dict()
        and complete_artifacts
        and complete_preview
    )


def split_package_asset_path(path: str) -> tuple[str, str]:
    """Split ``asset.usdz[member]`` into its package and member paths."""
    marker = path.find("[")
    if marker > 0 and path.endswith("]"):
        return path[:marker], path[marker + 1 : -1]
    return path, ""


def package_member_exists(package_path: str | Path, member_path: str) -> bool:
    """Return whether a USDZ/ZIP package contains the exact normalized member."""
    if not member_path:
        return True
    try:
        with zipfile.ZipFile(package_path) as package:
            return member_path.replace("\\", "/") in package.namelist()
    except (OSError, zipfile.BadZipFile):
        return False
