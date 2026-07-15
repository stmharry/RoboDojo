"""Pure-Python contracts shared by the scene-export CLI and runtime exporter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles import load_environment_profile, load_scene_profile

SCENE_EXPORT_FORMAT_VERSION = 6
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
    protocol: str
    layout: str
    episode_horizon: int
    native_eval_num: int
    recipe: str | None
    contract_hash: str | None
    profile: str
    scene_config: str
    seed: int
    layout_id: int
    repository_revision: str
    environment_profile_hash: str
    policy_contract: str
    scene_profile_hash: str
    layout_set_hash: str
    scene_asset_hash: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def scene_config_paths(
    repo_root: Path,
    profile: str,
    scene_config: str,
    task: str,
    components: Mapping[str, str],
) -> list[Path]:
    """Return the canonical configuration inputs for a scene export."""
    repository_paths = RepositoryPaths.resolve(repo_root)
    config_root = repository_paths.environment_configs
    scene_profile = load_scene_profile(repository_paths, scene_config)
    environment_profile = load_environment_profile(repository_paths, profile)
    return [
        *environment_profile.source_paths,
        config_root / "camera" / f"{components['camera']}.yml",
        scene_profile.path,
        scene_profile.component_path,
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


def exact_simulation_steps(duration_seconds: float, step_seconds: float) -> int:
    """Return the integral number of simulator steps for an exact duration."""
    if duration_seconds <= 0 or step_seconds <= 0:
        raise ValueError("duration and simulator step must be positive")
    raw_steps = duration_seconds / step_seconds
    steps = round(raw_steps)
    if not math.isclose(raw_steps, steps, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"duration {duration_seconds:g}s is not an integral number of {step_seconds:g}s simulator steps"
        )
    return steps


def camera_axes(camera_to_world: Any) -> dict[str, list[float]]:
    """Return USD camera axes in world coordinates from a camera-to-world matrix."""
    matrix = np.asarray(camera_to_world, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("camera-to-world transform must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]

    def unit(vector: np.ndarray) -> list[float]:
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("camera transform contains a zero-length axis")
        return (vector / norm).tolist()

    return {
        "right_world": unit(rotation[:, 0]),
        "up_world": unit(rotation[:, 1]),
        # UsdGeom.Camera looks down its local negative Z axis.
        "forward_world": unit(-rotation[:, 2]),
    }


def project_points_to_camera(
    points_world: Any,
    camera_to_world: Any,
    intrinsic_matrix: Any,
    resolution: tuple[int, int] | list[int],
) -> dict[str, Any]:
    """Project world points into a pinhole USD camera and summarize visibility."""
    points = np.asarray(points_world, dtype=np.float64)
    transform = np.asarray(camera_to_world, dtype=np.float64)
    intrinsic = np.asarray(intrinsic_matrix, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("world points must have shape (N, 3)")
    if transform.shape != (4, 4) or intrinsic.shape != (3, 3):
        raise ValueError("camera transform and intrinsic matrix must be 4x4 and 3x3")
    width, height = (int(resolution[0]), int(resolution[1]))
    if width <= 0 or height <= 0:
        raise ValueError("camera resolution must be positive")

    total = int(len(points))
    if total == 0:
        return {
            "particle_count": 0,
            "in_front_count": 0,
            "in_front_fraction": None,
            "visible_count": 0,
            "visible_fraction": None,
            "visible_pixel_bounds": None,
            "visible_normalized_bounds": None,
        }

    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    local = (points - translation) @ rotation
    depth = -local[:, 2]
    finite = np.isfinite(local).all(axis=1)
    in_front = finite & (depth > 1e-8)
    pixels = np.full((total, 2), np.nan, dtype=np.float64)
    pixels[in_front, 0] = intrinsic[0, 0] * local[in_front, 0] / depth[in_front] + intrinsic[0, 2]
    pixels[in_front, 1] = intrinsic[1, 2] - intrinsic[1, 1] * local[in_front, 1] / depth[in_front]
    visible = (
        in_front & (pixels[:, 0] >= 0.0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0.0) & (pixels[:, 1] < height)
    )
    visible_pixels = pixels[visible]
    bounds = None
    normalized_bounds = None
    if len(visible_pixels):
        minimum = np.min(visible_pixels, axis=0)
        maximum = np.max(visible_pixels, axis=0)
        bounds = {
            "min_xy": minimum.tolist(),
            "max_xy": maximum.tolist(),
        }
        scale = np.asarray([width, height], dtype=np.float64)
        normalized_bounds = {
            "min_xy": (minimum / scale).tolist(),
            "max_xy": (maximum / scale).tolist(),
        }
    in_front_count = int(np.count_nonzero(in_front))
    visible_count = int(np.count_nonzero(visible))
    return {
        "particle_count": total,
        "in_front_count": in_front_count,
        "in_front_fraction": in_front_count / total,
        "visible_count": visible_count,
        "visible_fraction": visible_count / total,
        "visible_pixel_bounds": bounds,
        "visible_normalized_bounds": normalized_bounds,
    }


def forward_ray_plane_intersection(camera_to_world: Any, plane_z_world: float | None) -> dict[str, Any]:
    """Intersect the USD camera forward ray with a horizontal world plane."""
    if plane_z_world is None or not math.isfinite(plane_z_world):
        return {
            "plane_z_world_m": None if plane_z_world is None else float(plane_z_world),
            "hit_in_front": None,
            "distance_m": None,
            "point_world_m": None,
        }
    matrix = np.asarray(camera_to_world, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("camera-to-world transform must be a finite 4x4 matrix")
    origin = matrix[:3, 3]
    forward = -matrix[:3, 2]
    if abs(float(forward[2])) <= 1e-12:
        return {
            "plane_z_world_m": float(plane_z_world),
            "hit_in_front": False,
            "distance_m": None,
            "point_world_m": None,
        }
    distance = float((plane_z_world - origin[2]) / forward[2])
    if distance <= 0:
        return {
            "plane_z_world_m": float(plane_z_world),
            "hit_in_front": False,
            "distance_m": distance,
            "point_world_m": None,
        }
    return {
        "plane_z_world_m": float(plane_z_world),
        "hit_in_front": True,
        "distance_m": distance,
        "point_world_m": (origin + distance * forward).tolist(),
    }


def geometric_cloth_support(
    points_world: Any,
    table_height_world: float | None,
    *,
    near_surface_tolerance_m: float = 0.03,
    below_surface_tolerance_m: float = 0.05,
) -> dict[str, Any]:
    """Summarize geometric table support without claiming contact-sensor data."""
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("cloth points must have shape (N, 3)")
    if len(points) == 0 or table_height_world is None or not np.isfinite(table_height_world):
        return {
            "table_height_world_m": None if table_height_world is None else float(table_height_world),
            "near_surface_tolerance_m": near_surface_tolerance_m,
            "below_surface_tolerance_m": below_surface_tolerance_m,
            "particle_fraction_near_surface": None,
            "particle_fraction_below_surface_tolerance": None,
            "geometrically_supported": None,
            "contact_count": None,
            "contact_measurement_available": False,
        }
    z = points[:, 2]
    near_fraction = float(np.mean(np.abs(z - table_height_world) <= near_surface_tolerance_m))
    below_fraction = float(np.mean(z < table_height_world - below_surface_tolerance_m))
    return {
        "table_height_world_m": float(table_height_world),
        "near_surface_tolerance_m": near_surface_tolerance_m,
        "below_surface_tolerance_m": below_surface_tolerance_m,
        "particle_fraction_near_surface": near_fraction,
        "particle_fraction_below_surface_tolerance": below_fraction,
        "geometrically_supported": bool(near_fraction > 0.0 and below_fraction == 0.0),
        "contact_count": None,
        "contact_measurement_available": False,
    }


def vector_drift(before: Any, after: Any) -> dict[str, float | None]:
    """Return maximum and RMS absolute drift for like-shaped numeric vectors."""
    first = np.asarray(before, dtype=np.float64)
    second = np.asarray(after, dtype=np.float64)
    if first.shape != second.shape or first.size == 0 or not np.isfinite(first).all() or not np.isfinite(second).all():
        return {"max_abs": None, "rms": None}
    delta = second - first
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rms": float(np.sqrt(np.mean(np.square(delta)))),
    }
