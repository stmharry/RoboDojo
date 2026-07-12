"""Dataset-driven calibration for rigid wrist cameras.

The transform convention is camera pose in the parent link frame.  Points are
converted to the OpenCV optical frame before applying OpenCV's native fisheye
projection model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import yaml


@dataclass(frozen=True)
class CameraFit:
    position: np.ndarray
    orientation: np.ndarray
    intrinsics: np.ndarray
    distortion: np.ndarray
    training_errors: np.ndarray
    held_out_errors: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frame(path: Path, expected_sha256: str) -> None:
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"checksum mismatch for {path}: expected {expected_sha256}, got {actual}")


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = yaml.safe_load(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported wrist calibration manifest schema")
    cameras = manifest.get("cameras", {})
    if set(cameras) != {"left", "right"}:
        raise ValueError("manifest must contain independent left and right cameras")
    for side, camera in cameras.items():
        if not camera.get("observations"):
            raise ValueError(f"{side} camera has no landmark observations")
    return manifest


def _camera_vector(camera: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        camera["initial_guess"]["position"]
        + camera["initial_guess"]["orientation"]
        + [camera["initial_guess"][key] for key in ("fx", "fy", "cx", "cy")]
        + camera["initial_guess"]["distortion"],
        dtype=np.float64,
    )


def fisheye_project(
    points_link: np.ndarray,
    position: np.ndarray,
    orientation_xyz_deg: np.ndarray,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """Project link-frame points using a camera-to-link rigid transform."""
    rotation_link_camera = Rotation.from_euler("XYZ", orientation_xyz_deg, degrees=True)
    points_usd_camera = rotation_link_camera.inv().apply(
        np.asarray(points_link, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    )
    # USD cameras look down local -Z with +Y up; OpenCV uses +Z forward and
    # +Y down. This fixed basis change is part of the simulator contract.
    points_camera = points_usd_camera * np.array([1.0, -1.0, -1.0])
    matrix = np.array(
        [[intrinsics[0], 0.0, intrinsics[2]], [0.0, intrinsics[1], intrinsics[3]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    projected, _ = cv2.fisheye.projectPoints(
        points_camera.reshape(1, -1, 3),
        np.zeros(3),
        np.zeros(3),
        matrix,
        np.asarray(distortion, dtype=np.float64),
    )
    return projected.reshape(-1, 2)


def _observations(camera: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray]:
    xyz: list[list[float]] = []
    uv: list[list[float]] = []
    for observation in camera["observations"]:
        if observation["split"] != split:
            continue
        for landmark in observation["landmarks"]:
            xyz.append(landmark["point_link_m"])
            uv.append(landmark["pixel"])
    return np.asarray(xyz, dtype=np.float64), np.asarray(uv, dtype=np.float64)


def fit_camera(camera: dict[str, Any]) -> CameraFit:
    train_xyz, train_uv = _observations(camera, "training")
    held_xyz, held_uv = _observations(camera, "held_out")
    initial = _camera_vector(camera)
    priors = camera["priors"]

    def unpack(values: np.ndarray):
        return values[:3], values[3:6], values[6:10], values[10:14]

    def residual(values: np.ndarray) -> np.ndarray:
        position, orientation, intrinsics, distortion = unpack(values)
        reprojection = (fisheye_project(train_xyz, position, orientation, intrinsics, distortion) - train_uv).ravel()
        # Holder registration and the 102-degree lens are weak priors. Dataset
        # landmarks dominate the solution.
        pose_prior = (values[:6] - initial[:6]) / np.asarray(priors["pose_sigma"])
        lens_prior = (values[6:10] - initial[6:10]) / np.asarray(priors["intrinsic_sigma"])
        distortion_prior = values[10:14] / float(priors["distortion_sigma"])
        return np.concatenate((reprojection, pose_prior, lens_prior, distortion_prior))

    result = least_squares(
        residual,
        initial,
        method="trf",
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=5000,
    )
    position, orientation, intrinsics, distortion = unpack(result.x)

    def errors(xyz: np.ndarray, uv: np.ndarray) -> np.ndarray:
        if not len(xyz):
            return np.empty(0)
        prediction = fisheye_project(xyz, position, orientation, intrinsics, distortion)
        return np.linalg.norm(prediction - uv, axis=1)

    return CameraFit(
        position=position,
        orientation=orientation,
        intrinsics=intrinsics,
        distortion=distortion,
        training_errors=errors(train_xyz, train_uv),
        held_out_errors=errors(held_xyz, held_uv),
    )


def fit_manifest(manifest: dict[str, Any]) -> dict[str, CameraFit]:
    return {side: fit_camera(manifest["cameras"][side]) for side in ("left", "right")}


def fit_metrics(fit: CameraFit) -> dict[str, float]:
    return {
        "training_median_px": float(np.median(fit.training_errors)),
        "training_p95_px": float(np.percentile(fit.training_errors, 95)),
        "held_out_median_px": float(np.median(fit.held_out_errors)),
        "held_out_p95_px": float(np.percentile(fit.held_out_errors, 95)),
    }


def held_out_geometry_metrics(camera: dict[str, Any], fit: CameraFit, frame_size=(1280, 720)) -> dict[str, float]:
    centroid_errors = []
    separation_errors = []
    roll_errors = []
    for observation in camera["observations"]:
        if observation["split"] != "held_out":
            continue
        landmarks = observation["landmarks"][:2]
        xyz = np.asarray([item["point_link_m"] for item in landmarks], dtype=np.float64)
        actual = np.asarray([item["pixel"] for item in landmarks], dtype=np.float64)
        predicted = fisheye_project(xyz, fit.position, fit.orientation, fit.intrinsics, fit.distortion)
        centroid_errors.append(np.linalg.norm(predicted.mean(axis=0) - actual.mean(axis=0)) / max(frame_size))
        predicted_delta = predicted[1] - predicted[0]
        actual_delta = actual[1] - actual[0]
        separation_errors.append(abs(np.linalg.norm(predicted_delta) - np.linalg.norm(actual_delta)) / max(frame_size))
        predicted_roll = np.degrees(np.arctan2(predicted_delta[1], predicted_delta[0]))
        actual_roll = np.degrees(np.arctan2(actual_delta[1], actual_delta[0]))
        roll_errors.append(abs(predicted_roll - actual_roll))
    return {
        "normalized_jaw_centroid_error": float(np.median(centroid_errors)),
        "normalized_jaw_separation_error": float(np.median(separation_errors)),
        "roll_error_deg": float(np.median(roll_errors)),
    }
