"""Dataset-driven calibration for rigid wrist cameras.

The transform convention is camera pose in the parent link frame.  Points are
converted to the OpenCV optical frame before applying OpenCV's native fisheye
projection model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation
import yaml

from robodojo.sim.environment.camera_manager.mount_registry import mount_orientation


@dataclass(frozen=True)
class CameraFit:
    position: np.ndarray
    orientation: np.ndarray
    intrinsics: np.ndarray
    distortion: np.ndarray
    training_errors: np.ndarray
    held_out_errors: np.ndarray


@dataclass(frozen=True)
class CorrectionBounds:
    shared_translation_m: float
    shared_rotation_deg: float
    per_arm_residual_translation_m: float
    per_arm_residual_rotation_deg: float
    visual_clamp_translation_m: float
    visual_clamp_rotation_deg: float


@dataclass(frozen=True)
class MirroredCorrectionFit:
    shared: np.ndarray
    left: np.ndarray
    right_mirrored: np.ndarray
    visual_clamp_left: np.ndarray
    visual_clamp_right: np.ndarray
    left_residual_translation_m: float
    left_residual_rotation_deg: float
    right_residual_translation_m: float
    right_residual_rotation_deg: float

    def to_dict(self) -> dict[str, Any]:
        def norms(correction: np.ndarray) -> dict[str, float]:
            return {
                "translation_m": float(np.linalg.norm(correction[:3])),
                "rotation_deg": float(np.linalg.norm(correction[3:])),
            }

        return {
            "parameter_order": ["tx_m", "ty_m", "tz_m", "rx_deg", "ry_deg", "rz_deg"],
            "camera_correction": {
                "shared": self.shared.tolist(),
                "left": self.left.tolist(),
                "right_mirrored": self.right_mirrored.tolist(),
            },
            "visual_clamp_correction": {
                "left": self.visual_clamp_left.tolist(),
                "right": self.visual_clamp_right.tolist(),
            },
            "residuals": {
                "left_translation_m": self.left_residual_translation_m,
                "left_rotation_deg": self.left_residual_rotation_deg,
                "right_translation_m": self.right_residual_translation_m,
                "right_rotation_deg": self.right_residual_rotation_deg,
            },
            "norms": {
                "camera_shared": norms(self.shared),
                "camera_left": norms(self.left),
                "camera_right_mirrored": norms(self.right_mirrored),
                "visual_clamp_left": norms(self.visual_clamp_left),
                "visual_clamp_right": norms(self.visual_clamp_right),
            },
        }


@dataclass(frozen=True)
class YamMatchedFrameFit:
    correction: MirroredCorrectionFit
    held_out_baseline_median_px: float
    held_out_corrected_median_px: float
    held_out_improvement_fraction: float
    held_out_by_side: dict[str, dict[str, float]]
    applied_camera_poses: dict[str, dict[str, list[float]]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "correction": self.correction.to_dict(),
            "held_out": {
                "baseline_median_px": self.held_out_baseline_median_px,
                "corrected_median_px": self.held_out_corrected_median_px,
                "improvement_fraction": self.held_out_improvement_fraction,
                "by_side": self.held_out_by_side,
            },
            "applied_camera_poses": self.applied_camera_poses,
        }


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


def _finite_array(value, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result


def _validate_yam_frame(
    frame: dict[str, Any], manifest: dict[str, Any], *, require_annotations: bool
) -> None:
    dataset_name = frame.get("dataset")
    if dataset_name not in manifest["datasets"]:
        raise ValueError(f"unknown YAM matched-frame dataset: {dataset_name!r}")
    if frame.get("split") != manifest["datasets"][dataset_name].get("split"):
        raise ValueError("YAM matched-frame split must match its pinned dataset split")
    dataset = manifest["datasets"][dataset_name]
    if frame.get("source_revision") != dataset.get("revision"):
        raise ValueError("YAM matched-frame source revision must match its pinned dataset")
    for name in ("episode_index", "frame_index"):
        if not isinstance(frame.get(name), int) or frame[name] < 0:
            raise ValueError(f"YAM matched frame {name} must be a non-negative integer")
    if not np.isfinite(frame.get("timestamp_s", np.nan)) or frame["timestamp_s"] < 0:
        raise ValueError("YAM matched frame timestamp_s must be finite and non-negative")
    state_dimension = int(manifest["dataset_contract"]["state_dimension"])
    _finite_array(frame.get("observation_state"), (state_dimension,), "YAM matched-frame observation_state")

    required_cameras = {key.rsplit(".", 1)[-1] for key in manifest["selection_contract"]["required_camera_keys"]}
    frame_sha256 = frame.get("frame_sha256", {})
    if set(frame_sha256) != required_cameras:
        raise ValueError("YAM matched frame must checksum all three pinned camera images")
    for digest in frame_sha256.values():
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("YAM matched frame image has an invalid sha256")
    camera_sources = frame.get("camera_sources", {})
    if set(camera_sources) != required_cameras:
        raise ValueError("YAM matched frame requires provenance for all three camera sources")
    for camera_name, source in camera_sources.items():
        if source.get("video_path") != dataset["videos"][camera_name]:
            raise ValueError("YAM matched-frame video path does not match its pinned dataset")
        if source.get("image_sha256") != frame_sha256[camera_name]:
            raise ValueError("YAM matched-frame camera provenance checksum is inconsistent")
        video_digest = source.get("video_sha256", "")
        if len(video_digest) != 64 or any(
            character not in "0123456789abcdef" for character in video_digest
        ):
            raise ValueError("YAM matched frame video has an invalid sha256")

    wrists = frame.get("wrist_annotations", {})
    if not wrists and not require_annotations:
        return
    if set(wrists) != {"left", "right"}:
        raise ValueError("YAM matched frame requires independent left and right wrist annotations")
    minimum = int(manifest["annotation_contract"]["minimum_landmarks_per_wrist_per_frame"])
    for side, annotation in wrists.items():
        landmarks = annotation.get("landmarks", [])
        if len(landmarks) < minimum:
            raise ValueError(f"YAM matched frame {side} wrist requires at least {minimum} landmarks")
        for landmark in landmarks:
            if not landmark.get("name"):
                raise ValueError("YAM matched-frame landmark requires a name")
            _finite_array(landmark.get("point_link_m"), (3,), "point_link_m")
            _finite_array(landmark.get("pixel"), (2,), "pixel")
            _finite_array(landmark.get("point_gripper_m"), (3,), "point_gripper_m")
            if landmark.get("link_name") != manifest["annotation_contract"]["parent_frame"]:
                raise ValueError("YAM matched-frame landmark uses the wrong parent frame")
            if landmark.get("visible") is not True or landmark.get("occluded") is not False:
                raise ValueError("YAM matched-frame calibration only accepts visible landmarks")
            uncertainty = landmark.get("uncertainty_px_radius")
            if not np.isfinite(uncertainty) or uncertainty <= 0:
                raise ValueError("YAM matched-frame landmark uncertainty must be positive")
        expected_index = 6 if side == "left" else 13
        if annotation.get("acting_gripper_state_index") != expected_index:
            raise ValueError(f"YAM matched frame {side} wrist uses the wrong acting gripper")
        if not np.isclose(
            annotation.get("acting_gripper_policy_value", np.nan),
            frame["observation_state"][expected_index],
            atol=0.0,
            rtol=0.0,
        ):
            raise ValueError(f"YAM matched frame {side} wrist acting gripper value is inconsistent")


def _normalize_yam_annotation_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Map the reviewed annotation artifact to the compact runtime contract."""
    source = sample["source"]
    camera_sources = sample["all_camera_sources"]
    wrist_annotations: dict[str, Any] = {}
    for side, camera in sample["wrist_cameras"].items():
        landmarks = []
        for tip_name, tip_landmarks in camera["landmarks_2d"].items():
            for landmark_name, landmark in tip_landmarks.items():
                landmarks.append(
                    {
                        "name": f"{tip_name}_{landmark_name}",
                        "point_link_m": landmark["point_link_m"],
                        "pixel": landmark["pixel_uv"],
                        "uncertainty_px_radius": landmark["uncertainty_px_radius"],
                        "visible": landmark["visible"],
                        "occluded": landmark["occluded"],
                        "link_name": landmark["link_name"],
                        "point_gripper_m": landmark["point_gripper_m"],
                    }
                )
        wrist_annotations[side] = {
            "acting_gripper_state_index": camera["acting_gripper_state_index"],
            "acting_gripper_policy_value": camera["acting_gripper_policy_value"],
            "landmarks": landmarks,
        }
    return {
        "sample_id": sample["sample_id"],
        "phase": sample["phase"],
        "dataset": source["repo_id"],
        "source_revision": source["revision"],
        "source_license": source["license"],
        "split": sample["split"],
        "episode_index": source["episode_index"],
        "frame_index": source["frame_index"],
        "timestamp_s": source["timestamp_seconds"],
        "data_provenance": source["data_provenance"],
        "observation_state": sample["state"],
        "frame_sha256": {
            name: camera["image_sha256"] for name, camera in camera_sources.items()
        },
        "camera_sources": camera_sources,
        "wrist_annotations": wrist_annotations,
    }


def _load_yam_annotation_artifact(manifest: dict[str, Any], manifest_path: Path) -> list[dict[str, Any]]:
    contract = manifest.get("annotation_artifact")
    if not isinstance(contract, dict):
        raise ValueError("YAM matched-frame calibration requires an annotation_artifact")
    artifact_path = manifest_path.parent / contract["path"]
    if not artifact_path.is_file():
        raise ValueError(
            f"YAM matched-frame landmark artifact is absent: {artifact_path}; "
            "restore the pinned artifact before calibration"
        )
    validate_frame(artifact_path, contract["sha256"])
    artifact = yaml.safe_load(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("format_version") != contract.get("format_version"):
        raise ValueError("YAM landmark artifact format_version does not match its manifest")
    if artifact.get("source_sample_manifest", {}).get("sha256") != contract.get(
        "sample_manifest_sha256"
    ):
        raise ValueError("YAM landmark artifact sample-manifest provenance does not match")
    if artifact.get("validation", {}).get("status") != "passed":
        raise ValueError("YAM landmark artifact did not pass its embedded validation")
    frames = [_normalize_yam_annotation_sample(sample) for sample in artifact.get("samples", [])]
    landmark_count = sum(
        len(frame["wrist_annotations"][side]["landmarks"])
        for frame in frames
        for side in ("left", "right")
    )
    if landmark_count != int(contract["landmark_count"]):
        raise ValueError("YAM landmark artifact count does not match its manifest")
    selected_ids = manifest["selection_contract"].get("frames", [])
    artifact_ids = [frame["sample_id"] for frame in frames]
    if selected_ids != artifact_ids:
        raise ValueError("YAM landmark artifact samples do not exactly match the selected frame IDs")
    return frames


def load_yam_matched_manifest(path: Path, *, require_complete: bool = False) -> dict[str, Any]:
    """Load the released-frame YAM contract without accepting partial fits."""
    manifest = yaml.safe_load(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("profile_id") != "bimanual_yam":
        raise ValueError("unsupported YAM matched-frame calibration manifest")
    status = manifest.get("status")
    if status not in {"incomplete", "annotated", "complete"}:
        raise ValueError("YAM matched-frame calibration status must be incomplete, annotated, or complete")
    if status == "incomplete" and not manifest.get("status_reason"):
        raise ValueError("incomplete YAM matched-frame calibration requires a status_reason")

    expected = int(manifest["selection_contract"]["expected_frame_count"])
    per_dataset = int(manifest["selection_contract"]["frames_per_dataset"])
    if expected != 24 or per_dataset != 8 or len(manifest.get("datasets", {})) != 3:
        raise ValueError("YAM matched-frame calibration must select 24 frames, eight from each dataset")
    split_by_dataset = {name: dataset.get("split") for name, dataset in manifest["datasets"].items()}
    if list(split_by_dataset.values()).count("training") != 2 or list(split_by_dataset.values()).count("held_out") != 1:
        raise ValueError("YAM matched-frame calibration requires two training datasets and one held-out dataset")
    for dataset_name, dataset in manifest["datasets"].items():
        if set(dataset.get("videos", {})) != {"top", "left", "right"}:
            raise ValueError(f"YAM matched-frame dataset {dataset_name} must pin all three video paths")
    frame_ids = manifest["selection_contract"].get("frames", [])
    if len(frame_ids) != expected or not all(isinstance(sample_id, str) for sample_id in frame_ids):
        raise ValueError("YAM matched-frame selection must contain exactly 24 released sample IDs")
    frames = _load_yam_annotation_artifact(manifest, path)
    manifest["selection_contract"]["selected_sample_ids"] = frame_ids
    manifest["selection_contract"]["frames"] = frames
    identities = [(frame.get("dataset"), frame.get("episode_index"), frame.get("frame_index")) for frame in frames]
    if len(identities) != len(set(identities)):
        raise ValueError("YAM matched-frame selection contains duplicate frame identities")
    for dataset_name in manifest["datasets"]:
        if sum(frame.get("dataset") == dataset_name for frame in frames) != per_dataset:
            raise ValueError(f"YAM matched-frame selection must contain eight frames from {dataset_name}")
    for frame in frames:
        _validate_yam_frame(frame, manifest, require_annotations=True)
    if status == "complete" and not isinstance(manifest["fit_contract"].get("fit_result"), dict):
        raise ValueError("complete YAM matched-frame calibration must persist its accepted fit_result")
    if require_complete and status != "complete":
        raise ValueError(f"YAM matched-frame calibration is incomplete: {manifest['status_reason']}")
    return manifest


def yam_matched_manifest_status(manifest: dict[str, Any]) -> dict[str, Any]:
    frames = manifest["selection_contract"].get("frames", [])
    return {
        "profile_id": manifest["profile_id"],
        "status": manifest["status"],
        "reason": manifest.get("status_reason"),
        "selected_frames": len(frames),
        "expected_frames": int(manifest["selection_contract"]["expected_frame_count"]),
        "fit_enabled": manifest["status"] in {"annotated", "complete"},
    }


def _parameter_mirror_matrix(mirror_matrix) -> np.ndarray:
    mirror = _finite_array(mirror_matrix, (3, 3), "mirror_matrix")
    if not np.allclose(mirror.T @ mirror, np.eye(3), atol=1e-9) or not np.allclose(
        mirror @ mirror, np.eye(3), atol=1e-9
    ):
        raise ValueError("mirror_matrix must be an orthogonal involution")
    determinant = float(np.linalg.det(mirror))
    if not np.isclose(determinant, -1.0, atol=1e-9):
        raise ValueError("mirror_matrix must describe a reflection")
    result = np.zeros((6, 6), dtype=np.float64)
    result[:3, :3] = mirror
    result[3:, 3:] = determinant * mirror
    return result


def _solve_bounded_correction(
    jacobian,
    residual,
    translation_limit: float,
    rotation_limit: float,
    label: str,
    *,
    center=None,
) -> np.ndarray:
    """Solve a deterministic linear least-squares fit inside two L2 balls."""
    jacobian = np.asarray(jacobian, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    if jacobian.ndim != 2 or jacobian.shape[1] != 6 or residual.shape != (jacobian.shape[0],):
        raise ValueError(f"{label} correction system must have shapes (N, 6) and (N,)")
    if not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(residual)):
        raise ValueError(f"{label} correction system must be finite")
    rank = np.linalg.matrix_rank(jacobian)
    if rank != 6:
        raise ValueError(f"{label} correction system is rank deficient ({rank}/6)")
    if translation_limit <= 0 or rotation_limit <= 0:
        raise ValueError(f"{label} correction bounds must be positive")
    origin = np.zeros(6, dtype=np.float64) if center is None else _finite_array(center, (6,), label)
    scale = np.asarray([translation_limit] * 3 + [rotation_limit] * 3, dtype=np.float64)

    def objective(unit_vector: np.ndarray) -> float:
        error = jacobian @ (origin + unit_vector * scale) - residual
        return float(np.dot(error, error) / error.size)

    constraints = (
        {"type": "ineq", "fun": lambda value: 1.0 - float(np.dot(value[:3], value[:3]))},
        {"type": "ineq", "fun": lambda value: 1.0 - float(np.dot(value[3:], value[3:]))},
    )
    result = minimize(
        objective,
        np.zeros(6, dtype=np.float64),
        method="SLSQP",
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise ValueError(f"{label} bounded correction fit failed: {result.message}")
    solution = origin + result.x * scale
    _check_correction_magnitude(solution - origin, translation_limit, rotation_limit, label)
    return solution


def _rotation_distance_deg(left: np.ndarray, right: np.ndarray) -> float:
    left_rotation = Rotation.from_rotvec(np.deg2rad(left[3:]))
    right_rotation = Rotation.from_rotvec(np.deg2rad(right[3:]))
    return float(np.rad2deg((left_rotation.inv() * right_rotation).magnitude()))


def _compose_corrections(base: np.ndarray, local: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(np.deg2rad(base[3:])) * Rotation.from_rotvec(np.deg2rad(local[3:]))
    result = np.empty(6, dtype=np.float64)
    result[:3] = base[:3] + local[:3]
    result[3:] = np.rad2deg(rotation.as_rotvec())
    return result


def _check_correction_magnitude(
    vector: np.ndarray, translation_limit: float, rotation_limit: float, label: str
) -> None:
    translation = float(np.linalg.norm(vector[:3]))
    rotation = float(np.linalg.norm(vector[3:]))
    if translation > translation_limit + 1e-12 or rotation > rotation_limit + 1e-12:
        raise ValueError(
            f"{label} exceeds bounds: translation={translation:.9f}m/{translation_limit:.9f}m, "
            f"rotation={rotation:.9f}deg/{rotation_limit:.9f}deg"
        )


def fit_bounded_mirrored_correction(
    left_jacobian,
    left_residual,
    right_jacobian,
    right_residual,
    mirror_matrix,
    bounds: CorrectionBounds,
) -> MirroredCorrectionFit:
    """Fit one mirrored wrist correction and enforce all physical bounds."""
    parameter_mirror = _parameter_mirror_matrix(mirror_matrix)
    left_jacobian = np.asarray(left_jacobian, dtype=np.float64)
    right_jacobian = np.asarray(right_jacobian, dtype=np.float64)
    left_residual = np.asarray(left_residual, dtype=np.float64)
    right_residual = np.asarray(right_residual, dtype=np.float64)

    shared = _solve_bounded_correction(
        np.vstack((left_jacobian, right_jacobian @ parameter_mirror)),
        np.concatenate((left_residual, right_residual)),
        bounds.shared_translation_m,
        bounds.shared_rotation_deg,
        "shared mirrored wrist correction",
    )
    left = _solve_bounded_correction(
        left_jacobian,
        left_residual,
        bounds.per_arm_residual_translation_m,
        bounds.per_arm_residual_rotation_deg,
        "left wrist residual correction",
        center=shared,
    )
    right_mirrored = _solve_bounded_correction(
        right_jacobian @ parameter_mirror,
        right_residual,
        bounds.per_arm_residual_translation_m,
        bounds.per_arm_residual_rotation_deg,
        "right wrist residual correction",
        center=shared,
    )

    left_translation_residual = float(np.linalg.norm(left[:3] - shared[:3]))
    right_translation_residual = float(np.linalg.norm(right_mirrored[:3] - shared[:3]))
    left_rotation_residual = _rotation_distance_deg(left, shared)
    right_rotation_residual = _rotation_distance_deg(right_mirrored, shared)
    if (
        max(left_translation_residual, right_translation_residual)
        > bounds.per_arm_residual_translation_m + 1e-9
    ):
        raise ValueError("per-arm wrist translation residual exceeds calibration bounds")
    if max(left_rotation_residual, right_rotation_residual) > bounds.per_arm_residual_rotation_deg + 1e-9:
        raise ValueError("per-arm wrist rotation residual exceeds calibration bounds")

    return MirroredCorrectionFit(
        shared=shared,
        left=left,
        right_mirrored=right_mirrored,
        visual_clamp_left=np.zeros(6, dtype=np.float64),
        visual_clamp_right=np.zeros(6, dtype=np.float64),
        left_residual_translation_m=left_translation_residual,
        left_residual_rotation_deg=left_rotation_residual,
        right_residual_translation_m=right_translation_residual,
        right_residual_rotation_deg=right_rotation_residual,
    )


def pinhole_project(
    points_link: np.ndarray,
    position: np.ndarray,
    orientation_wxyz: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Project link-frame points through an Isaac/USD pinhole camera."""
    points_link = np.asarray(points_link, dtype=np.float64)
    position = _finite_array(position, (3,), "pinhole camera position")
    orientation = _finite_array(orientation_wxyz, (4,), "pinhole camera orientation")
    intrinsics = _finite_array(intrinsics, (4,), "pinhole camera intrinsics")
    camera_to_link = Rotation.from_quat(orientation[[1, 2, 3, 0]])
    points_camera = camera_to_link.inv().apply(points_link - position)
    depth = -points_camera[:, 2]
    if np.any(depth <= 1e-9):
        raise ValueError("pinhole calibration landmark is on or behind the camera plane")
    fx, fy, cx, cy = intrinsics
    return np.column_stack((fx * points_camera[:, 0] / depth + cx, cy - fy * points_camera[:, 1] / depth))


def _correct_pinhole_pose(position: np.ndarray, orientation: np.ndarray, correction: np.ndarray):
    correction = _finite_array(correction, (6,), "pinhole pose correction")
    base = Rotation.from_quat(orientation[[1, 2, 3, 0]])
    delta = Rotation.from_rotvec(np.deg2rad(correction[3:]))
    xyzw = (delta * base).as_quat()
    return position + correction[:3], xyzw[[3, 0, 1, 2]]


def pinhole_pose_jacobian(
    points_link: np.ndarray,
    position: np.ndarray,
    orientation_wxyz: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Numerically differentiate matched pixels w.r.t. the bounded 6D pose correction."""
    points_link = np.asarray(points_link, dtype=np.float64)
    position = _finite_array(position, (3,), "pinhole camera position")
    orientation = _finite_array(orientation_wxyz, (4,), "pinhole camera orientation")
    baseline = pinhole_project(points_link, position, orientation, intrinsics)
    steps = np.asarray([1e-5, 1e-5, 1e-5, 1e-3, 1e-3, 1e-3], dtype=np.float64)
    jacobian = np.empty((baseline.size, 6), dtype=np.float64)
    for index, step in enumerate(steps):
        correction = np.zeros(6, dtype=np.float64)
        correction[index] = step
        corrected_position, corrected_orientation = _correct_pinhole_pose(position, orientation, correction)
        shifted = pinhole_project(points_link, corrected_position, corrected_orientation, intrinsics)
        jacobian[:, index] = ((shifted - baseline) / step).ravel()
    return jacobian


def _yam_camera_pose(camera_config: dict[str, Any], side: str):
    key = f"cam_{side}_wrist"
    camera = camera_config["camera_rig"]["cameras"][key]
    mount = camera["mount"]
    position = _finite_array(mount["position"], (3,), f"{key} position")
    orientation = mount_orientation(
        mount["orientation"],
        mount.get("pose_convention", "isaac_usd"),
        float(mount.get("optical_roll_deg", 0.0)),
    )
    projection = camera["projection"]
    intrinsics = np.asarray([projection[name] for name in ("fx", "fy", "cx", "cy")], dtype=np.float64)
    return position, orientation, intrinsics


def _yam_landmark_system(
    manifest: dict[str, Any], camera_config: dict[str, Any], side: str, split: str
) -> tuple[np.ndarray, np.ndarray]:
    position, orientation, intrinsics = _yam_camera_pose(camera_config, side)
    jacobians = []
    residuals = []
    for frame in manifest["selection_contract"]["frames"]:
        if frame["split"] != split:
            continue
        landmarks = frame["wrist_annotations"][side]["landmarks"]
        points = np.asarray([landmark["point_link_m"] for landmark in landmarks], dtype=np.float64)
        reference = np.asarray([landmark["pixel"] for landmark in landmarks], dtype=np.float64)
        rendered = pinhole_project(points, position, orientation, intrinsics)
        jacobians.append(pinhole_pose_jacobian(points, position, orientation, intrinsics))
        residuals.append((reference - rendered).ravel())
    return np.vstack(jacobians), np.concatenate(residuals)


def _correct_visual_clamp_points(
    points: np.ndarray,
    names: list[str],
    correction: np.ndarray,
    mirror_matrix,
) -> np.ndarray:
    """Move only jaw-derived visual points, mirroring tip_right from tip_left."""
    points = np.asarray(points, dtype=np.float64)
    correction = _finite_array(correction, (6,), "visual clamp correction")
    parameter_mirror = _parameter_mirror_matrix(mirror_matrix)
    corrected = np.empty_like(points)
    for index, (point, name) in enumerate(zip(points, names, strict=True)):
        if name.startswith("tip_left_"):
            local = correction
        elif name.startswith("tip_right_"):
            local = parameter_mirror @ correction
        else:
            raise ValueError(f"visual clamp landmark does not identify a jaw tip: {name!r}")
        rotation = Rotation.from_rotvec(np.deg2rad(local[3:]))
        corrected[index] = rotation.apply(point) + local[:3]
    return corrected


def _yam_visual_clamp_residual(
    manifest: dict[str, Any],
    camera_config: dict[str, Any],
    side: str,
    split: str,
    camera_correction: np.ndarray,
    clamp_correction: np.ndarray,
) -> np.ndarray:
    position, orientation, intrinsics = _yam_camera_pose(camera_config, side)
    position, orientation = _correct_pinhole_pose(position, orientation, camera_correction)
    residuals = []
    for frame in manifest["selection_contract"]["frames"]:
        if frame["split"] != split:
            continue
        landmarks = frame["wrist_annotations"][side]["landmarks"]
        names = [landmark["name"] for landmark in landmarks]
        points = np.asarray([landmark["point_link_m"] for landmark in landmarks], dtype=np.float64)
        points = _correct_visual_clamp_points(
            points,
            names,
            clamp_correction,
            manifest["fit_contract"]["mirror_matrix"],
        )
        reference = np.asarray([landmark["pixel"] for landmark in landmarks], dtype=np.float64)
        residuals.append((pinhole_project(points, position, orientation, intrinsics) - reference).ravel())
    if not residuals:
        raise ValueError(f"YAM visual clamp fit has no {split} observations for {side} wrist")
    return np.concatenate(residuals)


def _fit_yam_visual_clamp(
    manifest: dict[str, Any],
    camera_config: dict[str, Any],
    side: str,
    camera_correction: np.ndarray,
    bounds: CorrectionBounds,
) -> np.ndarray:
    """Fit a visual-only mirrored jaw correction with exact norm constraints."""
    scale = np.asarray(
        [bounds.visual_clamp_translation_m] * 3 + [bounds.visual_clamp_rotation_deg] * 3,
        dtype=np.float64,
    )

    def objective(unit_vector: np.ndarray) -> float:
        residual = _yam_visual_clamp_residual(
            manifest,
            camera_config,
            side,
            "training",
            camera_correction,
            unit_vector * scale,
        )
        return float(np.dot(residual, residual) / residual.size)

    constraints = (
        {"type": "ineq", "fun": lambda value: 1.0 - float(np.dot(value[:3], value[:3]))},
        {"type": "ineq", "fun": lambda value: 1.0 - float(np.dot(value[3:], value[3:]))},
    )
    result = minimize(
        objective,
        np.zeros(6, dtype=np.float64),
        method="SLSQP",
        bounds=[(-1.0, 1.0)] * 6,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise ValueError(f"{side} visual clamp bounded fit failed: {result.message}")
    correction = result.x * scale
    _check_correction_magnitude(
        correction,
        bounds.visual_clamp_translation_m,
        bounds.visual_clamp_rotation_deg,
        f"{side} visual clamp correction",
    )
    return correction


def _yam_held_out_errors(
    manifest: dict[str, Any],
    camera_config: dict[str, Any],
    correction: MirroredCorrectionFit,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, dict[str, list[float]]],
    dict[str, dict[str, float]],
]:
    parameter_mirror = _parameter_mirror_matrix(manifest["fit_contract"]["mirror_matrix"])
    corrections = {
        "left": correction.left,
        "right": parameter_mirror @ correction.right_mirrored,
    }
    clamp_corrections = {
        "left": correction.visual_clamp_left,
        "right": correction.visual_clamp_right,
    }
    baseline_errors = []
    corrected_errors = []
    applied_poses = {}
    metrics_by_side = {}
    for side in ("left", "right"):
        position, orientation, intrinsics = _yam_camera_pose(camera_config, side)
        corrected_position, corrected_orientation = _correct_pinhole_pose(position, orientation, corrections[side])
        applied_poses[side] = {
            "position": corrected_position.tolist(),
            "orientation_wxyz": corrected_orientation.tolist(),
        }
        side_baseline_errors = []
        side_corrected_errors = []
        for frame in manifest["selection_contract"]["frames"]:
            if frame["split"] != "held_out":
                continue
            landmarks = frame["wrist_annotations"][side]["landmarks"]
            raw_points = np.asarray([landmark["point_link_m"] for landmark in landmarks], dtype=np.float64)
            corrected_points = _correct_visual_clamp_points(
                raw_points,
                [landmark["name"] for landmark in landmarks],
                clamp_corrections[side],
                manifest["fit_contract"]["mirror_matrix"],
            )
            reference = np.asarray([landmark["pixel"] for landmark in landmarks], dtype=np.float64)
            baseline = pinhole_project(raw_points, position, orientation, intrinsics)
            corrected = pinhole_project(
                corrected_points,
                corrected_position,
                corrected_orientation,
                intrinsics,
            )
            frame_baseline_errors = np.linalg.norm(baseline - reference, axis=1)
            frame_corrected_errors = np.linalg.norm(corrected - reference, axis=1)
            baseline_errors.extend(frame_baseline_errors)
            corrected_errors.extend(frame_corrected_errors)
            side_baseline_errors.extend(frame_baseline_errors)
            side_corrected_errors.extend(frame_corrected_errors)
        side_baseline_median = float(np.median(side_baseline_errors))
        side_corrected_median = float(np.median(side_corrected_errors))
        metrics_by_side[side] = {
            "baseline_median_px": side_baseline_median,
            "corrected_median_px": side_corrected_median,
            "improvement_fraction": 1.0 - side_corrected_median / side_baseline_median,
        }
    return np.asarray(baseline_errors), np.asarray(corrected_errors), applied_poses, metrics_by_side


def fit_yam_matched_manifest(manifest: dict[str, Any], camera_config: dict[str, Any]) -> YamMatchedFrameFit:
    if manifest.get("profile_id") != "bimanual_yam" or manifest.get("status") not in {"annotated", "complete"}:
        raise ValueError("YAM matched-frame calibration is incomplete and cannot be fitted")
    left_jacobian, left_residual = _yam_landmark_system(manifest, camera_config, "left", "training")
    right_jacobian, right_residual = _yam_landmark_system(manifest, camera_config, "right", "training")
    bound_values = manifest["fit_contract"]["bounds"]
    bounds = CorrectionBounds(**bound_values)
    correction = fit_bounded_mirrored_correction(
        left_jacobian,
        left_residual,
        right_jacobian,
        right_residual,
        manifest["fit_contract"]["mirror_matrix"],
        bounds,
    )
    parameter_mirror = _parameter_mirror_matrix(manifest["fit_contract"]["mirror_matrix"])
    correction = replace(
        correction,
        visual_clamp_left=_fit_yam_visual_clamp(
            manifest,
            camera_config,
            "left",
            correction.left,
            bounds,
        ),
        visual_clamp_right=_fit_yam_visual_clamp(
            manifest,
            camera_config,
            "right",
            parameter_mirror @ correction.right_mirrored,
            bounds,
        ),
    )
    baseline_errors, corrected_errors, applied_poses, metrics_by_side = _yam_held_out_errors(
        manifest, camera_config, correction
    )
    if not len(baseline_errors) or not len(corrected_errors):
        raise ValueError("YAM matched-frame calibration has no held-out landmark errors")
    baseline_median = float(np.median(baseline_errors))
    corrected_median = float(np.median(corrected_errors))
    improvement = 0.0 if baseline_median <= 1e-12 else 1.0 - corrected_median / baseline_median
    acceptance = manifest["fit_contract"]["held_out_acceptance"]
    median_limit = float(acceptance["median_pixel_error"])
    improvement_limit = float(acceptance["minimum_improvement_fraction"])
    for side, metrics in metrics_by_side.items():
        if metrics["corrected_median_px"] > median_limit + 1e-12:
            raise ValueError(
                f"YAM {side} held-out median pixel error exceeds calibration acceptance: "
                f"{metrics['corrected_median_px']:.6f}px > {median_limit:.6f}px"
            )
        if metrics["improvement_fraction"] < improvement_limit - 1e-12:
            raise ValueError(
                f"YAM {side} held-out pixel improvement is below calibration acceptance: "
                f"{metrics['improvement_fraction']:.6f} < {improvement_limit:.6f}"
            )
    if corrected_median > median_limit + 1e-12:
        raise ValueError(
            "YAM held-out median pixel error exceeds calibration acceptance: "
            f"{corrected_median:.6f}px > {median_limit:.6f}px "
            f"(baseline {baseline_median:.6f}px, improvement {improvement:.6f})"
        )
    if improvement < improvement_limit - 1e-12:
        raise ValueError(
            "YAM held-out pixel improvement is below calibration acceptance: "
            f"{improvement:.6f} < {improvement_limit:.6f} "
            f"(baseline {baseline_median:.6f}px, corrected {corrected_median:.6f}px)"
        )
    return YamMatchedFrameFit(
        correction=correction,
        held_out_baseline_median_px=baseline_median,
        held_out_corrected_median_px=corrected_median,
        held_out_improvement_fraction=improvement,
        held_out_by_side=metrics_by_side,
        applied_camera_poses=applied_poses,
    )


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
