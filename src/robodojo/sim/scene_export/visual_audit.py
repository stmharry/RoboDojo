"""Runtime-only visual audit for post-reset scene exports."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from robodojo.sim.scene_export.contracts import (
    camera_axes,
    exact_simulation_steps,
    forward_ray_plane_intersection,
    geometric_cloth_support,
    project_points_to_camera,
    vector_drift,
)

logger = logging.getLogger(__name__)

AUDIT_DIRECTORY_NAME = "visual_audit"
AUDIT_DURATION_SECONDS = 2.0
METRICS_NAME = "metrics.json"
METADATA_NAME = "metadata.json"


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        return [_json_value(item) for item in value]
    except TypeError:
        return str(value)


def _as_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, RuntimeError):
        return None
    return array if np.isfinite(array).all() else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(repo_root: Path) -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def _capture_rgb_frames(env) -> dict[str, np.ndarray]:
    """Capture camera RGB directly, bypassing the policy observation manager."""
    env.render()
    camera_names = list(env.camera_manager.camera_names[0])
    data = env.capture_manager.step(env_ids=[0])
    if len(data) != len(camera_names):
        raise RuntimeError(f"camera capture count mismatch: got {len(data)}, expected {len(camera_names)}")
    frames = {}
    for cam_id, camera_name in enumerate(camera_names):
        rgb_rows = data[cam_id].get("rgb")
        if not rgb_rows:
            raise RuntimeError(f"camera {camera_name!r} has no enabled RGB annotator")
        frame = np.asarray(rgb_rows[0]["data"])
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise RuntimeError(f"camera {camera_name!r} returned invalid RGB shape {frame.shape}")
        frames[camera_name] = np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8)
    return frames


def _simulation_time(env) -> float | None:
    owners = (
        getattr(env, "sim", None),
        getattr(getattr(env, "sim", None), "unwrapped", None),
        getattr(getattr(env, "sim", None), "sim", None),
    )
    for owner in owners:
        for name in ("current_time", "_current_time"):
            if owner is not None and hasattr(owner, name):
                try:
                    return float(getattr(owner, name))
                except (TypeError, ValueError):
                    pass
    return None


def _simulation_step_seconds(env) -> tuple[float, int, float]:
    direct_env = getattr(getattr(env, "sim", None), "unwrapped", None)
    cfg = getattr(direct_env, "cfg", None)
    decimation = int(getattr(cfg, "decimation", 1))
    physics_dt = float(getattr(direct_env, "physics_dt", env.dt))
    # BaseEnv.sim_step advances one physics tick. Decimation is retained as
    # runtime provenance but does not scale this diagnostic hold count.
    return physics_dt, decimation, physics_dt


def _camera_runtime(env) -> dict[str, dict[str, Any]]:
    result = {}
    rig_by_key = {camera.observation_key: camera for camera in env.camera_rig.cameras}
    for cam_id, camera_name in enumerate(env.camera_manager.camera_names[0]):
        spec = rig_by_key[camera_name]
        extrinsic = np.asarray(env.camera_manager.get_camera_extrinsics(cam_id, 0), dtype=np.float64)
        width, height = [int(value) for value in spec.sensor["stream_resolution"]]
        runtime_intrinsic = env.camera_manager.get_camera_intrinsics(cam_id, 0)
        fx = float(spec.projection.get("fx", runtime_intrinsic[0, 0]))
        fy = float(spec.projection.get("fy", runtime_intrinsic[1, 1]))
        cx = float(spec.projection.get("cx", width / 2.0))
        cy = float(spec.projection.get("cy", height / 2.0))
        result[camera_name] = {
            "role": spec.role,
            "projection_model": str(spec.projection.get("model", "pinhole")),
            "resolution": [width, height],
            "intrinsic_matrix": np.asarray(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            "camera_to_world": extrinsic,
        }
    return result


def _robot_runtime(env) -> list[dict[str, Any]]:
    result = []
    manager = env.robot_manager
    for index, robot in enumerate(manager.robot_list):
        if robot.type != "target" or robot.robot_type != "arm":
            continue
        articulation = manager.robot_key[index]
        arm_indices = list(robot.arm_joint_indices)
        finger_indices = list(robot.gripper_joint_indices)
        positions = _as_numpy(articulation.data.joint_pos)
        velocities = _as_numpy(articulation.data.joint_vel)
        targets = _as_numpy(getattr(articulation.data, "joint_pos_target", None))
        end_effector_pose = None
        try:
            end_effector_pose = _as_numpy(manager.get_real_endpose(robot, env_idx_list=[0], is_relative=False).get(0))
            if end_effector_pose is not None:
                end_effector_pose = end_effector_pose.reshape(-1).tolist()
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            pass

        def select(array: np.ndarray | None, indices: list[int]) -> list[float] | None:
            if array is None:
                return None
            try:
                return np.asarray(array[0, indices], dtype=np.float64).reshape(-1).tolist()
            except (IndexError, TypeError, ValueError):
                return None

        result.append(
            {
                "arm_name": str(robot.arm_name),
                "gripper_name": str(robot.gripper_name),
                "arm_joint_names": [articulation.joint_names[joint] for joint in arm_indices],
                "finger_joint_names": [articulation.joint_names[joint] for joint in finger_indices],
                "arm_joint_positions": select(positions, arm_indices),
                "finger_joint_positions": select(positions, finger_indices),
                "arm_joint_velocities": select(velocities, arm_indices),
                "finger_joint_velocities": select(velocities, finger_indices),
                "arm_position_targets": select(targets, arm_indices),
                "finger_position_targets": select(targets, finger_indices),
                "end_effector_pose_world": end_effector_pose,
            }
        )
    return result


def _table_height_world(env) -> float | None:
    try:
        table = env.scene_manager.layout_manager.table_info[0]
        height = float(table["height"])
        origin = _as_numpy(env.env_origins[0])
        if origin is not None:
            height += float(origin.reshape(-1)[2])
        return height
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None


def _cloth_runtime(env) -> list[dict[str, Any]]:
    result = []
    try:
        garments = env.scene_manager._garment_objects[0]
    except (AttributeError, IndexError):
        return result
    table_height = _table_height_world(env)
    for name, garment in garments.items():
        points = None
        error = None
        try:
            points = _as_numpy(garment.sample_mesh_vertices()[0])
            if points is not None:
                points = points.reshape(-1, 3)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        velocities = None
        try:
            velocities = _as_numpy(garment._cloth_prim_view.get_velocities())
            if velocities is not None:
                velocities = velocities.reshape(-1, 3)
        except Exception:
            pass
        result.append(
            {
                "name": str(name),
                "prim_path": str(garment.usd_prim_path),
                "points_world": points,
                "velocities_world": velocities,
                "table_height_world_m": table_height,
                "capture_error": error,
            }
        )
    return result


def _cloth_summary(cloth: dict[str, Any]) -> dict[str, Any]:
    points = cloth["points_world"]
    velocities = cloth["velocities_world"]
    summary = {
        "name": cloth["name"],
        "prim_path": cloth["prim_path"],
        "capture_error": cloth["capture_error"],
        "particle_count": None,
        "bounds_world_m": None,
        "centroid_world_m": None,
        "max_particle_speed_m_s": None,
        "support": geometric_cloth_support(np.empty((0, 3)), cloth["table_height_world_m"]),
    }
    if points is not None:
        summary["particle_count"] = int(len(points))
        if len(points):
            summary["bounds_world_m"] = {
                "min": np.min(points, axis=0).tolist(),
                "max": np.max(points, axis=0).tolist(),
            }
            summary["centroid_world_m"] = np.mean(points, axis=0).tolist()
        summary["support"] = geometric_cloth_support(points, cloth["table_height_world_m"])
    if velocities is not None and len(velocities):
        summary["max_particle_speed_m_s"] = float(np.max(np.linalg.norm(velocities, axis=1)))
    return summary


def _phase_metrics(env, frames: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    cameras = _camera_runtime(env)
    robots = _robot_runtime(env)
    cloth = _cloth_runtime(env)
    cloth_points = [item["points_world"] for item in cloth if item["points_world"] is not None]
    combined_points = np.concatenate(cloth_points, axis=0) if cloth_points else None
    table_height = _table_height_world(env)
    cloth_centroid = None if combined_points is None or not len(combined_points) else np.mean(combined_points, axis=0)
    for robot in robots:
        pose = _as_numpy(robot["end_effector_pose_world"])
        tool_position = None if pose is None or pose.size < 3 else pose.reshape(-1)[:3]
        geometry = {
            "table_clearance_m": None,
            "cloth_centroid_distance_m": None,
            "nearest_cloth_particle_distance_m": None,
        }
        if tool_position is not None and table_height is not None:
            geometry["table_clearance_m"] = float(tool_position[2] - table_height)
        if tool_position is not None and cloth_centroid is not None:
            geometry["cloth_centroid_distance_m"] = float(np.linalg.norm(tool_position - cloth_centroid))
        if tool_position is not None and combined_points is not None and len(combined_points):
            geometry["nearest_cloth_particle_distance_m"] = float(
                np.min(np.linalg.norm(combined_points - tool_position, axis=1))
            )
        robot["end_effector_geometry"] = geometry
    camera_metrics = {}
    for camera_name, camera in cameras.items():
        projection = None
        unavailable_reason = None
        if combined_points is None:
            unavailable_reason = "no runtime cloth particles were available"
        elif camera["projection_model"] != "pinhole":
            unavailable_reason = f"projection model {camera['projection_model']!r} is not supported by this audit"
        else:
            projection = project_points_to_camera(
                combined_points,
                camera["camera_to_world"],
                camera["intrinsic_matrix"],
                camera["resolution"],
            )
        table_hit = forward_ray_plane_intersection(camera["camera_to_world"], table_height)
        table_hit["cloth_centroid_xy_distance_m"] = None
        if table_hit["point_world_m"] is not None and cloth_centroid is not None:
            table_hit["cloth_centroid_xy_distance_m"] = float(
                np.linalg.norm(np.asarray(table_hit["point_world_m"][:2]) - cloth_centroid[:2])
            )
        camera_metrics[camera_name] = {
            "role": camera["role"],
            "image_shape": list(frames[camera_name].shape),
            "position_world_m": camera["camera_to_world"][:3, 3].tolist(),
            "axes": camera_axes(camera["camera_to_world"]),
            "intrinsic_matrix": camera["intrinsic_matrix"].tolist(),
            "forward_ray_table_intersection": table_hit,
            "cloth_projection": projection,
            "cloth_projection_unavailable_reason": unavailable_reason,
        }
    phase = {
        "simulation_time_s": _simulation_time(env),
        "cameras": camera_metrics,
        "robots": robots,
        "cloth": [_cloth_summary(item) for item in cloth],
    }
    runtime = {"robots": robots, "cloth": cloth, "cameras": cameras}
    return phase, runtime


def _robot_drift(reset: list[dict[str, Any]], held: list[dict[str, Any]]) -> dict[str, Any]:
    held_by_name = {robot["arm_name"]: robot for robot in held}
    rows = []
    for before in reset:
        after = held_by_name.get(before["arm_name"])
        if after is None:
            rows.append({"arm_name": before["arm_name"], "unavailable_reason": "arm missing after hold"})
            continue
        arm_velocity = _as_numpy(after["arm_joint_velocities"])
        finger_velocity = _as_numpy(after["finger_joint_velocities"])
        all_velocity = [value for value in (arm_velocity, finger_velocity) if value is not None]
        max_velocity = None
        if all_velocity:
            max_velocity = float(np.max(np.abs(np.concatenate(all_velocity))))
        rows.append(
            {
                "arm_name": before["arm_name"],
                "arm_joint_position": vector_drift(before["arm_joint_positions"], after["arm_joint_positions"]),
                "finger_joint_position": vector_drift(
                    before["finger_joint_positions"], after["finger_joint_positions"]
                ),
                "arm_position_target": vector_drift(before["arm_position_targets"], after["arm_position_targets"]),
                "finger_position_target": vector_drift(
                    before["finger_position_targets"], after["finger_position_targets"]
                ),
                "held_max_abs_joint_velocity": max_velocity,
                "unavailable_reason": None,
            }
        )
    return {"arms": rows}


def _cloth_drift(reset: list[dict[str, Any]], held: list[dict[str, Any]]) -> dict[str, Any]:
    held_by_name = {cloth["name"]: cloth for cloth in held}
    rows = []
    for before in reset:
        after = held_by_name.get(before["name"])
        points_before = before["points_world"]
        points_after = None if after is None else after["points_world"]
        displacement = {"max_m": None, "rms_m": None, "centroid_m": None}
        reason = None
        if after is None:
            reason = "garment missing after hold"
        elif points_before is None or points_after is None:
            reason = "runtime cloth particles unavailable"
        elif points_before.shape != points_after.shape:
            reason = f"particle shape changed from {points_before.shape} to {points_after.shape}"
        elif len(points_before):
            deltas = points_after - points_before
            distances = np.linalg.norm(deltas, axis=1)
            displacement = {
                "max_m": float(np.max(distances)),
                "rms_m": float(np.sqrt(np.mean(np.square(distances)))),
                "centroid_m": float(np.linalg.norm(np.mean(points_after, axis=0) - np.mean(points_before, axis=0))),
            }
        rows.append(
            {
                "name": before["name"],
                "particle_displacement": displacement,
                "held_support": None if after is None else _cloth_summary(after)["support"],
                "unavailable_reason": reason,
            }
        )
    return {"garments": rows}


def _save_frames(output: Path, phase: str, frames: dict[str, np.ndarray]) -> list[Path]:
    paths = []
    for camera_name, frame in frames.items():
        path = output / f"{phase}_{camera_name}.png"
        Image.fromarray(frame, mode="RGB").save(path)
        paths.append(path)
    return paths


def _save_contact_sheet(output: Path, phase: str, frames: dict[str, np.ndarray]) -> Path:
    label_height = 28
    images = []
    for name, frame in frames.items():
        image = Image.fromarray(frame, mode="RGB")
        labeled = Image.new("RGB", (image.width, image.height + label_height), "#111318")
        labeled.paste(image, (0, label_height))
        ImageDraw.Draw(labeled).text((8, 7), f"{phase}: {name}", fill="white")
        images.append(labeled)
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    sheet = Image.new("RGB", (width, height), "black")
    offset = 0
    for image in images:
        sheet.paste(image, (offset, 0))
        offset += image.width
    path = output / f"{phase}_contact_sheet.png"
    sheet.save(path)
    return path


def _completed_audit_matches(output: Path, scene_manifest_sha256: str) -> bool:
    try:
        metadata = json.loads((output / METADATA_NAME).read_text(encoding="utf-8"))
        metrics = json.loads((output / METRICS_NAME).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return False
    return bool(metadata.get("complete") and metrics.get("complete")) and (
        metadata.get("scene_export", {}).get("manifest_sha256") == scene_manifest_sha256
    )


def run_scene_visual_audit(env, scene_export_dir: str | os.PathLike[str], layout_id: int) -> Path:
    """Capture reset/held RGB and metrics without invoking a policy or observation path."""
    scene_export = Path(scene_export_dir).expanduser().resolve()
    scene_manifest = scene_export / "scene_manifest.json"
    if not scene_manifest.is_file():
        raise FileNotFoundError(f"scene visual audit requires a completed scene manifest: {scene_manifest}")
    manifest_sha256 = _sha256_file(scene_manifest)
    output = Path(env.save_dir).expanduser().resolve() / AUDIT_DIRECTORY_NAME
    if output.exists():
        if _completed_audit_matches(output, manifest_sha256):
            logger.info("[scene-visual-audit] reusing completed audit: %s", output)
            return output
        raise FileExistsError(f"visual audit directory already exists and does not match this scene: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".visual_audit.tmp-", dir=output.parent))
    try:
        reset_frames = _capture_rgb_frames(env)
        reset_metrics, reset_runtime = _phase_metrics(env, reset_frames)
        physics_dt, decimation, step_seconds = _simulation_step_seconds(env)
        hold_steps = exact_simulation_steps(AUDIT_DURATION_SECONDS, step_seconds)
        time_before = _simulation_time(env)
        for _ in range(hold_steps):
            # Articulation targets set during reset remain active. Deliberately
            # avoid RobotManager.control_robot and every policy-facing path.
            env.sim_step(render=False)
        time_after = _simulation_time(env)
        held_frames = _capture_rgb_frames(env)
        held_metrics, held_runtime = _phase_metrics(env, held_frames)

        artifact_paths = _save_frames(temporary, "reset", reset_frames)
        artifact_paths.extend(_save_frames(temporary, "held", held_frames))
        artifact_paths.append(_save_contact_sheet(temporary, "reset", reset_frames))
        artifact_paths.append(_save_contact_sheet(temporary, "held", held_frames))

        measured_duration = None
        if time_before is not None and time_after is not None:
            measured_duration = time_after - time_before
        metrics = {
            "format_version": 1,
            "complete": True,
            "hold": {
                "control_source": "unchanged_post_reset_articulation_targets",
                "policy_calls": 0,
                "requested_duration_s": AUDIT_DURATION_SECONDS,
                "physics_dt_s": physics_dt,
                "decimation": decimation,
                "sim_step_duration_s": step_seconds,
                "sim_steps": hold_steps,
                "expected_duration_s": hold_steps * step_seconds,
                "measured_duration_s": measured_duration,
            },
            "phases": {"reset": reset_metrics, "held": held_metrics},
            "drift": {
                "robots": _robot_drift(reset_runtime["robots"], held_runtime["robots"]),
                "cloth": _cloth_drift(reset_runtime["cloth"], held_runtime["cloth"]),
            },
            "measurement_notes": {
                "camera_convention": "USD: local -Z forward, +Y up, +X right",
                "cloth_visibility": "Pinhole projection of runtime cloth particles into the stream resolution.",
                "cloth_support": (
                    "Geometric proximity to the configured table surface; direct cloth contact counts are unavailable."
                ),
            },
        }
        (temporary / METRICS_NAME).write_text(json.dumps(_json_value(metrics), indent=2) + "\n", encoding="utf-8")

        repo_root = Path(__file__).resolve().parents[4]
        revision, dirty = _git_revision(repo_root)
        scene_payload = json.loads(scene_manifest.read_text(encoding="utf-8"))
        metadata = {
            "format_version": 1,
            "complete": True,
            "created_at": datetime.now(UTC).isoformat(),
            "command": {"argv": list(sys.argv), "shell": shlex.join(sys.argv), "cwd": str(Path.cwd())},
            "repository": {"root": str(repo_root), "revision": revision, "dirty": dirty},
            "run": {
                "run_id": str(getattr(env, "run_id", os.environ.get("ROBODOJO_RUN_ID", ""))),
                "run_directory": str(Path(env.save_dir).resolve()),
                "task": str(env.task_name),
                "environment": str(env.environment),
                "scene": str(env.scene),
                "policy_name": str(env.eval_cfg.get("policy_name", "")),
                "checkpoint_label": os.environ.get("ROBODOJO_CKPT_LABEL"),
                "seed": int(env.eval_seed),
                "layout_id": int(layout_id),
                "camera_order": list(env.camera_manager.camera_names[0]),
                "config": _json_value(env.eval_cfg.get("config", {})),
                "config_sha256": scene_payload.get("config_sha256"),
            },
            "scene_export": {
                "directory": str(scene_export),
                "manifest": str(scene_manifest),
                "manifest_sha256": manifest_sha256,
                "snapshot_boundary": scene_payload.get("snapshot_boundary"),
            },
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "ROBODOJO_SCENE_VISUAL_AUDIT",
                    "ROBODOJO_EXPORT_SCENE",
                    "ROBODOJO_EXPORT_SCENE_ONLY",
                    "ROBODOJO_EXPORT_SCENE_DIR",
                    "ROBODOJO_EXPORT_LAYOUT_ID",
                    "ROBODOJO_RUN_ID",
                    "CUDA_VISIBLE_DEVICES",
                )
            },
            "artifacts": {
                path.name: {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
                for path in artifact_paths + [temporary / METRICS_NAME]
            },
        }
        (temporary / METADATA_NAME).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output)
        logger.info("[scene-visual-audit] wrote reset/held RGB, contact sheets, and metrics to %s", output)
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
