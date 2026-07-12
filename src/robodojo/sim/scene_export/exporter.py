"""Export the composed, post-reset Isaac stage without editing the live stage."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
from pxr import Ar, Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdUtils, Vt

from robodojo.sim.environment.global_configs import ROOT_DIR
from robodojo.sim.scene_export.contracts import ExportIdentity, calculate_fov_degrees, completed_export_matches

USD_EXTENSIONS = frozenset({".usd", ".usda", ".usdc", ".usdz"})
MANIFEST_NAME = "scene_manifest.json"
REFERENCED_NAME = "scene_referenced.usda"
FLATTENED_NAME = "scene_flattened.usdc"


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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_list(matrix: Gf.Matrix4d) -> list[list[float]]:
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def _local_and_world(stage: Usd.Stage, prim_path: str) -> dict[str, Any]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return {"local_matrix": None, "world_matrix": None}
    xform = UsdGeom.Xformable(prim)
    if not xform:
        return {"local_matrix": None, "world_matrix": None}
    local = xform.GetLocalTransformation(Usd.TimeCode.Default())
    world = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    transform = Gf.Transform(world)
    quat = transform.GetRotation().GetQuat()
    return {
        "local_matrix": _matrix_list(local),
        "world_matrix": _matrix_list(world),
        "world_translation": [float(value) for value in transform.GetTranslation()],
        "world_orientation_wxyz": [
            float(quat.GetReal()),
            *[float(value) for value in quat.GetImaginary()],
        ],
    }


def _camera_state(env, stage: Usd.Stage) -> list[dict[str, Any]]:
    cameras = []
    rig_by_key = {camera.observation_key: camera for camera in env.camera_rig.cameras}
    for cam_id, camera_name in enumerate(env.camera_manager.camera_names[0]):
        spec = rig_by_key[camera_name]
        runtime = env.camera_manager.camera_config[camera_name].camera
        xform_path = env.camera_manager.cameras_xform_path[0][cam_id]
        holder_paths = env.camera_manager.mount_hardware_paths[0]
        holder_path = holder_paths[cam_id] if cam_id < len(holder_paths) else None
        sensor_path = str(env.camera_manager.cameras[0][cam_id].prim_path)
        sensor_prim = stage.GetPrimAtPath(sensor_path)
        usd_camera = UsdGeom.Camera(sensor_prim) if sensor_prim and sensor_prim.IsA(UsdGeom.Camera) else None
        width, height = [int(value) for value in spec.sensor["stream_resolution"]]
        fx = float(spec.projection.get("fx", env.camera_manager.get_camera_intrinsics(cam_id, 0)[0, 0]))
        fy = float(spec.projection.get("fy", fx))
        cx = float(spec.projection.get("cx", width / 2.0))
        cy = float(spec.projection.get("cy", height / 2.0))
        parent = stage.GetPrimAtPath(xform_path).GetParent()
        backing = {}
        if usd_camera:
            backing = {
                "focal_length": _json_value(usd_camera.GetFocalLengthAttr().Get()),
                "horizontal_aperture": _json_value(usd_camera.GetHorizontalApertureAttr().Get()),
                "vertical_aperture": _json_value(usd_camera.GetVerticalApertureAttr().Get()),
                "clipping_range": _json_value(usd_camera.GetClippingRangeAttr().Get()),
                "projection": _json_value(usd_camera.GetProjectionAttr().Get()),
            }
        cameras.append(
            {
                "observation_key": camera_name,
                "role": spec.role,
                "mount_kind": spec.mount["kind"],
                "mount_target": spec.mount.get("target"),
                "resolved_parent_path": str(parent.GetPath()),
                "holder_path": holder_path,
                "holder_transform": _local_and_world(stage, holder_path) if holder_path else None,
                "xform_path": xform_path,
                "xform_transform": _local_and_world(stage, xform_path),
                "sensor_path": sensor_path,
                "sensor_transform": _local_and_world(stage, sensor_path),
                "native_resolution": [int(value) for value in spec.sensor.get("native_resolution", [width, height])],
                "stream_resolution": [width, height],
                "effective_intrinsic_matrix": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                "effective_fov_degrees": calculate_fov_degrees(width, height, fx, fy),
                "published_diagonal_fov_degrees": float(spec.sensor["diagonal_fov_deg"]),
                "projection_model": spec.projection.get("model", "pinhole"),
                "projection_backend": spec.projection.get("backend", "native"),
                "distortion_coefficients": _json_value(spec.projection.get("distortion_coefficients", [])),
                "backing_usd_camera": backing,
                "runtime_camera": _json_value(runtime),
            }
        )
    return cameras


def _robot_state(env) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for key in env.robot_manager.robot_key:
        if id(key) in seen:
            continue
        seen.add(id(key))
        root_paths = _json_value(getattr(getattr(key, "root_physx_view", None), "prim_paths", []))
        result.append(
            {
                "root_paths": root_paths,
                "joint_names": list(key.joint_names),
                "body_names": list(key.body_names),
                "joint_positions": _json_value(key.data.joint_pos),
                "joint_velocities": _json_value(key.data.joint_vel),
                "root_state_world": _json_value(key.data.root_state_w),
                "body_link_poses_world": _json_value(key.data.body_link_pose_w),
                "body_link_velocities_world": _json_value(key.data.body_link_vel_w),
            }
        )
    return result


def _cloth_state(env, stage: Usd.Stage) -> list[dict[str, Any]]:
    result = []
    for env_objects in env.scene_manager._garment_objects:
        for name, garment in env_objects.items():
            prim = stage.GetPrimAtPath(garment.mesh_prim_path)
            points = prim.GetAttribute("points").Get() if prim else None
            velocities = None
            try:
                velocities = garment._cloth_prim_view.get_velocities()
            except Exception:
                pass
            result.append(
                {
                    "name": name,
                    "prim_path": garment.usd_prim_path,
                    "mesh_prim_path": garment.mesh_prim_path,
                    "particle_system_path": garment.particle_system_path,
                    "particle_material_path": garment.particle_material_path,
                    "points_local": _json_value(points if points is not None else []),
                    "velocities_world": _json_value(velocities),
                    "physics": _json_value(garment.physics_cfg),
                }
            )
    return result


def _physics_inventory(stage: Usd.Stage) -> dict[str, list[str]]:
    inventory: dict[str, list[str]] = {
        "articulation_roots": [],
        "rigid_bodies": [],
        "collisions": [],
        "cloth": [],
        "particle_systems": [],
        "lights": [],
        "materials": [],
        "cameras": [],
    }
    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            inventory["articulation_roots"].append(path)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            inventory["rigid_bodies"].append(path)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            inventory["collisions"].append(path)
        if prim.HasAPI(PhysxSchema.PhysxParticleClothAPI):
            inventory["cloth"].append(path)
        if prim.IsA(PhysxSchema.PhysxParticleSystem):
            inventory["particle_systems"].append(path)
        if prim.HasAPI(UsdLux.LightAPI):
            inventory["lights"].append(path)
        if prim.IsA(UsdGeom.Camera):
            inventory["cameras"].append(path)
        if prim.GetTypeName() in {"Material", "Shader"}:
            inventory["materials"].append(path)
    return inventory


def _capture_state(env, stage: Usd.Stage) -> dict[str, Any]:
    return {
        "cameras": _camera_state(env, stage),
        "robots": _robot_state(env),
        "cloth": _cloth_state(env, stage),
        "inventory": _physics_inventory(stage),
    }


def _live_guard(env, stage: Usd.Stage) -> dict[str, Any]:
    layer_state = []
    for layer in stage.GetLayerStack(includeSessionLayers=True):
        try:
            content = layer.ExportToString().encode("utf-8")
        except Exception:
            content = layer.identifier.encode("utf-8")
        layer_state.append(
            {
                "identifier": layer.identifier,
                "dirty": bool(layer.dirty),
                "sha256": _sha256_bytes(content),
            }
        )
    state = _capture_state(env, stage)
    runtime_fingerprint = _sha256_bytes(json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    timeline_time = None
    try:
        import omni.timeline

        timeline_time = float(omni.timeline.get_timeline_interface().get_current_time())
    except Exception:
        pass
    simulation_time = None
    for owner in (getattr(env, "sim", None), getattr(getattr(env, "sim", None), "sim", None)):
        for name in ("current_time", "_current_time"):
            if owner is not None and hasattr(owner, name):
                try:
                    simulation_time = float(getattr(owner, name))
                except (TypeError, ValueError):
                    pass
                break
    return {
        "layers": layer_state,
        "timeline_time": timeline_time,
        "simulation_time": simulation_time,
        "runtime_fingerprint": runtime_fingerprint,
        "state": state,
    }


def _apply_snapshot_state(stage: Usd.Stage, state: dict[str, Any]) -> None:
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(0.0)
    world = stage.OverridePrim("/World")
    world.CreateAttribute("robodojo:snapshotTimeCode", Sdf.ValueTypeNames.Double, custom=True).Set(0.0)
    world.CreateAttribute("robodojo:snapshotKind", Sdf.ValueTypeNames.String, custom=True).Set("post_reset_pre_rollout")

    for robot_index, robot in enumerate(state["robots"]):
        root_paths = robot.get("root_paths") or []
        root_path = root_paths[0] if root_paths else f"/World/RoboDojoSnapshot/robot_{robot_index}"
        prim = stage.OverridePrim(root_path)
        prim.CreateAttribute("robodojo:jointNames", Sdf.ValueTypeNames.StringArray, custom=True).Set(
            Vt.StringArray(robot["joint_names"])
        )
        joint_positions = np.asarray(robot["joint_positions"], dtype=np.float64).reshape(-1).tolist()
        joint_velocities = np.asarray(robot["joint_velocities"], dtype=np.float64).reshape(-1).tolist()
        prim.CreateAttribute("robodojo:jointPositions", Sdf.ValueTypeNames.DoubleArray, custom=True).Set(
            Vt.DoubleArray(joint_positions)
        )
        prim.CreateAttribute("robodojo:jointVelocities", Sdf.ValueTypeNames.DoubleArray, custom=True).Set(
            Vt.DoubleArray(joint_velocities)
        )

    for cloth in state["cloth"]:
        prim = stage.GetPrimAtPath(cloth["mesh_prim_path"])
        if not prim or not prim.IsValid():
            continue
        points = cloth.get("points_local") or []
        if points:
            UsdGeom.Mesh(prim).GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(points, dtype=np.float32)))
        velocities = cloth.get("velocities_world")
        if velocities is not None:
            velocity_array = np.asarray(velocities, dtype=np.float32).reshape(-1, 3)
            if len(velocity_array) == len(points):
                UsdGeom.PointBased(prim).GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(velocity_array))


def _resolve_asset(path: str, repo_root: Path) -> Path | None:
    if not path or path.startswith("anon:") or "://" in path:
        return None
    candidate = Path(path)
    if candidate.is_absolute() and candidate.is_file():
        return candidate.resolve()
    try:
        resolved = str(Ar.GetResolver().Resolve(path))
    except Exception:
        resolved = ""
    if resolved and Path(resolved).is_file():
        return Path(resolved).resolve()
    for base in (repo_root, Path.cwd()):
        candidate = base / path
        if candidate.is_file():
            return candidate.resolve()
    return None


def _split_package_path(path: str) -> tuple[str, str]:
    """Split `asset.usdz[member]` while leaving ordinary paths unchanged."""
    marker = path.find("[")
    if marker > 0 and path.endswith("]"):
        return path[:marker], path[marker:]
    return path, ""


def _modify_asset_paths(layer: Sdf.Layer, callback) -> None:
    modifier = getattr(UsdUtils, "ModifyAssetPaths", None)
    if modifier is None:
        return
    modifier(layer, callback)


def _make_referenced_paths_portable(layer: Sdf.Layer, output_dir: Path, repo_root: Path) -> list[dict[str, Any]]:
    dependencies: dict[str, dict[str, Any]] = {}

    def rewrite(path: str) -> str:
        resolved = _resolve_asset(path, repo_root)
        if resolved is None:
            if path:
                dependencies.setdefault(path, {"authored_path": path, "status": "unresolved"})
            return path
        relative = os.path.relpath(resolved, output_dir)
        dependencies[str(resolved)] = {
            "authored_path": path,
            "resolved_path": str(resolved),
            "export_path": relative,
            "sha256": _sha256_file(resolved),
            "status": "external",
        }
        return relative

    _modify_asset_paths(layer, rewrite)
    return sorted(dependencies.values(), key=lambda item: item.get("resolved_path", item["authored_path"]))


def _bundle_flattened_assets(
    layer: Sdf.Layer, output_dir: Path, repo_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dependency_dir = output_dir / "dependencies"
    bundled: dict[str, dict[str, Any]] = {}
    unresolved: dict[str, dict[str, Any]] = {}

    def rewrite(path: str) -> str:
        outer_path, package_member = _split_package_path(path)
        resolved = _resolve_asset(outer_path, repo_root)
        if resolved is None:
            if path:
                unresolved.setdefault(path, {"authored_path": path, "status": "unresolved"})
            return path
        source = str(resolved)
        if source not in bundled:
            digest = _sha256_file(resolved)
            destination_name = f"{digest[:12]}_{resolved.name}"
            destination = dependency_dir / destination_name
            dependency_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved, destination)
            bundled[source] = {
                "authored_path": path,
                "resolved_path": source,
                "bundled_path": f"dependencies/{destination_name}",
                "sha256": digest,
                "kind": "usd" if resolved.suffix.lower() in USD_EXTENSIONS else "asset",
                "status": "bundled",
            }
        return bundled[source]["bundled_path"] + package_member

    _modify_asset_paths(layer, rewrite)
    return list(bundled.values()), list(unresolved.values())


def _git_revision(repo_root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
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


def _runtime_versions() -> dict[str, str | None]:
    versions = {}
    for package in ("isaacsim", "isaaclab", "usd-core", "torch", "numpy"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _source_revisions(repo_root: Path) -> dict[str, Any]:
    result = {}
    for name, path in {
        "tracked_openarm_sources": repo_root / "configs/tooling/openarm/sources.json",
        "generated_openarm_manifest": repo_root / "Assets/Robots/openarm/manifest.json",
    }.items():
        try:
            result[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            result[name] = None
    return result


def _config_hashes(repo_root: Path, env) -> dict[str, str]:
    eval_cfg = env.eval_cfg
    paths = [
        repo_root / "configs" / f"{env.config_name}.yml",
        repo_root / "configs/camera" / f"{eval_cfg['config']['camera']}.yml",
        repo_root / "configs/scene" / f"{eval_cfg['config']['scene']}.yml",
        repo_root / "configs/robot" / f"{eval_cfg['config']['robot']}.yml",
        repo_root / "configs/sim" / f"{eval_cfg['config']['sim']}.yml",
        repo_root / "task/RoboDojo/config" / f"{env.task_name}.yml",
    ]
    return {str(path.relative_to(repo_root)): _sha256_file(path) for path in paths if path.is_file()}


def _list_op_items(value) -> list[Any]:
    if value is None:
        return []
    items = []
    for name in ("explicitItems", "prependedItems", "appendedItems", "addedItems"):
        items.extend(getattr(value, name, []) or [])
    return items


def _validate_flattened(stage: Usd.Stage) -> tuple[list[str], list[str]]:
    external_arcs = []
    internal_arcs = []
    for prim in stage.TraverseAll():
        if prim.HasAuthoredReferences():
            references = _list_op_items(prim.GetMetadata("references"))
            external = [reference.assetPath for reference in references if reference.assetPath]
            target = external_arcs if external else internal_arcs
            target.append(f"reference:{prim.GetPath()}:{external}")
        if prim.HasAuthoredPayloads():
            payloads = _list_op_items(prim.GetMetadata("payload"))
            external = [payload.assetPath for payload in payloads if payload.assetPath]
            target = external_arcs if external else internal_arcs
            target.append(f"payload:{prim.GetPath()}:{external}")
    if stage.GetRootLayer().subLayerPaths:
        external_arcs.extend(f"sublayer:{path}" for path in stage.GetRootLayer().subLayerPaths)
    return external_arcs, internal_arcs


def _validate_reopened_stage(stage: Usd.Stage, state: dict[str, Any], label: str) -> None:
    missing = []
    for paths in state["inventory"].values():
        for path in paths:
            if not stage.GetPrimAtPath(path).IsValid():
                missing.append(path)
    if missing:
        raise RuntimeError(f"{label} export is missing expected prims: {missing[:20]}")
    for camera in state["cameras"]:
        actual = _local_and_world(stage, camera["sensor_path"])["world_matrix"]
        expected = camera["sensor_transform"]["world_matrix"]
        if actual is None or not np.allclose(actual, expected, rtol=0.0, atol=1e-7):
            raise RuntimeError(f"{label} camera transform mismatch: {camera['observation_key']}")


def export_scene_snapshot(env, output_dir: str | os.PathLike[str], layout_id: int) -> Path:
    """Export one fully reset environment and return the completed directory."""
    repo_root = Path(ROOT_DIR).resolve()
    output = Path(output_dir).expanduser().resolve()
    revision, dirty = _git_revision(repo_root)
    identity = ExportIdentity(
        task=str(env.task_name),
        profile=str(env.config_name),
        seed=int(env.eval_seed),
        layout_id=int(layout_id),
        repository_revision=revision,
    )
    if output.exists():
        if completed_export_matches(output, identity):
            print(f"[scene-export] reusing completed snapshot: {output}")
            return output
        raise FileExistsError(f"scene export directory already exists and does not match this run: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        stage = env.stage
        before = _live_guard(env, stage)
        state = before.pop("state")
        referenced_layer = UsdUtils.FlattenLayerStack(stage)
        referenced_stage = Usd.Stage.Open(referenced_layer)
        _apply_snapshot_state(referenced_stage, state)
        del referenced_stage
        referenced_dependencies = _make_referenced_paths_portable(referenced_layer, temporary, repo_root)
        referenced_layer.Export(str(temporary / REFERENCED_NAME))
        reopened_referenced = Usd.Stage.Open(str(temporary / REFERENCED_NAME), load=Usd.Stage.LoadAll)
        _validate_reopened_stage(reopened_referenced, state, "referenced")

        flattened_layer = stage.Flatten()
        flattened_stage = Usd.Stage.Open(flattened_layer)
        _apply_snapshot_state(flattened_stage, state)
        bundled, unresolved = _bundle_flattened_assets(flattened_stage.GetRootLayer(), temporary, repo_root)
        flattened_stage.GetRootLayer().Export(str(temporary / FLATTENED_NAME))

        reopened_flattened = Usd.Stage.Open(str(temporary / FLATTENED_NAME), load=Usd.Stage.LoadAll)
        _validate_reopened_stage(reopened_flattened, state, "flattened")
        external_arcs, internal_arcs = _validate_flattened(reopened_flattened)
        if external_arcs:
            raise RuntimeError(f"flattened export retained external USD composition arcs: {external_arcs}")

        layout_path = Path(env.seed_manager.seed_info[int(layout_id)]["scene_layout"]).resolve()
        root_layer = stage.GetRootLayer()
        manifest = {
            "format_version": 1,
            "complete": True,
            "identity": identity.to_dict(),
            "created_at": datetime.now(UTC).isoformat(),
            "snapshot_boundary": "post_reset_pre_rollout",
            "task": env.task_name,
            "profile": {"config_name": env.config_name, "camera_profile_id": env.camera_rig.profile_id},
            "seed": int(env.eval_seed),
            "layout": {
                "id": int(layout_id),
                "path": str(layout_path),
                "sha256": _sha256_file(layout_path),
            },
            "repository": {"revision": revision, "dirty": dirty},
            "source_revisions": _source_revisions(repo_root),
            "config_sha256": _config_hashes(repo_root, env),
            "runtime_versions": _runtime_versions(),
            "stage": {
                "up_axis": UsdGeom.GetStageUpAxis(stage),
                "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
                "time_codes_per_second": stage.GetTimeCodesPerSecond(),
                "source_root_layer": root_layer.identifier,
                "source_session_layer": stage.GetSessionLayer().identifier,
                "source_layer_stack": [layer.identifier for layer in stage.GetLayerStack(includeSessionLayers=True)],
                "exported_start_time_code": 0,
                "exported_end_time_code": 0,
            },
            "simulation": {
                "dt": float(env.dt),
                "device": str(env.device),
                "use_fabric": bool(env.use_fabric),
                "timeline_time_at_capture": before["timeline_time"],
                "simulation_time_at_capture": before["simulation_time"],
            },
            "cameras": state["cameras"],
            "articulations": state["robots"],
            "cloth": state["cloth"],
            "prim_inventory": state["inventory"],
            "artifacts": {
                "referenced_usda": {
                    "path": REFERENCED_NAME,
                    "sha256": _sha256_file(temporary / REFERENCED_NAME),
                },
                "flattened_usdc": {
                    "path": FLATTENED_NAME,
                    "sha256": _sha256_file(temporary / FLATTENED_NAME),
                },
            },
            "dependencies": {
                "referenced_external": referenced_dependencies,
                "flattened_bundled": bundled,
                "unresolved": unresolved,
                "flattened_internal_arcs": internal_arcs,
            },
            "limitations": [
                "PhysX contact caches, GPU buffers, tensor handles, and solver warm-start state are not "
                "serializable to USD.",
                "The manifest is authoritative for postprocessed fisheye projection; UsdGeom.Camera stores "
                "the backing camera.",
                "Generic USD viewers may not reproduce Isaac RTX/MDL appearance exactly.",
            ],
        }
        (temporary / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        after = _live_guard(env, stage)
        after.pop("state")
        if after != before:
            raise RuntimeError("scene export mutated the live simulator stage or runtime state")
        os.replace(temporary, output)
        print(f"[scene-export] wrote referenced USDA, flattened USDC, and manifest to {output}")
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
