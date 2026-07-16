"""Export the composed, post-reset Isaac stage without editing the live stage."""

from __future__ import annotations

from collections.abc import Mapping
import json
import logging
from typing import Any

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

from robodojo.sim.scene_export.contracts import (
    calculate_fisheye_fov_degrees,
    calculate_fov_degrees,
)
from robodojo.sim.scene_export.services.bundling import _sha256_bytes

logger = logging.getLogger(__name__)

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
    holder_paths = env.camera_manager.mount_hardware_paths[0]
    holder_id = 0
    for cam_id, camera_name in enumerate(env.camera_manager.camera_names[0]):
        spec = rig_by_key[camera_name]
        runtime = env.camera_manager.camera_config[camera_name].camera
        xform_path = env.camera_manager.cameras_xform_path[0][cam_id]
        has_holder = runtime.get("mount_hardware_enabled", True) and runtime.get("mount_hardware_asset")
        holder_path = holder_paths[holder_id] if has_holder and holder_id < len(holder_paths) else None
        holder_id += int(bool(has_holder))
        sensor_path = str(env.camera_manager.cameras[0][cam_id].prim_path)
        sensor_prim = stage.GetPrimAtPath(sensor_path)
        usd_camera = UsdGeom.Camera(sensor_prim) if sensor_prim and sensor_prim.IsA(UsdGeom.Camera) else None
        width, height = [int(value) for value in spec.sensor["stream_resolution"]]
        fx = float(spec.projection.get("fx", env.camera_manager.get_camera_intrinsics(cam_id, 0)[0, 0]))
        fy = float(spec.projection.get("fy", fx))
        cx = float(spec.projection.get("cx", width / 2.0))
        cy = float(spec.projection.get("cy", height / 2.0))
        coefficients = _json_value(spec.projection.get("distortion_coefficients", []))
        if spec.projection.get("model") == "opencvFisheye":
            effective_fov = calculate_fisheye_fov_degrees(width, height, fx, fy, coefficients)
        else:
            effective_fov = calculate_fov_degrees(width, height, fx, fy)
        published_diagonal_fov = float(spec.sensor["diagonal_fov_deg"])
        fitted_diagonal_fov = float(spec.projection.get("fitted_diagonal_fov_deg", published_diagonal_fov))
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
                "effective_fov_degrees": effective_fov,
                "published_diagonal_fov_degrees": published_diagonal_fov,
                "fitted_diagonal_fov_degrees": fitted_diagonal_fov,
                "diagonal_fov_error_degrees": effective_fov["diagonal"] - fitted_diagonal_fov,
                "projection_model": spec.projection.get("model", "pinhole"),
                "projection_backend": spec.projection.get("backend", "native"),
                "distortion_coefficients": coefficients,
                "zero_distortion_postprocess": bool(
                    spec.projection.get("backend") == "pinhole_postprocess" and coefficients and not any(coefficients)
                ),
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
            bound_material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial() if prim else (None, None)
            bound_material_path = str(bound_material.GetPath()) if bound_material else None
            configured_material = getattr(garment, "visual_usd_path", None)
            expected_root = getattr(garment, "visual_material_path", None)
            if configured_material:
                surface_output = bound_material.GetSurfaceOutput() if bound_material else None
                surface_source = surface_output.GetConnectedSource() if surface_output else None
                matches_override = bool(
                    bound_material and expected_root and bound_material.GetPath().HasPrefix(Sdf.Path(expected_root))
                )
                if not matches_override or not surface_source:
                    raise RuntimeError(
                        "garment visual material override did not compose and bind: "
                        f"garment={name} configured={configured_material} bound={bound_material_path}"
                    )
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
                    "visual_material": {
                        "configured_asset": configured_material,
                        "composition_root": expected_root,
                        "bound_material_path": bound_material_path,
                    },
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
        # TraverseAll includes inactive opinions, but flattened USD omits them.
        # Inventory only live scene physics so reopened-stage validation uses
        # the same composition semantics as the exported artifact.
        if not prim.IsActive():
            continue
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


def _stage_layer_stack(stage: Usd.Stage) -> list[Sdf.Layer]:
    """Return the live root/session sublayer stack without the ABI-fragile USD binding."""
    layers = []
    seen = set()

    def visit(layer: Sdf.Layer | None) -> None:
        if layer is None or layer.identifier in seen:
            return
        seen.add(layer.identifier)
        layers.append(layer)
        for sublayer_path in layer.subLayerPaths:
            visit(Sdf.Layer.FindRelativeToLayer(layer, sublayer_path))

    visit(stage.GetSessionLayer())
    visit(stage.GetRootLayer())
    return layers


def _live_guard(env, stage: Usd.Stage) -> dict[str, Any]:
    layer_state = []
    for layer in _stage_layer_stack(stage):
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
