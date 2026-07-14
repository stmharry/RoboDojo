"""Visual-only robot calibration that never mutates collision or rigid bodies."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def correction_matrix(correction) -> np.ndarray:
    correction = np.asarray(correction, dtype=np.float64)
    if correction.shape != (6,) or not np.all(np.isfinite(correction)):
        raise ValueError("visual calibration correction must be a finite 6-D vector")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_rotvec(np.deg2rad(correction[3:])).as_matrix()
    matrix[:3, 3] = correction[:3]
    return matrix


def visual_only_local_matrix(
    tip_world: np.ndarray,
    calibration_frame_world: np.ndarray,
    original_visual_local: np.ndarray,
    correction,
) -> np.ndarray:
    """Conjugate a frame-space correction into a visual prim's local frame."""
    tip_world = np.asarray(tip_world, dtype=np.float64)
    frame_world = np.asarray(calibration_frame_world, dtype=np.float64)
    visual_local = np.asarray(original_visual_local, dtype=np.float64)
    if any(matrix.shape != (4, 4) for matrix in (tip_world, frame_world, visual_local)):
        raise ValueError("visual calibration transforms must be 4x4 matrices")
    visual_world = tip_world @ visual_local
    desired_world = frame_world @ correction_matrix(correction) @ np.linalg.inv(frame_world) @ visual_world
    return np.linalg.inv(tip_world) @ desired_world


def validate_visual_calibration(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict) or config.get("enabled") is not True:
        raise ValueError("visual calibration must be an enabled mapping")
    if config.get("frame") != "gripper/molmo_link6":
        raise ValueError("YAM visual calibration must use gripper/molmo_link6")
    visuals = config.get("visuals")
    if not isinstance(visuals, dict) or set(visuals) != {"tip_left/visuals", "tip_right/visuals"}:
        raise ValueError("YAM visual calibration must target exactly both jaw visual prims")
    for correction in visuals.values():
        correction_matrix(correction)
    return config


def plan_visual_calibration_matrices(
    config: dict[str, Any],
    calibration_frame_world: np.ndarray,
    visual_contexts: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Return updates for the declared visual roots and no other prims."""
    config = validate_visual_calibration(config)
    if set(visual_contexts) != set(config["visuals"]):
        raise ValueError("visual calibration contexts must exactly match the declared visual roots")
    return {
        path: visual_only_local_matrix(
            visual_contexts[path][0],
            calibration_frame_world,
            visual_contexts[path][1],
            correction,
        )
        for path, correction in config["visuals"].items()
    }


def apply_visual_calibration(stage, robot_root: str, config: dict[str, Any], cache: dict) -> None:
    """Author calibrated local matrices on visual roots, leaving physics untouched."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    config = validate_visual_calibration(config)
    frame_path = f"{robot_root}/{config['frame']}"
    frame_prim = stage.GetPrimAtPath(frame_path)
    if not frame_prim.IsValid():
        raise ValueError(f"visual calibration frame is absent: {frame_path}; rebuild the YAM asset")
    frame_world = np.asarray(
        UsdGeom.Xformable(frame_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
        dtype=np.float64,
    ).T
    # Validate the entire target set before authoring any transform. A bad or
    # physics-bearing second target therefore cannot leave the first mutated.
    targets = {}
    visual_contexts = {}
    for relative_path in config["visuals"]:
        visual_path = f"{robot_root}/{relative_path}"
        visual_prim = stage.GetPrimAtPath(visual_path)
        if not visual_prim.IsValid():
            raise ValueError(f"visual calibration prim is absent: {visual_path}; rebuild the YAM asset")
        for prim in Usd.PrimRange(visual_prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError(f"visual-only calibration refuses physics prim {prim.GetPath()}")
        tip_prim = visual_prim.GetParent()
        tip_world = np.asarray(
            UsdGeom.Xformable(tip_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
            dtype=np.float64,
        ).T
        xformable = UsdGeom.Xformable(visual_prim)
        cache_key = (robot_root, relative_path)
        if cache_key not in cache:
            cache[cache_key] = np.asarray(
                xformable.GetLocalTransformation(Usd.TimeCode.Default()),
                dtype=np.float64,
            ).T
        targets[relative_path] = xformable
        visual_contexts[relative_path] = (tip_world, cache[cache_key])
    calibrated_matrices = plan_visual_calibration_matrices(config, frame_world, visual_contexts)
    for relative_path, calibrated in calibrated_matrices.items():
        xformable = targets[relative_path]
        authored = Gf.Matrix4d(1.0)
        row_matrix = calibrated.T
        for row in range(4):
            authored.SetRow(row, Gf.Vec4d(*[float(value) for value in row_matrix[row]]))
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(authored)
