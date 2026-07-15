"""Prepare the narrow simulator assets used by the public MolmoAct2 YAM profile."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from robodojo.core.storage import assets_root

logger = logging.getLogger(__name__)

SOURCE_GARMENT_INDEX = 9
DERIVED_GARMENT_INDEX = 12


def reshape_long_sleeves_for_molmoact2(points: np.ndarray) -> np.ndarray:
    """Return a topology-preserving short-sleeve variant of a garment mesh."""
    result = np.asarray(points, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"garment points must have shape (N, 3), got {result.shape}")
    distance = np.abs(result[:, 0])
    sleeve_weight = np.clip((distance - 0.10) / 0.06, 0.0, 1.0)
    sleeve_sign = np.where(result[:, 0] < 0.0, -1.0, 1.0)
    target_x = sleeve_sign * (0.10 + (distance - 0.10) * 0.43)
    target_y = 0.109 + (result[:, 1] - 0.109) * 0.5
    result[:, 0] += sleeve_weight * (target_x - result[:, 0])
    result[:, 1] += sleeve_weight * (target_y - result[:, 1])
    return result


def update_garment_metadata(metadata: dict, points: np.ndarray, face_count: int) -> dict:
    """Update source geometry facts while preserving its functional landmarks."""
    if face_count <= 0:
        raise ValueError("garment mesh must contain at least one face")
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"garment points must have shape (N, 3), got {points.shape}")

    result = json.loads(json.dumps(metadata))
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    extents = upper - lower
    bounds = [
        [x, y, z]
        for x in (float(lower[0]), float(upper[0]))
        for y in (float(lower[1]), float(upper[1]))
        for z in (float(lower[2]), float(upper[2]))
    ]
    geometry = result.setdefault("geometry", {})
    geometry.update(
        {
            "faces": int(face_count),
            "vertices": int(len(points)),
            "aligned_bbox": {"vertices": bounds, "extents": extents.astype(float).tolist()},
            "oriented_bbox": {"vertices": bounds, "extents": extents.astype(float).tolist()},
            "radius": float(extents.max() * 0.5),
        }
    )
    return result


def prepare_molmoact2_yam_garment() -> Path:
    """Derive the policy shirt from the downloaded upstream garment asset."""
    from pxr import Usd, UsdGeom, Vt

    garment_root = assets_root() / "Object" / "RoboDojo" / "Garment" / "Top_Long"
    source_root = garment_root / f"{SOURCE_GARMENT_INDEX:05d}"
    source = source_root / "object.usdz"
    source_metadata = source_root / "metadata.json"
    if not source.is_file() or not source_metadata.is_file():
        raise FileNotFoundError(
            f"MolmoAct2 garment source is incomplete under {source_root}; download RoboDojo assets first"
        )

    destination = garment_root / f"{DERIVED_GARMENT_INDEX:05d}" / "object.usd"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.usd")
    temporary.unlink(missing_ok=True)
    source_stage = Usd.Stage.Open(str(source))
    if source_stage is None:
        raise RuntimeError(f"could not open MolmoAct2 garment source {source}")
    if not source_stage.Flatten().Export(str(temporary)):
        raise RuntimeError(f"could not export flattened MolmoAct2 garment to {temporary}")

    stage = Usd.Stage.Open(str(temporary))
    if stage is None:
        raise RuntimeError(f"could not open flattened MolmoAct2 garment {temporary}")
    meshes = [UsdGeom.Mesh(prim) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise RuntimeError(f"expected one garment mesh in {source}, found {len(meshes)}")
    mesh = meshes[0]
    points = reshape_long_sleeves_for_molmoact2(np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32))
    usd_points = Vt.Vec3fArray.FromNumpy(points)
    mesh.GetPointsAttr().Set(usd_points)
    mesh.GetExtentAttr().Set(UsdGeom.PointBased.ComputeExtent(usd_points))
    face_count = len(mesh.GetFaceVertexCountsAttr().Get())
    stage.GetRootLayer().Save()
    temporary.replace(destination)

    metadata = destination.with_name("metadata.json")
    metadata_temporary = metadata.with_suffix(".tmp.json")
    with source_metadata.open(encoding="utf-8") as stream:
        source_metadata_data = json.load(stream)
    derived_metadata = update_garment_metadata(source_metadata_data, points, face_count)
    with metadata_temporary.open("w", encoding="utf-8") as stream:
        json.dump(derived_metadata, stream, indent=2)
        stream.write("\n")
    metadata_temporary.replace(metadata)
    logger.info("prepared inherited MolmoAct2 YAM garment %s from %s", destination, source)
    return destination
