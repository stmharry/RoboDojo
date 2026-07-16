"""Geometry and USD authoring for the Moonlake office fixture."""

from __future__ import annotations

import hashlib
import itertools
import logging
import math
from pathlib import Path
import struct

logger = logging.getLogger(__name__)

Gf = None
Sdf = None
Usd = None
UsdGeom = None
UsdPhysics = None
UsdShade = None


def _load_pxr() -> None:
    """Load USD bindings after Isaac Sim's application kernel is available."""
    global Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    if Usd is not None:
        return
    from pxr import (
        Gf as _Gf,
        Sdf as _Sdf,
        Usd as _Usd,
        UsdGeom as _UsdGeom,
        UsdPhysics as _UsdPhysics,
        UsdShade as _UsdShade,
    )

    Gf = _Gf
    Sdf = _Sdf
    Usd = _Usd
    UsdGeom = _UsdGeom
    UsdPhysics = _UsdPhysics
    UsdShade = _UsdShade


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_source(source_root: Path, relative: str) -> Path:
    source_root = source_root.resolve()
    path = (source_root / relative).resolve()
    if not path.is_relative_to(source_root) or not path.is_file():
        raise FileNotFoundError(f"required Moonlake source input not found: {relative}")
    return path


def verify_source_inputs(source_root: Path, manifest: dict) -> dict[str, dict[str, str]]:
    verified = {}
    for name, spec in manifest["sources"]["spatio_monorepo"]["files"].items():
        path = _required_source(source_root, spec["path"])
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise RuntimeError(f"checksum mismatch for {name}: {actual} != {spec['sha256']}")
        verified[name] = {"path": spec["path"], "sha256": actual, "usage": spec["usage"]}
    return verified


def load_binary_stl(path: Path) -> list[tuple[tuple[float, float, float], ...]]:
    """Load the binary STL representation used by the pinned 2060 extrusion."""
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"binary STL is truncated: {path}")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + triangle_count * 50
    if len(data) != expected_size:
        raise ValueError(f"binary STL size mismatch for {path}: {len(data)} != {expected_size}")
    triangles = []
    for index in range(triangle_count):
        values = struct.unpack_from("<12fH", data, 84 + index * 50)
        triangles.append(
            (
                (values[3], values[4], values[5]),
                (values[6], values[7], values[8]),
                (values[9], values[10], values[11]),
            )
        )
    return triangles


def _material(stage: Usd.Stage, path: str, color, *, roughness: float, metallic: float = 0.0):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(value) for value in color]))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind(geometry, material) -> None:
    UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim()).Bind(material)


def _box(stage: Usd.Stage, path: str, size, translation, material, *, collision: bool):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    transform = UsdGeom.Xformable(cube)
    transform.AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in translation]))
    transform.AddScaleOp().Set(Gf.Vec3f(*[float(value) for value in size]))
    _bind(cube, material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _cylinder(stage: Usd.Stage, path: str, radius: float, height: float, translation, material):
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.GetRadiusAttr().Set(float(radius))
    cylinder.GetHeightAttr().Set(float(height))
    cylinder.GetAxisAttr().Set(UsdGeom.Tokens.z)
    UsdGeom.Xformable(cylinder).AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in translation]))
    _bind(cylinder, material)
    return cylinder


def _extrusion_mesh(stage: Usd.Stage, path: str, stl_path: Path, scale: float, color) -> int:
    triangles = load_binary_stl(stl_path)
    points = [
        Gf.Vec3f(*(float(component) * float(scale) for component in vertex))
        for triangle in triangles
        for vertex in triangle
    ]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set([3] * len(triangles))
    mesh.GetFaceVertexIndicesAttr().Set(list(range(len(points))))
    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.GetDisplayColorAttr().Set([Gf.Vec3f(*[float(value) for value in color])])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return len(triangles)


def _bbox_metadata(stage: Usd.Stage, prim_path: str) -> dict:
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=False,
    )
    aligned = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
    minimum = aligned.GetMin()
    maximum = aligned.GetMax()
    vertices = [
        [float(x), float(y), float(z)]
        for x, y, z in itertools.product(
            (minimum[0], maximum[0]),
            (minimum[1], maximum[1]),
            (minimum[2], maximum[2]),
        )
    ]
    extents = [float(maximum[index] - minimum[index]) for index in range(3)]
    return {
        "vertices": vertices,
        "extents": extents,
        "radius": 0.5 * math.sqrt(sum(extent * extent for extent in extents)),
    }
