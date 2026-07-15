"""Author the source-pinned, static Moonlake office fixture USD."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import math
from pathlib import Path
import shutil
import struct
import uuid

import yaml

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
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*[float(value) for value in color])
    )
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


def _validate_static_fixture(stage: Usd.Stage, root_path: str, mount_frame: str) -> dict:
    required = (
        f"{root_path}/TableFrame/LegFL",
        f"{root_path}/ArmRail/Extrusion2060",
        f"{root_path}/CameraStand/D435Assembly/Body",
        f"{root_path}/{mount_frame}",
    )
    missing = [path for path in required if not stage.GetPrimAtPath(path).IsValid()]
    if missing:
        raise RuntimeError(f"generated Moonlake fixture is missing required prims: {missing}")

    collision_paths = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.MassAPI):
            raise RuntimeError(f"generated Moonlake fixture contains dynamic physics at {prim.GetPath()}")
        if prim.IsA(UsdPhysics.Joint):
            raise RuntimeError(f"generated Moonlake fixture contains a joint at {prim.GetPath()}")
        if prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"generated Moonlake fixture contains a camera sensor at {prim.GetPath()}")
        drive_attributes = [
            attribute.GetName()
            for attribute in prim.GetAttributes()
            if attribute.GetName().startswith("drive:")
        ]
        if drive_attributes:
            raise RuntimeError(f"generated Moonlake fixture contains drive attributes at {prim.GetPath()}")
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_paths.append(str(prim.GetPath()))

    expected_collisions = [
        f"{root_path}/TableFrame/LegFL",
        f"{root_path}/TableFrame/LegFR",
        f"{root_path}/TableFrame/LegBL",
        f"{root_path}/TableFrame/LegBR",
        f"{root_path}/ArmRail/Extrusion2060",
    ]
    if collision_paths != expected_collisions:
        raise RuntimeError(f"generated Moonlake fixture collision contract changed: {collision_paths}")

    optical = stage.GetPrimAtPath(f"{root_path}/{mount_frame}")
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(optical)
    return {
        "static_only": True,
        "camera_sensor_count": 0,
        "joint_count": 0,
        "collision_prims": collision_paths,
        "mount_frame_transform": [[float(matrix[row][column]) for column in range(4)] for row in range(4)],
    }


def author_fixture(extrusion_stl: Path, output_root: Path, manifest: dict) -> dict:
    _load_pxr()
    fixture = manifest["fixture"]
    instance = output_root / f"{int(fixture['instance_index']):05d}"
    if instance.exists():
        shutil.rmtree(instance)
    instance.mkdir(parents=True)
    output = instance / fixture["output"]

    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root_path = f"/{fixture['default_prim']}"
    root = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root.GetPrim())

    colors = fixture["materials"]
    looks = f"{root_path}/Looks"
    table_leg = _material(stage, f"{looks}/TableLeg", colors["table_leg"], roughness=0.4)
    motor_black = _material(stage, f"{looks}/MotorBlack", colors["motor_black"], roughness=0.3)
    camera_body = _material(
        stage, f"{looks}/CameraBody", colors["camera_body"], roughness=0.25, metallic=0.25
    )
    camera_face = _material(
        stage, f"{looks}/CameraFace", colors["camera_face"], roughness=0.22, metallic=0.3
    )
    lens = _material(stage, f"{looks}/LensGlass", colors["lens_glass"], roughness=0.06, metallic=0.75)

    table = fixture["table"]
    table_x, table_y, table_thickness = [float(value) for value in table["size_m"]]
    table_top = float(table["top_z_m"])
    leg_x, leg_y = [float(value) for value in table["leg_size_xy_m"]]
    inset = float(table["leg_edge_inset_m"])
    leg_height = table_top - table_thickness
    for name, sign_x, sign_y in (
        ("LegFL", -1, 1),
        ("LegFR", 1, 1),
        ("LegBL", -1, -1),
        ("LegBR", 1, -1),
    ):
        _box(
            stage,
            f"{root_path}/TableFrame/{name}",
            (leg_x, leg_y, leg_height),
            (
                sign_x * (table_x / 2.0 - inset),
                sign_y * (table_y / 2.0 - inset),
                leg_height / 2.0,
            ),
            table_leg,
            collision=True,
        )

    rail = fixture["rail"]
    triangle_count = _extrusion_mesh(
        stage,
        f"{root_path}/ArmRail/Extrusion2060",
        extrusion_stl,
        float(rail["stl_scale_to_m"]),
        colors["motor_black"],
    )
    rail_prim = stage.GetPrimAtPath(f"{root_path}/ArmRail/Extrusion2060")
    UsdGeom.Xformable(rail_prim).AddTranslateOp().Set(
        Gf.Vec3d(*[float(value) for value in rail["translation_m"]])
    )

    camera = fixture["top_camera"]
    stand_y = float(camera["stand_y_m"])
    top_camera_height = float(camera["body_center_m"][2])
    _box(
        stage,
        f"{root_path}/CameraStand/Clamp",
        (0.08, 0.05, 0.10),
        (0.0, stand_y, table_top - 0.015),
        motor_black,
        collision=False,
    )
    _cylinder(
        stage,
        f"{root_path}/CameraStand/LowerTube",
        0.014,
        0.32,
        (0.0, stand_y, table_top + 0.16),
        motor_black,
    )
    _cylinder(
        stage,
        f"{root_path}/CameraStand/MiddleTube",
        0.0125,
        0.30,
        (0.0, stand_y, table_top + 0.40),
        motor_black,
    )
    _cylinder(
        stage,
        f"{root_path}/CameraStand/UpperTube",
        0.011,
        0.24,
        (0.0, stand_y, table_top + 0.65),
        motor_black,
    )
    _cylinder(
        stage,
        f"{root_path}/CameraStand/BallHead",
        0.018,
        0.035,
        (0.0, stand_y, top_camera_height - 0.03),
        motor_black,
    )

    body_center = Gf.Vec3d(*[float(value) for value in camera["body_center_m"]])
    target = Gf.Vec3d(*[float(value) for value in camera["look_at_target_m"]])
    up = Gf.Vec3d(*[float(value) for value in camera["up_axis"]])
    camera_transform = Gf.Matrix4d().SetLookAt(body_center, target, up).GetInverse()
    assembly_path = f"{root_path}/CameraStand/D435Assembly"
    assembly = UsdGeom.Xform.Define(stage, assembly_path)
    UsdGeom.Xformable(assembly).AddTransformOp().Set(camera_transform)
    _box(stage, f"{assembly_path}/Body", (0.09, 0.025, 0.025), (0.0, 0.0, 0.0), camera_body, collision=False)
    _box(
        stage,
        f"{assembly_path}/FrontBezel",
        (0.084, 0.020, 0.001),
        (0.0, 0.0, -0.01255),
        camera_face,
        collision=False,
    )
    for index, lens_x in enumerate((-0.028, 0.0, 0.028)):
        _cylinder(
            stage,
            f"{assembly_path}/Lens{index}",
            0.0055,
            0.001,
            (lens_x, 0.0, -0.01275),
            lens,
        )

    mounts = UsdGeom.Xform.Define(stage, f"{root_path}/Mounts")
    UsdGeom.Xformable(mounts).AddTransformOp().Set(camera_transform)
    optical_frame = UsdGeom.Xform.Define(stage, f"{root_path}/{camera['mount_frame']}")
    UsdGeom.Xformable(optical_frame).AddTranslateOp().Set(
        Gf.Vec3d(*[float(value) for value in camera["optical_translation_m"]])
    )

    validation = _validate_static_fixture(stage, root_path, camera["mount_frame"])
    stage.GetRootLayer().Save()
    bbox = _bbox_metadata(stage, root_path)
    metadata = {
        "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, manifest["sources"]["spatio_monorepo"]["revision"])),
        "physics": {
            "type": "geometry",
            "static": True,
            "collision_prims": [
                "TableFrame/LegFL",
                "TableFrame/LegFR",
                "TableFrame/LegBL",
                "TableFrame/LegBR",
                "ArmRail/Extrusion2060",
            ],
        },
        "visual": {"source_matched": True},
        "geometry": {
            "aligned_bbox": {"vertices": bbox["vertices"], "extents": bbox["extents"]},
            "radius": bbox["radius"],
            "mesh_triangles": triangle_count,
            "up_axis": "Z",
            "meters_per_unit": 1.0,
        },
    }
    (instance / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (instance / "description.json").write_text(
        json.dumps(
            {
                "caption": "Moonlake office bimanual YAM static fixture",
                "description": ["Internal source-pinned table frame, arm rail, and overhead camera stand."],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "output": str(output.relative_to(output_root)),
        "output_sha256": sha256(output),
        "metadata": metadata,
        "mount_frames": [camera["mount_frame"]],
        "arm_mounts": fixture["arm_mounts"],
        "validation": validation,
    }


def build(source_root: Path, output_root: Path, manifest_path: Path) -> dict:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    verified_inputs = verify_source_inputs(source_root, manifest)
    extrusion = _required_source(
        source_root,
        manifest["sources"]["spatio_monorepo"]["files"]["extrusion_2060"]["path"],
    )
    authored = author_fixture(extrusion, output_root, manifest)
    source = manifest["sources"]["spatio_monorepo"]
    result = {
        "format_version": 1,
        "source": {
            "repository": source["repository"],
            "revision": source["revision"],
            "license": source["license"],
            "redistribution": source["redistribution"],
            "inputs": verified_inputs,
        },
        "build_manifest_sha256": sha256(manifest_path),
        "transformations": manifest["fixture"]["transformations"],
        **authored,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _load_pxr()
        logger.info("Built Moonlake office fixture: %s", build(args.source_root, args.output_root, args.manifest))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
