"""Geometry and USD authoring for Moonlake packing assets."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import uuid

Gf = None
PhysxSchema = None
Sdf = None
Usd = None
UsdGeom = None
UsdPhysics = None
UsdShade = None


def _load_pxr() -> None:
    global Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    if Usd is not None:
        return
    from pxr import (
        Gf as _Gf,
        PhysxSchema as _PhysxSchema,
        Sdf as _Sdf,
        Usd as _Usd,
        UsdGeom as _UsdGeom,
        UsdPhysics as _UsdPhysics,
        UsdShade as _UsdShade,
    )

    Gf = _Gf
    PhysxSchema = _PhysxSchema
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


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bbox_vertices(minimum, maximum) -> list[list[float]]:
    return [
        [float(x), float(y), float(z)]
        for x, y, z in itertools.product(
            (minimum[0], maximum[0]),
            (minimum[1], maximum[1]),
            (minimum[2], maximum[2]),
        )
    ]


def _geometry_metadata(minimum, maximum, **extra) -> dict:
    extents = [float(maximum[index] - minimum[index]) for index in range(3)]
    vertices = _bbox_vertices(minimum, maximum)
    return {
        "aligned_bbox": {"vertices": vertices, "extents": extents},
        "oriented_bbox": {"vertices": vertices, "extents": extents},
        "radius": 0.5 * math.sqrt(sum(extent * extent for extent in extents)),
        **extra,
    }


def _material(stage, path: str, color, *, roughness: float = 0.45, metallic: float = 0.0):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind(geometry, material) -> None:
    UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim()).Bind(material)


def _cube(stage, path: str, size, center, material, *, collision: bool = True):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    transform = UsdGeom.Xformable(cube)
    transform.AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in center]))
    transform.AddScaleOp().Set(Gf.Vec3f(*[float(value) for value in size]))
    _bind(cube, material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _sphere(stage, path: str, radius: float, center, scale, material, *, collision: bool = True):
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.GetRadiusAttr().Set(float(radius))
    transform = UsdGeom.Xformable(sphere)
    transform.AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in center]))
    transform.AddScaleOp().Set(Gf.Vec3f(*[float(value) for value in scale]))
    _bind(sphere, material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    return sphere


def _cylinder(stage, path: str, radius: float, height: float, center, material, *, collision: bool = True):
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.GetAxisAttr().Set(UsdGeom.Tokens.x)
    cylinder.GetRadiusAttr().Set(float(radius))
    cylinder.GetHeightAttr().Set(float(height))
    UsdGeom.Xformable(cylinder).AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in center]))
    _bind(cylinder, material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    return cylinder


def _capsule(stage, path: str, radius: float, height: float, material, *, collision: bool = True):
    capsule = UsdGeom.Capsule.Define(stage, path)
    capsule.GetAxisAttr().Set(UsdGeom.Tokens.x)
    capsule.GetRadiusAttr().Set(float(radius))
    capsule.GetHeightAttr().Set(float(height))
    _bind(capsule, material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
    return capsule


def _rigid_body(prim, mass: float, *, disable_gravity: bool = False) -> None:
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
    if disable_gravity:
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)


def _new_stage(output: Path, default_prim: str):
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{default_prim}")
    stage.SetDefaultPrim(root.GetPrim())
    return stage, root


def _base_metadata(spec: dict, minimum, maximum, physics_type: str, **geometry_extra) -> dict:
    return {
        "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"robodojo:moonlake-packing:{spec['category']}")),
        "physics": {"type": physics_type, "mass_kg": float(spec.get("mass_kg", 0.0))},
        "geometry": _geometry_metadata(minimum, maximum, **geometry_extra),
    }


def _author_spoon(spec: dict, output: Path) -> dict:
    stage, root = _new_stage(output, "MoonlakeMeasuringSpoon")
    _rigid_body(root.GetPrim(), spec["mass_kg"])
    pink = _material(stage, "/MoonlakeMeasuringSpoon/Looks/Pink", (0.94, 0.28, 0.52), roughness=0.38)
    length, width, height = [float(value) for value in spec["dimensions_m"]]
    _cube(
        stage,
        "/MoonlakeMeasuringSpoon/Handle",
        (length * 0.68, width * 0.22, height * 0.42),
        (-length * 0.15, 0.0, height * 0.30),
        pink,
    )
    _sphere(
        stage,
        "/MoonlakeMeasuringSpoon/Bowl",
        0.5,
        (length * 0.34, 0.0, height * 0.50),
        (length * 0.16, width, height),
        pink,
    )
    stage.GetRootLayer().Save()
    return _base_metadata(spec, (-length / 2, -width / 2, 0.0), (length / 2, width / 2, height), "rigid")


def _author_phone(spec: dict, output: Path) -> dict:
    stage, root = _new_stage(output, "MoonlakePhone15Dummy")
    _rigid_body(root.GetPrim(), spec["mass_kg"])
    pink = _material(stage, "/MoonlakePhone15Dummy/Looks/Pink", (0.94, 0.56, 0.68), roughness=0.26)
    screen = _material(stage, "/MoonlakePhone15Dummy/Looks/Screen", (0.12, 0.30, 0.56), roughness=0.12)
    lens = _material(stage, "/MoonlakePhone15Dummy/Looks/Lens", (0.02, 0.025, 0.035), roughness=0.08)
    length, width, height = [float(value) for value in spec["dimensions_m"]]
    _cube(stage, "/MoonlakePhone15Dummy/Body", (length, width, height), (0.0, 0.0, height / 2), pink)
    _cube(
        stage,
        "/MoonlakePhone15Dummy/Screen",
        (length * 0.92, width * 0.88, height * 0.08),
        (0.0, 0.0, height * 1.02),
        screen,
        collision=False,
    )
    for index, x in enumerate((-length * 0.36, -length * 0.26)):
        _cylinder(
            stage,
            f"/MoonlakePhone15Dummy/Camera{index}",
            width * 0.09,
            height * 0.10,
            (x, width * 0.32, height * 1.06),
            lens,
            collision=False,
        )
    stage.GetRootLayer().Save()
    return _base_metadata(spec, (-length / 2, -width / 2, 0.0), (length / 2, width / 2, height), "rigid")


def _author_apple(spec: dict, output: Path) -> dict:
    stage, root = _new_stage(output, "MoonlakeFakeApple")
    _rigid_body(root.GetPrim(), spec["mass_kg"])
    red = _material(stage, "/MoonlakeFakeApple/Looks/Red", (0.72, 0.035, 0.025), roughness=0.34)
    brown = _material(stage, "/MoonlakeFakeApple/Looks/Stem", (0.18, 0.065, 0.025), roughness=0.72)
    x_size, y_size, z_size = [float(value) for value in spec["dimensions_m"]]
    _sphere(
        stage,
        "/MoonlakeFakeApple/Fruit",
        0.5,
        (0.0, 0.0, z_size * 0.47),
        (x_size, y_size, z_size * 0.94),
        red,
    )
    stem = UsdGeom.Cylinder.Define(stage, "/MoonlakeFakeApple/Stem")
    stem.GetAxisAttr().Set(UsdGeom.Tokens.z)
    stem.GetRadiusAttr().Set(0.002)
    stem.GetHeightAttr().Set(z_size * 0.12)
    UsdGeom.Xformable(stem).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z_size * 0.94))
    _bind(stem, brown)
    stage.GetRootLayer().Save()
    return _base_metadata(
        spec,
        (-x_size / 2, -y_size / 2, 0.0),
        (x_size / 2, y_size / 2, z_size),
        "rigid",
    )


def _author_screwdriver(spec: dict, output: Path) -> dict:
    stage, root = _new_stage(output, "MoonlakePhillipsScrewdriver")
    _rigid_body(root.GetPrim(), spec["mass_kg"])
    handle_material = _material(stage, "/MoonlakePhillipsScrewdriver/Looks/Handle", (0.92, 0.55, 0.035), roughness=0.48)
    steel = _material(
        stage, "/MoonlakePhillipsScrewdriver/Looks/Steel", (0.46, 0.48, 0.50), roughness=0.22, metallic=0.85
    )
    length, width, height = [float(value) for value in spec["dimensions_m"]]
    shaft_length = float(spec["shaft_length_m"])
    handle_length = length - shaft_length
    handle_center = -length / 2 + handle_length / 2
    shaft_center = length / 2 - shaft_length / 2
    _cylinder(
        stage,
        "/MoonlakePhillipsScrewdriver/Handle",
        width / 2,
        handle_length,
        (handle_center, 0.0, height / 2),
        handle_material,
    )
    _cylinder(stage, "/MoonlakePhillipsScrewdriver/Shaft", 0.0032, shaft_length, (shaft_center, 0.0, height / 2), steel)
    _cube(
        stage,
        "/MoonlakePhillipsScrewdriver/Tip",
        (0.008, 0.006, 0.006),
        (length / 2 - 0.004, 0.0, height / 2),
        steel,
    )
    stage.GetRootLayer().Save()
    return _base_metadata(spec, (-length / 2, -width / 2, 0.0), (length / 2, width / 2, height), "rigid")


def _author_block(spec: dict, output: Path) -> dict:
    stage, root = _new_stage(output, "MoonlakeABCBlock")
    _rigid_body(root.GetPrim(), spec["mass_kg"])
    wood = _material(stage, "/MoonlakeABCBlock/Looks/Wood", (0.73, 0.52, 0.28), roughness=0.68)
    blue = _material(stage, "/MoonlakeABCBlock/Looks/Blue", (0.08, 0.42, 0.72), roughness=0.55)
    x_size, y_size, z_size = [float(value) for value in spec["dimensions_m"]]
    _cube(stage, "/MoonlakeABCBlock/Body", (x_size, y_size, z_size), (0.0, 0.0, z_size / 2), wood)
    _cube(
        stage,
        "/MoonlakeABCBlock/FaceInset",
        (x_size * 0.64, y_size * 0.02, z_size * 0.64),
        (0.0, -y_size * 0.505, z_size / 2),
        blue,
        collision=False,
    )
    stage.GetRootLayer().Save()
    return _base_metadata(spec, (-x_size / 2, -y_size / 2, 0.0), (x_size / 2, y_size / 2, z_size), "rigid")


def _author_container(spec: dict, output: Path) -> dict:
    stage, root = _new_stage(output, "MoonlakeMagneticGiftBox")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim()).CreateEnabledSelfCollisionsAttr(False)
    black = _material(stage, "/MoonlakeMagneticGiftBox/Looks/BlackPaper", (0.012, 0.012, 0.016), roughness=0.58)
    sx, sy, sz = [float(value) for value in spec["dimensions_m"]]
    thickness = float(spec["wall_thickness_m"])

    base = UsdGeom.Xform.Define(stage, "/MoonlakeMagneticGiftBox/base")
    _rigid_body(base.GetPrim(), spec["base_mass_kg"])
    _cube(stage, "/MoonlakeMagneticGiftBox/base/Bottom", (sx, sy, thickness), (0.0, 0.0, thickness / 2), black)
    _cube(
        stage,
        "/MoonlakeMagneticGiftBox/base/WallFront",
        (sx, thickness, sz),
        (0.0, -sy / 2 + thickness / 2, sz / 2),
        black,
    )
    _cube(
        stage,
        "/MoonlakeMagneticGiftBox/base/WallBack",
        (sx, thickness, sz),
        (0.0, sy / 2 - thickness / 2, sz / 2),
        black,
    )
    _cube(
        stage,
        "/MoonlakeMagneticGiftBox/base/WallLeft",
        (thickness, sy - 2 * thickness, sz),
        (-sx / 2 + thickness / 2, 0.0, sz / 2),
        black,
    )
    _cube(
        stage,
        "/MoonlakeMagneticGiftBox/base/WallRight",
        (thickness, sy - 2 * thickness, sz),
        (sx / 2 - thickness / 2, 0.0, sz / 2),
        black,
    )

    lid = UsdGeom.Xform.Define(stage, "/MoonlakeMagneticGiftBox/lid")
    lid_transform = UsdGeom.Xformable(lid)
    lid_transform.AddTranslateOp().Set(Gf.Vec3d(0.0, sy / 2, sz))
    lid_transform.AddRotateXOp().Set(float(spec["lid_open_deg"]))
    _rigid_body(lid.GetPrim(), spec["lid_mass_kg"], disable_gravity=True)
    _cube(
        stage,
        "/MoonlakeMagneticGiftBox/lid/Panel",
        (sx, sy, thickness * 1.5),
        (0.0, -sy / 2, thickness * 0.75),
        black,
    )
    _cube(
        stage,
        "/MoonlakeMagneticGiftBox/lid/FrontLip",
        (sx, thickness * 2.0, 0.018),
        (0.0, -sy + thickness, -0.006),
        black,
    )

    joint = UsdPhysics.RevoluteJoint.Define(stage, "/MoonlakeMagneticGiftBox/lid_hinge")
    joint.CreateBody0Rel().SetTargets([base.GetPath()])
    joint.CreateBody1Rel().SetTargets([lid.GetPath()])
    joint.CreateAxisAttr(UsdPhysics.Tokens.x)
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, sy / 2, sz))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(float(spec["lid_open_deg"]))
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(0.08)
    drive.CreateTargetPositionAttr(float(spec["lid_open_deg"]))

    stage.GetRootLayer().Save()
    metadata = _base_metadata(
        {**spec, "mass_kg": float(spec["base_mass_kg"]) + float(spec["lid_mass_kg"])},
        (-sx / 2, -sy / 2, 0.0),
        (sx / 2, sy / 2, sz + thickness * 1.5),
        "articulation",
    )
    metadata["passive"] = {
        "functional": {"lid": {"parent_joint": "lid_hinge"}},
        "volumes": {
            "packing_cavity": {
                "base_link": "base",
                "minimum": [-sx / 2 + thickness, -sy / 2 + thickness, thickness],
                "maximum": [sx / 2 - thickness, sy / 2 - thickness, sz - thickness],
            }
        },
    }
    metadata["physics"].update(
        {
            "base_mass_kg": float(spec["base_mass_kg"]),
            "lid_mass_kg": float(spec["lid_mass_kg"]),
            "lid_open_deg": float(spec["lid_open_deg"]),
        }
    )
    return metadata


def _quatf(quaternion) -> object:
    return Gf.Quatf(
        float(quaternion.GetReal()),
        Gf.Vec3f(*[float(value) for value in quaternion.GetImaginary()]),
    )


def _segment_points(segment_count: int, segment_length: float, z: float) -> list[tuple[float, float, float]]:
    points = [(-2.5 * segment_length, 2.0 * segment_length, z)]
    direction = 1.0
    for row in range(5):
        for _ in range(5):
            x, y, _ = points[-1]
            points.append((x + direction * segment_length, y, z))
        if row < 4:
            x, y, _ = points[-1]
            points.append((x, y - segment_length, z))
        direction *= -1.0
    x, y, _ = points[-1]
    points.append((x, y - segment_length, z))
    if len(points) != segment_count + 1:
        raise RuntimeError(f"cable path produced {len(points) - 1} segments, expected {segment_count}")
    return points


def _link_orientation(direction) -> object:
    return Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), Gf.Vec3d(*direction)).GetQuat()


def _author_cable(spec: dict, output: Path) -> dict:
    stage, root = _new_stage(output, "MoonlakeAnkerCable")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim()).CreateEnabledSelfCollisionsAttr(False)
    cable_material = _material(stage, "/MoonlakeAnkerCable/Looks/Braid", (0.015, 0.017, 0.022), roughness=0.72)
    connector_material = _material(stage, "/MoonlakeAnkerCable/Looks/Connector", (0.08, 0.09, 0.10), roughness=0.34)
    metal = _material(stage, "/MoonlakeAnkerCable/Looks/Metal", (0.45, 0.47, 0.49), roughness=0.20, metallic=0.85)

    count = int(spec["segment_count"])
    radius = float(spec["diameter_m"]) / 2.0
    usb_a_length = float(spec["usb_a_length_m"])
    usb_c_length = float(spec["usb_c_length_m"])
    flexible_length = float(spec["length_m"]) - usb_a_length - usb_c_length
    segment_length = flexible_length / count
    points = _segment_points(count, segment_length, radius)
    segment_mass = float(spec["mass_kg"]) * 0.80 / count
    connector_mass = float(spec["mass_kg"]) * 0.10
    link_bboxes = {}
    links = []
    orientations = []

    for index, (start, end) in enumerate(zip(points[:-1], points[1:], strict=True)):
        direction = tuple((end[axis] - start[axis]) / segment_length for axis in range(3))
        center = tuple((start[axis] + end[axis]) / 2.0 for axis in range(3))
        orientation = _link_orientation(direction)
        link = UsdGeom.Xform.Define(stage, f"/MoonlakeAnkerCable/segment_{index:02d}")
        transform = UsdGeom.Xformable(link)
        transform.AddTranslateOp().Set(Gf.Vec3d(*center))
        transform.AddOrientOp().Set(_quatf(orientation))
        _rigid_body(link.GetPrim(), segment_mass)
        body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(link.GetPrim())
        body_api.CreateLinearDampingAttr(0.10)
        body_api.CreateAngularDampingAttr(0.20)
        body_api.CreateSolverPositionIterationCountAttr(8)
        body_api.CreateSolverVelocityIterationCountAttr(2)
        _capsule(
            stage,
            f"/MoonlakeAnkerCable/segment_{index:02d}/Body",
            radius,
            max(segment_length - 2 * radius, radius),
            cable_material,
        )
        links.append(link)
        orientations.append(orientation)
        link_bboxes[f"segment_{index:02d}"] = {
            "vertices": _bbox_vertices(
                (-segment_length / 2, -radius, -radius),
                (segment_length / 2, radius, radius),
            )
        }

    for index in range(count - 1):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/MoonlakeAnkerCable/joint_{index:02d}")
        joint.CreateBody0Rel().SetTargets([links[index].GetPath()])
        joint.CreateBody1Rel().SetTargets([links[index + 1].GetPath()])
        joint.CreateAxisAttr(UsdPhysics.Tokens.z if index % 2 == 0 else UsdPhysics.Tokens.y)
        joint.CreateLocalPos0Attr(Gf.Vec3f(segment_length / 2, 0.0, 0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(-segment_length / 2, 0.0, 0.0))
        joint.CreateLocalRot0Attr(_quatf(orientations[index].GetInverse()))
        joint.CreateLocalRot1Attr(_quatf(orientations[index + 1].GetInverse()))
        joint.CreateLowerLimitAttr(-85.0)
        joint.CreateUpperLimitAttr(85.0)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(0.0005)

    connector_specs = (
        ("usb_a", usb_a_length, points[0], points[1]),
        ("usb_c", usb_c_length, points[-1], points[-2]),
    )
    connectors = {}
    for name, length, endpoint, neighbor in connector_specs:
        inward = tuple((neighbor[axis] - endpoint[axis]) / segment_length for axis in range(3))
        outward = tuple(-inward[axis] for axis in range(3))
        center = tuple(endpoint[axis] + outward[axis] * length / 2.0 for axis in range(3))
        orientation = _link_orientation(outward)
        link = UsdGeom.Xform.Define(stage, f"/MoonlakeAnkerCable/{name}")
        transform = UsdGeom.Xformable(link)
        transform.AddTranslateOp().Set(Gf.Vec3d(*center))
        transform.AddOrientOp().Set(_quatf(orientation))
        _rigid_body(link.GetPrim(), connector_mass)
        body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(link.GetPrim())
        body_api.CreateLinearDampingAttr(0.10)
        body_api.CreateAngularDampingAttr(0.20)
        body_api.CreateSolverPositionIterationCountAttr(8)
        body_api.CreateSolverVelocityIterationCountAttr(2)
        width = 0.012 if name == "usb_a" else 0.010
        height = float(spec["diameter_m"])
        _cube(
            stage,
            f"/MoonlakeAnkerCable/{name}/Housing",
            (length, width, height),
            (0.0, 0.0, 0.0),
            connector_material,
        )
        _cube(
            stage,
            f"/MoonlakeAnkerCable/{name}/Plug",
            (length * 0.28, width * 0.82, height * 0.74),
            (length * 0.58, 0.0, 0.0),
            metal,
            collision=False,
        )
        connectors[name] = (link, orientation, length, width, height)
        link_bboxes[name] = {
            "vertices": _bbox_vertices((-length / 2, -width / 2, -height / 2), (length / 2, width / 2, height / 2))
        }

    first_joint = UsdPhysics.FixedJoint.Define(stage, "/MoonlakeAnkerCable/usb_a_fixed")
    first_joint.CreateBody0Rel().SetTargets([links[0].GetPath()])
    first_joint.CreateBody1Rel().SetTargets([connectors["usb_a"][0].GetPath()])
    first_joint.CreateLocalPos0Attr(Gf.Vec3f(-segment_length / 2, 0.0, 0.0))
    first_joint.CreateLocalPos1Attr(Gf.Vec3f(-usb_a_length / 2, 0.0, 0.0))
    first_joint.CreateLocalRot0Attr(_quatf(orientations[0].GetInverse()))
    first_joint.CreateLocalRot1Attr(_quatf(connectors["usb_a"][1].GetInverse()))

    last_joint = UsdPhysics.FixedJoint.Define(stage, "/MoonlakeAnkerCable/usb_c_fixed")
    last_joint.CreateBody0Rel().SetTargets([links[-1].GetPath()])
    last_joint.CreateBody1Rel().SetTargets([connectors["usb_c"][0].GetPath()])
    last_joint.CreateLocalPos0Attr(Gf.Vec3f(segment_length / 2, 0.0, 0.0))
    last_joint.CreateLocalPos1Attr(Gf.Vec3f(-usb_c_length / 2, 0.0, 0.0))
    last_joint.CreateLocalRot0Attr(_quatf(orientations[-1].GetInverse()))
    last_joint.CreateLocalRot1Attr(_quatf(connectors["usb_c"][1].GetInverse()))

    stage.GetRootLayer().Save()
    all_x = [point[0] for point in points]
    all_y = [point[1] for point in points]
    minimum = (min(all_x) - usb_a_length - radius, min(all_y) - radius, 0.0)
    maximum = (max(all_x) + usb_c_length + radius, max(all_y) + radius, radius * 2)
    metadata = _base_metadata(
        spec,
        minimum,
        maximum,
        "articulation",
        link_bboxes=link_bboxes,
        centerline_length_m=float(spec["length_m"]),
    )
    metadata["physics"].update(
        {
            "segment_count": count,
            "rigid_body_count": count + 2,
            "joint_count": count + 1,
            "joint_model": "alternating_revolute",
            "self_collision": False,
        }
    )
    return metadata
