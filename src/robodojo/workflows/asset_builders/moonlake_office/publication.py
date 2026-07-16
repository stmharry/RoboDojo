"""Conversion and atomic publication for the Moonlake office fixture."""

import argparse
import json
import logging
from pathlib import Path
import shutil
import uuid

import yaml

from robodojo.workflows.asset_builders.moonlake_office import geometry, validation as validator

logger = logging.getLogger(__name__)


def author_fixture(extrusion_stl: Path, output_root: Path, manifest: dict) -> dict:
    geometry._load_pxr()
    fixture = manifest["fixture"]
    instance = output_root / f"{int(fixture['instance_index']):05d}"
    if instance.exists():
        shutil.rmtree(instance)
    instance.mkdir(parents=True)
    output = instance / fixture["output"]

    stage = geometry.Usd.Stage.CreateNew(str(output))
    geometry.UsdGeom.SetStageUpAxis(stage, geometry.UsdGeom.Tokens.z)
    geometry.UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root_path = f"/{fixture['default_prim']}"
    root = geometry.UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root.GetPrim())

    colors = fixture["materials"]
    looks = f"{root_path}/Looks"
    table_leg = geometry._material(stage, f"{looks}/TableLeg", colors["table_leg"], roughness=0.4)
    motor_black = geometry._material(stage, f"{looks}/MotorBlack", colors["motor_black"], roughness=0.3)
    camera_body = geometry._material(stage, f"{looks}/CameraBody", colors["camera_body"], roughness=0.25, metallic=0.25)
    camera_face = geometry._material(stage, f"{looks}/CameraFace", colors["camera_face"], roughness=0.22, metallic=0.3)
    lens = geometry._material(stage, f"{looks}/LensGlass", colors["lens_glass"], roughness=0.06, metallic=0.75)

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
        geometry._box(
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
    triangle_count = geometry._extrusion_mesh(
        stage,
        f"{root_path}/ArmRail/Extrusion2060",
        extrusion_stl,
        float(rail["stl_scale_to_m"]),
        colors["motor_black"],
    )
    rail_prim = stage.GetPrimAtPath(f"{root_path}/ArmRail/Extrusion2060")
    geometry.UsdGeom.Xformable(rail_prim).AddTranslateOp().Set(
        geometry.Gf.Vec3d(*[float(value) for value in rail["translation_m"]])
    )

    camera = fixture["top_camera"]
    stand_y = float(camera["stand_y_m"])
    top_camera_height = float(camera["body_center_m"][2])
    geometry._box(
        stage,
        f"{root_path}/CameraStand/Clamp",
        (0.08, 0.05, 0.10),
        (0.0, stand_y, table_top - 0.015),
        motor_black,
        collision=False,
    )
    geometry._cylinder(
        stage,
        f"{root_path}/CameraStand/LowerTube",
        0.014,
        0.32,
        (0.0, stand_y, table_top + 0.16),
        motor_black,
    )
    geometry._cylinder(
        stage,
        f"{root_path}/CameraStand/MiddleTube",
        0.0125,
        0.30,
        (0.0, stand_y, table_top + 0.40),
        motor_black,
    )
    geometry._cylinder(
        stage,
        f"{root_path}/CameraStand/UpperTube",
        0.011,
        0.24,
        (0.0, stand_y, table_top + 0.65),
        motor_black,
    )
    geometry._cylinder(
        stage,
        f"{root_path}/CameraStand/BallHead",
        0.018,
        0.035,
        (0.0, stand_y, top_camera_height - 0.03),
        motor_black,
    )

    body_center = geometry.Gf.Vec3d(*[float(value) for value in camera["body_center_m"]])
    target = geometry.Gf.Vec3d(*[float(value) for value in camera["look_at_target_m"]])
    up = geometry.Gf.Vec3d(*[float(value) for value in camera["up_axis"]])
    camera_transform = geometry.Gf.Matrix4d().SetLookAt(body_center, target, up).GetInverse()
    assembly_path = f"{root_path}/CameraStand/D435Assembly"
    assembly = geometry.UsdGeom.Xform.Define(stage, assembly_path)
    geometry.UsdGeom.Xformable(assembly).AddTransformOp().Set(camera_transform)
    geometry._box(stage, f"{assembly_path}/Body", (0.09, 0.025, 0.025), (0.0, 0.0, 0.0), camera_body, collision=False)
    geometry._box(
        stage,
        f"{assembly_path}/FrontBezel",
        (0.084, 0.020, 0.001),
        (0.0, 0.0, -0.01255),
        camera_face,
        collision=False,
    )
    for index, lens_x in enumerate((-0.028, 0.0, 0.028)):
        geometry._cylinder(
            stage,
            f"{assembly_path}/Lens{index}",
            0.0055,
            0.001,
            (lens_x, 0.0, -0.01275),
            lens,
        )

    mounts = geometry.UsdGeom.Xform.Define(stage, f"{root_path}/Mounts")
    geometry.UsdGeom.Xformable(mounts).AddTransformOp().Set(camera_transform)
    optical_frame = geometry.UsdGeom.Xform.Define(stage, f"{root_path}/{camera['mount_frame']}")
    geometry.UsdGeom.Xformable(optical_frame).AddTranslateOp().Set(
        geometry.Gf.Vec3d(*[float(value) for value in camera["optical_translation_m"]])
    )

    validation = validator._validate_static_fixture(stage, root_path, camera["mount_frame"])
    stage.GetRootLayer().Save()
    bbox = geometry._bbox_metadata(stage, root_path)
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
        "output_sha256": geometry.sha256(output),
        "metadata": metadata,
        "mount_frames": [camera["mount_frame"]],
        "arm_mounts": fixture["arm_mounts"],
        "validation": validation,
    }


def build(source_root: Path, output_root: Path, manifest_path: Path) -> dict:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    verified_inputs = geometry.verify_source_inputs(source_root, manifest)
    extrusion = geometry._required_source(
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
        "build_manifest_sha256": geometry.sha256(manifest_path),
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
        geometry._load_pxr()
        logger.info("Built Moonlake office fixture: %s", build(args.source_root, args.output_root, args.manifest))
    finally:
        simulation_app.close()
