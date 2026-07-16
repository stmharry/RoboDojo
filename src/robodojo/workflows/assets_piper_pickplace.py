"""Procedural PiPER cube-to-bin task asset builder."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _vertices(minimum, maximum) -> list[list[float]]:
    return [
        [float(x), float(y), float(z)]
        for x, y, z in itertools.product(
            (minimum[0], maximum[0]), (minimum[1], maximum[1]), (minimum[2], maximum[2])
        )
    ]


def _metadata(spec: dict, minimum, maximum, *, physics_type: str, **extra) -> dict:
    extents = [float(maximum[index] - minimum[index]) for index in range(3)]
    vertices = _vertices(minimum, maximum)
    return {
        "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"robodojo:piper-pickplace:{spec['category']}")),
        "physics": {"type": physics_type, "mass_kg": float(spec["mass_kg"])},
        "geometry": {
            "aligned_bbox": {"vertices": vertices, "extents": extents},
            "oriented_bbox": {"vertices": vertices, "extents": extents},
            **extra,
        },
    }


def _material(stage, path: str, color):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.42)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _cube(stage, path: str, size, center, material):
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    transform = UsdGeom.Xformable(cube)
    transform.AddTranslateOp().Set(Gf.Vec3d(*center))
    transform.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _new_stage(path: Path, default_prim: str):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{default_prim}")
    stage.SetDefaultPrim(root.GetPrim())
    return stage, root


def _author_cube(spec: dict, output: Path) -> dict:
    from pxr import UsdPhysics

    stage, root = _new_stage(output, "PiperBlueCube")
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(float(spec["mass_kg"]))
    material = _material(stage, "/PiperBlueCube/Looks/Blue", spec["color_linear_rgb"])
    x, y, z = [float(value) for value in spec["dimensions_m"]]
    _cube(stage, "/PiperBlueCube/Body", (x, y, z), (0.0, 0.0, z / 2), material)
    stage.GetRootLayer().Save()
    return _metadata(spec, (-x / 2, -y / 2, 0.0), (x / 2, y / 2, z), physics_type="rigid")


def _author_bin(spec: dict, output: Path) -> dict:
    from pxr import UsdGeom, UsdPhysics

    stage, root = _new_stage(output, "PiperRedBin")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    base = UsdGeom.Xform.Define(stage, "/PiperRedBin/base")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    UsdPhysics.MassAPI.Apply(base.GetPrim()).CreateMassAttr(float(spec["mass_kg"]))
    material = _material(stage, "/PiperRedBin/Looks/Red", spec["color_linear_rgb"])
    cavity_x, cavity_y, cavity_z = [float(value) for value in spec["cavity_dimensions_m"]]
    wall = float(spec["wall_thickness_m"])
    outer_x, outer_y = cavity_x + 2 * wall, cavity_y + 2 * wall
    _cube(stage, "/PiperRedBin/base/Floor", (outer_x, outer_y, wall), (0.0, 0.0, wall / 2), material)
    _cube(
        stage,
        "/PiperRedBin/base/WallLeft",
        (wall, outer_y, cavity_z),
        (-(cavity_x + wall) / 2, 0.0, wall + cavity_z / 2),
        material,
    )
    _cube(
        stage,
        "/PiperRedBin/base/WallRight",
        (wall, outer_y, cavity_z),
        ((cavity_x + wall) / 2, 0.0, wall + cavity_z / 2),
        material,
    )
    _cube(
        stage,
        "/PiperRedBin/base/WallNear",
        (cavity_x, wall, cavity_z),
        (0.0, -(cavity_y + wall) / 2, wall + cavity_z / 2),
        material,
    )
    _cube(
        stage,
        "/PiperRedBin/base/WallFar",
        (cavity_x, wall, cavity_z),
        (0.0, (cavity_y + wall) / 2, wall + cavity_z / 2),
        material,
    )
    stage.GetRootLayer().Save()
    maximum = (outer_x / 2, outer_y / 2, wall + cavity_z)
    minimum = (-outer_x / 2, -outer_y / 2, 0.0)
    metadata = _metadata(
        spec,
        minimum,
        maximum,
        physics_type="articulation",
        link_bboxes={"base": {"vertices": _vertices(minimum, maximum)}},
    )
    metadata["passive"] = {
        "volumes": {
            spec["functional_volume_tag"]: {
                "base_link": "base",
                "minimum": [-cavity_x / 2, -cavity_y / 2, wall],
                "maximum": [cavity_x / 2, cavity_y / 2, wall + cavity_z],
            }
        }
    }
    return metadata


def _validate(path: Path, object_type: str) -> dict:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    if stage is None or not stage.GetDefaultPrim().IsValid():
        raise RuntimeError(f"generated asset could not be reopened: {path}")
    inventory = {"articulations": 0, "rigid_bodies": 0, "collisions": 0}
    for prim in stage.Traverse():
        inventory["articulations"] += int(prim.HasAPI(UsdPhysics.ArticulationRootAPI))
        inventory["rigid_bodies"] += int(prim.HasAPI(UsdPhysics.RigidBodyAPI))
        inventory["collisions"] += int(prim.HasAPI(UsdPhysics.CollisionAPI))
    expected_articulations = int(object_type == "Articulation")
    if inventory["articulations"] != expected_articulations or inventory["rigid_bodies"] != 1:
        raise RuntimeError(f"invalid generated {object_type} inventory: {inventory}")
    if inventory["collisions"] < 1:
        raise RuntimeError(f"generated {object_type} has no collision geometry")
    return inventory


def _publish(staged: list[tuple[Path, Path]]) -> None:
    published = []
    try:
        for source, destination in staged:
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup = destination.with_name(f".{destination.name}.piper-pickplace-backup")
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink(missing_ok=True)
            if destination.exists():
                os.replace(destination, backup)
            os.replace(source, destination)
            published.append((destination, backup))
    except Exception:
        for destination, backup in reversed(published):
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink(missing_ok=True)
            if backup.exists():
                os.replace(backup, destination)
        raise
    for _, backup in published:
        if backup.is_dir():
            shutil.rmtree(backup)
        else:
            backup.unlink(missing_ok=True)


def build(output_root: Path, manifest_path: Path) -> dict:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".piper-pickplace-", dir=output_root))
    publications = []
    records = {}
    authors = {"cube": _author_cube, "bin": _author_bin}
    try:
        for key, spec in manifest["assets"].items():
            relative = Path("Object/RoboDojo") / spec["object_type"] / spec["category"] / f"{int(spec['index']):05d}"
            instance = staging_root / relative
            instance.mkdir(parents=True)
            metadata = authors[key](spec, instance / "object.usd")
            validation = _validate(instance / "object.usd", spec["object_type"])
            _write_json(instance / "metadata.json", metadata)
            _write_json(
                instance / "description.json",
                {"caption": spec["description"], "description": [spec["description"]]},
            )
            provenance = {
                "format_version": 1,
                "distribution": manifest["distribution"],
                "reference": manifest["references"]["training_dataset"],
                "specification": spec,
                "tooling_manifest_sha256": _sha256(manifest_path),
            }
            _write_json(instance / "provenance.json", provenance)
            files = {
                name: _sha256(instance / name)
                for name in ("object.usd", "metadata.json", "description.json", "provenance.json")
            }
            records[key] = {
                "asset": relative.as_posix(),
                "metadata": metadata,
                "validation": validation,
                "files": files,
            }
            publications.append((instance, output_root / relative))
        shared = {
            "format_version": 1,
            "distribution": manifest["distribution"],
            "tooling_manifest_sha256": _sha256(manifest_path),
            "assets": records,
        }
        staged_manifest = staging_root / "piper_pickplace.json"
        _write_json(staged_manifest, shared)
        publications.append((staged_manifest, output_root / "manifests" / "piper_pickplace.json"))
        _publish(publications)
        return shared
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        build(args.output_root, args.manifest)
    finally:
        app.close()


if __name__ == "__main__":
    main()
