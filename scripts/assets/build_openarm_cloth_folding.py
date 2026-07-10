#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402
import trimesh  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shift_joint_anchor(stage: Usd.Stage, side: str, distance: float) -> str:
    matches = [
        prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint) and f"{side}_joint4" in prim.GetName()
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {side} joint4, found {[str(p.GetPath()) for p in matches]}")
    joint = UsdPhysics.Joint(matches[0])
    candidates = [joint.GetLocalPos0Attr(), joint.GetLocalPos1Attr()]
    values = [attr.Get() for attr in candidates]
    index = max(range(2), key=lambda i: Gf.Vec3d(values[i]).GetLength())
    vector = Gf.Vec3d(values[index])
    if vector.GetLength() == 0:
        raise RuntimeError(f"cannot derive local upper-arm axis for {matches[0].GetPath()}")
    shifted = vector + vector.GetNormalized() * distance
    candidates[index].Set(Gf.Vec3f(*shifted))
    return str(matches[0].GetPath())


def add_mesh(stage: Usd.Stage, parent_path: str, name: str, stl_path: Path, collision: bool) -> str:
    loaded = trimesh.load_mesh(stl_path, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise RuntimeError(f"expected one mesh in {stl_path}")
    vertices = loaded.vertices.copy()
    if max(loaded.extents) > 1.0:
        vertices *= 0.001
    vertices -= (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    path = f"{parent_path}/{name}"
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(loaded.faces))
    mesh.CreateFaceVertexIndicesAttr(loaded.faces.reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--hardware-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-template", type=Path, required=True)
    args = parser.parse_args()

    source_dir = args.source_root / (
        "source/openarm/openarm/tasks/manager_based/openarm_manipulation/usds/openarm_bimanual"
    )
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    args.output_root.mkdir(parents=True, exist_ok=True)
    runtime_source = args.output_root / "source"
    if runtime_source.exists():
        shutil.rmtree(runtime_source)
    shutil.copytree(source_dir, runtime_source)
    # The root USD uses sibling `configuration/*.usd` payloads. Preserve that
    # composition layout beside the generated root layer as well as retaining
    # the untouched source copy for provenance and checksum inspection.
    runtime_configuration = args.output_root / "configuration"
    if runtime_configuration.exists():
        shutil.rmtree(runtime_configuration)
    shutil.copytree(source_dir / "configuration", runtime_configuration)
    shutil.copy2(args.config_template, args.output_root / "robot_config.yml")

    base = runtime_source / "openarm_bimanual.usd"
    output = args.output_root / "openarm_bimanual_cloth_folding.usd"
    stage = Usd.Stage.Open(str(base), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"could not open {base}")
    joint_paths = [shift_joint_anchor(stage, side, 0.05) for side in ("left", "right")]

    prims = list(stage.Traverse())
    prim_paths = {str(prim.GetPath()) for prim in prims}
    articulation_roots = [str(prim.GetPath()) for prim in prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    rigid_body_paths = [str(prim.GetPath()) for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    revolute_joint_paths = [str(prim.GetPath()) for prim in prims if prim.IsA(UsdPhysics.RevoluteJoint)]
    jaw_paths = []
    cover_paths = []
    for side in ("left", "right"):
        finger_candidates = sorted(
            path
            for path in prim_paths
            if path.startswith(f"/openarm/openarm_{side}_") and "finger" in path and path.count("/") == 2
        )
        if not finger_candidates:
            fingers = sorted(path for path in prim_paths if "finger" in path)
            raise RuntimeError(f"no {side} finger link found; finger paths={fingers}")
        for finger_path in finger_candidates:
            jaw_paths.append(
                add_mesh(stage, finger_path, "cloth_folding_jaw", args.hardware_root / "jaw_normal.stl", True)
            )
        upper_arm_candidates = sorted(
            path for path in prim_paths if f"openarm_{side}_link3" in path and "/joints/" not in path
        )
        if not upper_arm_candidates:
            raise RuntimeError(f"no {side} upper-arm link found")
        for label, filename in (
            ("extended_cover_front", "J3-J4_Cover front extended.stl"),
            ("extended_cover_back", "J3-J4_Cover back extended.stl"),
        ):
            cover_paths.append(add_mesh(stage, upper_arm_candidates[0], label, args.hardware_root / filename, True))
    stage.GetRootLayer().Export(str(output))

    sources = [base, *(args.hardware_root / name for name in (
        "J3-J4_Cover front extended.stl", "J3-J4_Cover back extended.stl", "jaw_normal.stl"
    ))]
    manifest = {
        "format": 1,
        "functional_twin": True,
        "upper_arm_extension_m": 0.05,
        "articulation_roots": articulation_roots,
        "rigid_body_paths": rigid_body_paths,
        "revolute_joint_paths": revolute_joint_paths,
        "joint_paths": joint_paths,
        "jaw_paths": jaw_paths,
        "cover_paths": cover_paths,
        "output": output.name,
        "sources": {path.name: sha256(path) for path in sources},
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    simulation_app.close()


if __name__ == "__main__":
    main()
