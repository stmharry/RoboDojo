#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

from isaacsim import SimulationApp
import numpy as np

simulation_app = SimulationApp({"headless": True})

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
import trimesh  # noqa: E402
import yaml  # noqa: E402


@dataclass(frozen=True)
class HolderFrame:
    filename: str
    mount_origin_mm: tuple[float, float, float]
    optical_origin_mm: tuple[float, float, float]
    optical_direction_cad: tuple[float, float, float]
    optical_up_cad: tuple[float, float, float]
    cad_to_mount: tuple[tuple[float, float, float], ...]

    def transformed_points_m(self, points) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        rotation = np.asarray(self.cad_to_mount, dtype=np.float64)
        origin = np.asarray(self.mount_origin_mm, dtype=np.float64)
        return (points - origin) @ rotation.T * 0.001

    def optical_frame_matrix(self) -> np.ndarray:
        rotation = np.asarray(self.cad_to_mount, dtype=np.float64)
        look = rotation @ np.asarray(self.optical_direction_cad, dtype=np.float64)
        look /= np.linalg.norm(look)
        up_hint = rotation @ np.asarray(self.optical_up_cad, dtype=np.float64)
        up_hint /= np.linalg.norm(up_hint)
        right = np.cross(look, up_hint)
        right /= np.linalg.norm(right)
        up = np.cross(right, look)
        up /= np.linalg.norm(up)
        optical_rotation = np.column_stack((right, up, -look))
        if not np.allclose(optical_rotation.T @ optical_rotation, np.eye(3), atol=1e-9):
            raise ValueError("camera optical basis is not orthonormal")
        if np.linalg.det(optical_rotation) < 0.999999:
            raise ValueError("camera optical basis is not right-handed")
        matrix = np.eye(4)
        matrix[:3, :3] = optical_rotation
        matrix[:3, 3] = self.transformed_points_m([self.optical_origin_mm])[0]
        return matrix


HEAD_CAD_TO_FIXTURE = np.asarray(((0.0, 1.0, 0.0), (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0)))
HEAD_HOLDER = HolderFrame(
    filename="head camera holder v4.stl",
    mount_origin_mm=(-17.56719649, 41.0, 262.79797363),
    optical_origin_mm=(49.40184321, 41.0, 19.09895447),
    optical_direction_cad=(-0.3420201433, 0.0, -0.9396926208),
    optical_up_cad=(0.9396926208, 0.0, -0.3420201433),
    cad_to_mount=tuple(tuple(float(value) for value in row) for row in HEAD_CAD_TO_FIXTURE),
)

WRIST_X_CAD = np.asarray((1.0, 0.0, 0.0))
WRIST_Z_CAD = np.asarray((0.0, -0.7660444431, 0.6427876097))
WRIST_CAD_TO_ANCHOR = np.vstack((WRIST_X_CAD, np.cross(WRIST_Z_CAD, WRIST_X_CAD), WRIST_Z_CAD))
WRIST_HOLDER = HolderFrame(
    filename="arducam_holder.stl",
    mount_origin_mm=(12.75, -42.62586212, -42.55709187),
    optical_origin_mm=(12.75, -6.21393824, 14.26977779),
    optical_direction_cad=(0.0, 1.0, 0.0),
    optical_up_cad=(1.0, 0.0, 0.0),
    cad_to_mount=tuple(tuple(float(value) for value in row) for row in WRIST_CAD_TO_ANCHOR),
)


def head_points_m(points) -> np.ndarray:
    return HEAD_HOLDER.transformed_points_m(points)


def wrist_points_m(points, side: str) -> np.ndarray:
    normalized = WRIST_HOLDER.transformed_points_m(points)
    if side == "right":
        normalized[:, 0] *= -1.0
    elif side != "left":
        raise ValueError(f"unknown OpenARM side: {side}")
    return normalized


def holder_optical_frame(side: str) -> np.ndarray:
    if side == "head":
        return HEAD_HOLDER.optical_frame_matrix()
    if side not in ("left", "right"):
        raise ValueError(f"unknown holder side: {side}")
    frame = WRIST_HOLDER.optical_frame_matrix()
    if side == "right":
        frame[0, 3] *= -1.0
        frame[:3, 0] *= -1.0
        frame[:3, 1] *= -1.0
    return frame


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


def build_holder_asset(stl_path: Path, output: Path, transform_points, prim_name: str, side: str) -> dict:
    """Author a holder in its CAD-derived attachment frame."""
    loaded = trimesh.load_mesh(stl_path, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise RuntimeError(f"expected one mesh in {stl_path}")
    vertices = transform_points(loaded.vertices)
    normalized = loaded.copy()
    normalized.vertices = vertices
    anchor_embedded = bool(normalized.contains([[0.0, 0.0, 0.0]])[0])
    stage = Usd.Stage.CreateNew(str(output))
    root = UsdGeom.Xform.Define(stage, f"/{prim_name}")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Xform.Define(stage, f"/{prim_name}/MountFrame")
    optical_frame = UsdGeom.Xform.Define(stage, f"/{prim_name}/OpticalFrame")
    optical_matrix = holder_optical_frame(side)
    xyzw = Rotation.from_matrix(optical_matrix[:3, :3]).as_quat()
    optical_xform = UsdGeom.Xformable(optical_frame)
    optical_xform.AddTranslateOp().Set(Gf.Vec3d(*optical_matrix[:3, 3].tolist()))
    optical_xform.AddOrientOp().Set(Gf.Quatf(float(xyzw[3]), *[float(value) for value in xyzw[:3]]))
    mesh = UsdGeom.Mesh.Define(stage, f"/{prim_name}/geometry")
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(loaded.faces))
    mesh.CreateFaceVertexIndicesAttr(loaded.faces.reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    stage.GetRootLayer().Save()
    return {
        "file": output.name,
        "prim": f"/{prim_name}",
        "mesh": f"/{prim_name}/geometry",
        "mount_frame": f"/{prim_name}/MountFrame",
        "optical_frame": f"/{prim_name}/OpticalFrame",
        "optical_frame_matrix": optical_matrix.tolist(),
        "collision": "convexHull",
        "attachment_anchor_embedded": anchor_embedded,
        "bounds_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--hardware-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    build_manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    asset_config = build_manifest["asset"]

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
    (args.output_root / "robot_config.yml").write_text(
        yaml.safe_dump(build_manifest["robot_config"], sort_keys=False), encoding="utf-8"
    )

    base = runtime_source / "openarm_bimanual.usd"
    output = args.output_root / asset_config["output"]
    stage = Usd.Stage.Open(str(base), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"could not open {base}")
    extension = float(asset_config["upper_arm_extension_m"])
    joint_paths = [shift_joint_anchor(stage, side, extension) for side in ("left", "right")]

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
                add_mesh(stage, finger_path, "extended_jaw", args.hardware_root / asset_config["jaw_mesh"], True)
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

    holder_assets = {
        "head": build_holder_asset(
            args.hardware_root / HEAD_HOLDER.filename,
            args.output_root / "head_camera_holder.usd",
            head_points_m,
            "head_camera_holder",
            "head",
        ),
        "left_wrist": build_holder_asset(
            args.hardware_root / WRIST_HOLDER.filename,
            args.output_root / "left_wrist_camera_holder.usd",
            lambda points: wrist_points_m(points, "left"),
            "left_wrist_camera_holder",
            "left",
        ),
        "right_wrist": build_holder_asset(
            args.hardware_root / WRIST_HOLDER.filename,
            args.output_root / "right_wrist_camera_holder.usd",
            lambda points: wrist_points_m(points, "right"),
            "right_wrist_camera_holder",
            "right",
        ),
    }

    sources = [
        base,
        *(
            args.hardware_root / name
            for name in (
                "J3-J4_Cover front extended.stl",
                "J3-J4_Cover back extended.stl",
                "jaw_normal.stl",
                HEAD_HOLDER.filename,
                "arducam_holder.step",
                WRIST_HOLDER.filename,
            )
        ),
    ]
    manifest = {
        "format": 1,
        "functional_twin": True,
        "upper_arm_extension_m": extension,
        "articulation_roots": articulation_roots,
        "rigid_body_paths": rigid_body_paths,
        "revolute_joint_paths": revolute_joint_paths,
        "joint_paths": joint_paths,
        "jaw_paths": jaw_paths,
        "cover_paths": cover_paths,
        "camera_holders": holder_assets,
        "output": output.name,
        "sources": {path.name: sha256(path) for path in sources},
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    simulation_app.close()


if __name__ == "__main__":
    main()
