import json
from pathlib import Path
import struct

import pytest
import yaml

pxr = pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from robodojo.core.paths import RepositoryPaths  # noqa: E402
from robodojo.workflows.asset_builders.moonlake_office.publication import author_fixture  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PATHS = RepositoryPaths.resolve(ROOT)


def _binary_triangle_stl(path: Path) -> None:
    header = b"moonlake test rail".ljust(80, b"\0")
    record = struct.pack(
        "<12fH",
        0.0,
        0.0,
        1.0,
        -500.0,
        -30.0,
        -10.0,
        500.0,
        -30.0,
        -10.0,
        0.0,
        30.0,
        10.0,
        0,
    )
    path.write_bytes(header + struct.pack("<I", 1) + record)


def test_generated_fixture_has_only_static_workspace_and_named_camera_frame(tmp_path: Path):
    tooling = yaml.safe_load(PATHS.moonlake_office_manifest.read_text(encoding="utf-8"))
    stl = tmp_path / "extrusion.stl"
    _binary_triangle_stl(stl)

    authored = author_fixture(stl, tmp_path / "output", tooling)
    output = tmp_path / "output" / authored["output"]
    stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)

    assert stage and str(stage.GetDefaultPrim().GetPath()) == "/MoonlakeOfficeFixture"
    assert stage.GetPrimAtPath("/MoonlakeOfficeFixture/TableFrame/LegFL").IsValid()
    rail = stage.GetPrimAtPath("/MoonlakeOfficeFixture/ArmRail/Extrusion2060")
    assert rail.IsValid() and rail.HasAPI(UsdPhysics.CollisionAPI)
    assert stage.GetPrimAtPath("/MoonlakeOfficeFixture/CameraStand/D435Assembly/Body").IsValid()
    optical = stage.GetPrimAtPath("/MoonlakeOfficeFixture/Mounts/D435OpticalFrame")
    assert optical.IsValid()

    expected_stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(expected_stage, "/Mounts")
    camera = tooling["fixture"]["top_camera"]
    view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*camera["body_center_m"]),
        Gf.Vec3d(*camera["look_at_target_m"]),
        Gf.Vec3d(*camera["up_axis"]),
    )
    UsdGeom.Xformable(parent).AddTransformOp().Set(view.GetInverse())
    expected_optical = UsdGeom.Xform.Define(expected_stage, "/Mounts/D435OpticalFrame")
    UsdGeom.Xformable(expected_optical).AddTranslateOp().Set(Gf.Vec3d(*camera["optical_translation_m"]))
    expected_transform = UsdGeom.XformCache().GetLocalToWorldTransform(expected_optical.GetPrim())
    actual_transform = UsdGeom.XformCache().GetLocalToWorldTransform(optical)
    assert actual_transform == expected_transform

    collision_paths = []
    for prim in stage.Traverse():
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert not prim.HasAPI(UsdPhysics.MassAPI)
        assert not prim.IsA(UsdPhysics.Joint)
        assert not prim.IsA(UsdGeom.Camera)
        assert not any(token in prim.GetName() for token in ("YAM", "Container", "TaskCube"))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_paths.append(str(prim.GetPath()))
        for attribute in prim.GetAttributes():
            assert not attribute.GetName().startswith("drive:")
    assert collision_paths == [
        "/MoonlakeOfficeFixture/TableFrame/LegFL",
        "/MoonlakeOfficeFixture/TableFrame/LegFR",
        "/MoonlakeOfficeFixture/TableFrame/LegBL",
        "/MoonlakeOfficeFixture/TableFrame/LegBR",
        "/MoonlakeOfficeFixture/ArmRail/Extrusion2060",
    ]

    metadata = json.loads((output.parent / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["physics"]["static"] is True
    assert metadata["geometry"]["mesh_triangles"] == 1
    assert metadata["geometry"]["aligned_bbox"]["extents"] == pytest.approx([1.13, 0.715, 1.55423322])
