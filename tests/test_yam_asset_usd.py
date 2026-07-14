from pathlib import Path

import pytest
import yaml

pxr = pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from robodojo.workflows.assets_yam import (  # noqa: E402
    _appearance_contract,
    _author_d405_visual_proxy,
    _author_preview_appearance,
    _remove_empty_generated_visual_prims,
    _stage_physics_digest,
    _validate_generated_visuals,
    _visual_proxy_contracts,
)

ROOT = Path(__file__).resolve().parents[1]
YAM_VISUAL_LINKS = ("base", "gripper", "link1", "link2", "link3", "link4", "link5", "tip_left", "tip_right")


def _generated_asset(
    root: Path,
    *,
    include_base_geometry: bool = True,
    include_root_geometry: bool = False,
) -> Path:
    output = root / "YAM.usda"
    stage = Usd.Stage.CreateNew(str(output))
    yam = UsdGeom.Xform.Define(stage, "/yam").GetPrim()
    stage.SetDefaultPrim(yam)
    UsdGeom.Xform.Define(stage, "/yam/root")
    root_visuals = UsdGeom.Xform.Define(stage, "/yam/root/visuals").GetPrim()
    root_visuals.GetReferences().AddInternalReference(Sdf.Path("/visuals/root"))
    UsdGeom.Xform.Define(stage, "/yam/base")
    base_visuals = UsdGeom.Xform.Define(stage, "/yam/base/visuals").GetPrim()
    base_visuals.GetReferences().AddInternalReference(Sdf.Path("/visuals/base"))
    UsdGeom.Scope.Define(stage, "/visuals")
    if include_base_geometry:
        UsdGeom.Xform.Define(stage, "/visuals/base")
        UsdGeom.Mesh.Define(stage, "/visuals/base/Mesh")
    if include_root_geometry:
        UsdGeom.Xform.Define(stage, "/visuals/root")
        UsdGeom.Mesh.Define(stage, "/visuals/root/Mesh")
    stage.GetRootLayer().Save()
    return output


def test_empty_dangling_nonvisual_prim_is_removed_without_touching_real_visuals(tmp_path: Path):
    output = _generated_asset(tmp_path)

    removed = _remove_empty_generated_visual_prims(output, tmp_path, ["root"])

    assert removed == ["/yam/root/visuals"]
    stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
    assert stage and not stage.GetPrimAtPath("/yam/root/visuals")
    assert stage.GetPrimAtPath("/yam/base/visuals/Mesh")
    assert _validate_generated_visuals(stage, ["base"], ["root"]) == ["/yam/base/visuals/Mesh"]


def test_nonvisual_link_with_resolved_geometry_is_rejected(tmp_path: Path):
    output = _generated_asset(tmp_path, include_root_geometry=True)

    with pytest.raises(RuntimeError, match="refusing to remove resolved visual reference for root"):
        _remove_empty_generated_visual_prims(output, tmp_path, ["root"])


def test_expected_visual_link_without_renderable_geometry_is_rejected(tmp_path: Path):
    output = _generated_asset(tmp_path, include_base_geometry=False)
    _remove_empty_generated_visual_prims(output, tmp_path, ["root"])
    stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)

    with pytest.raises(RuntimeError, match="missing renderable geometry below /yam/base/visuals"):
        _validate_generated_visuals(stage, ["base"], ["root"])


def _appearance_stage(path: Path) -> tuple[Usd.Stage, list[str]]:
    stage = Usd.Stage.CreateNew(str(path))
    yam = UsdGeom.Xform.Define(stage, "/yam").GetPrim()
    stage.SetDefaultPrim(yam)
    visual_paths = []
    for link in YAM_VISUAL_LINKS:
        link_prim = UsdGeom.Xform.Define(stage, f"/yam/{link}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(link_prim)
        UsdGeom.Xform.Define(stage, f"/yam/{link}/visuals")
        visual = UsdGeom.Mesh.Define(stage, f"/yam/{link}/visuals/{link}_mesh").GetPrim()
        visual_paths.append(str(visual.GetPath()))
        UsdGeom.Xform.Define(stage, f"/yam/{link}/collisions")
        collision = UsdGeom.Mesh.Define(stage, f"/yam/{link}/collisions/{link}_mesh").GetPrim()
        UsdPhysics.CollisionAPI.Apply(collision)
    UsdPhysics.RevoluteJoint.Define(stage, "/yam/dof_joint1")
    return stage, visual_paths


def test_preview_appearance_is_deterministic_and_render_only(tmp_path: Path):
    tooling = yaml.safe_load((ROOT / "configs/tooling/yam.yml").read_text())
    appearance = _appearance_contract(tooling, list(YAM_VISUAL_LINKS))
    stage, visual_paths = _appearance_stage(tmp_path / "appearance.usda")
    physics_before = _stage_physics_digest(stage)

    generated = _author_preview_appearance(stage, appearance, visual_paths)

    assert _stage_physics_digest(stage) == physics_before
    assert [binding["target"] for binding in generated["bindings"]] == [
        f"/yam/{link}/visuals" for link in sorted(YAM_VISUAL_LINKS)
    ]
    assert generated["link_materials"] == appearance["link_materials"]
    assert {
        name: material["sha256"] for name, material in generated["palette"].items()
    } == {name: material["sha256"] for name, material in appearance["palette"].items()}

    bound_targets = set()
    for binding in generated["bindings"]:
        for renderable_path in binding["renderable_paths"]:
            prim = stage.GetPrimAtPath(renderable_path)
            bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            assert str(bound.GetPath()) == f"/yam/Looks/{binding['material']}"
            display_color = UsdGeom.PrimvarsAPI(prim).FindPrimvarWithInheritance("displayColor").Get()
            assert tuple(display_color[0]) == pytest.approx(
                appearance["palette"][binding["material"]]["diffuse_color"]
            )
            bound_targets.add(str(prim.GetPath()))
    assert bound_targets == set(visual_paths)

    for link in YAM_VISUAL_LINKS:
        collision = stage.GetPrimAtPath(f"/yam/{link}/collisions/{link}_mesh")
        assert collision.HasAPI(UsdPhysics.CollisionAPI)
        assert not UsdShade.MaterialBindingAPI(collision).ComputeBoundMaterial()[0]

    for name, parameters in appearance["palette"].items():
        shader = UsdShade.Shader(stage.GetPrimAtPath(f"/yam/Looks/{name}/PreviewSurface"))
        assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
        assert tuple(shader.GetInput("diffuseColor").Get()) == pytest.approx(parameters["diffuse_color"])
        assert shader.GetInput("roughness").Get() == pytest.approx(parameters["roughness"])
        assert shader.GetInput("metallic").Get() == pytest.approx(parameters["metallic"])

    stage.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(tmp_path / "appearance.usda"), load=Usd.Stage.LoadAll)
    assert reopened and _stage_physics_digest(reopened) == physics_before


def test_d405_proxy_is_deterministic_identity_optical_frame_without_physics(tmp_path: Path):
    tooling = yaml.safe_load((ROOT / "configs/tooling/yam.yml").read_text())
    appearance = _appearance_contract(tooling, list(YAM_VISUAL_LINKS))
    contract = _visual_proxy_contracts(tooling, appearance)["d405"]

    first = _author_d405_visual_proxy(tmp_path, contract, appearance)
    first_bytes = (tmp_path / "D405_proxy.usd").read_bytes()
    second = _author_d405_visual_proxy(tmp_path, contract, appearance)

    assert first == second
    assert (tmp_path / "D405_proxy.usd").read_bytes() == first_bytes
    assert first["sha256"] == "9023afd80b23366f330a91944192fc0f70440bc5f1e693f039fc1944e3e7c74b"
    assert first["contract_sha256"] == "3c15307252439ebc8635cf0117371adc5d520a21b394e76f94cf12932395920d"
    assert first["visual_paths"] == [
        "/D405/FrontPanel",
        "/D405/Housing",
        "/D405/LeftLens",
        "/D405/RightLens",
    ]

    stage = Usd.Stage.Open(str(tmp_path / "D405_proxy.usd"), load=Usd.Stage.LoadAll)
    assert stage and str(stage.GetDefaultPrim().GetPath()) == "/D405"
    optical_frame = stage.GetPrimAtPath("/D405/OpticalFrame")
    assert optical_frame.IsValid()
    assert not UsdGeom.Xformable(optical_frame).GetOrderedXformOps()
    housing_ops = UsdGeom.Xformable(stage.GetPrimAtPath("/D405/Housing")).GetOrderedXformOps()
    assert tuple(housing_ops[0].Get()) == pytest.approx((0.0, 0.0, 0.0115))
    assert tuple(housing_ops[1].Get()) == pytest.approx((0.042, 0.042, 0.023))
    for prim in stage.Traverse():
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert not prim.HasAPI(UsdPhysics.CollisionAPI)
        assert not prim.HasAPI(UsdPhysics.MassAPI)
        assert not prim.IsA(UsdPhysics.Joint)

    referenced = Usd.Stage.CreateInMemory()
    referenced.DefinePrim("/Holder", "Xform").GetReferences().AddReference(
        str(tmp_path / "D405_proxy.usd")
    )
    assert referenced.GetPrimAtPath("/Holder/OpticalFrame").IsValid()
