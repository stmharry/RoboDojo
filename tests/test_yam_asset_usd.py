from pathlib import Path

import pytest

pxr = pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdGeom  # noqa: E402

from robodojo.workflows.assets_yam import (  # noqa: E402
    _remove_empty_generated_visual_prims,
    _validate_generated_visuals,
)


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
