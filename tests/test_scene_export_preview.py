from pathlib import Path
import zipfile

import pytest

pxr = pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils  # noqa: E402

from robodojo.sim.scene_export.preview import PREVIEW_NAME, create_blender_preview  # noqa: E402


def _mesh(stage: Usd.Stage, path: str) -> Usd.Prim:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    return mesh.GetPrim()


def _mdl_material(
    stage: Usd.Stage,
    path: str,
    *,
    sub_identifier: str,
    texture: str | None = None,
    misleading_preview_id: bool = False,
    opacity: float | None = None,
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.GetPrim().CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token).Set("sourceAsset")
    shader.GetPrim().CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("OmniPBR.mdl"))
    shader.GetPrim().CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token).Set(sub_identifier)
    if misleading_preview_id:
        shader.GetPrim().CreateAttribute("info:id", Sdf.ValueTypeNames.Token).Set("UsdPreviewSurface")
    if texture:
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
        shader.CreateInput("albedo_brightness", Sdf.ValueTypeNames.Float).Set(0.75)
        shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(2.0, 3.0))
    if opacity is not None:
        shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(opacity)
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.4)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _preview_material(stage: Usd.Stage, path: str, texture: str) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, f"{path}/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    reader = UsdShade.Shader.Define(stage, f"{path}/Reader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    image = UsdShade.Shader.Define(stage, f"{path}/Image")
    image.CreateIdAttr("UsdUVTexture")
    image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
    image.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    image.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    image.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(image.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    return material


def test_create_blender_preview_translates_and_packages_materials(tmp_path: Path):
    texture_dir = tmp_path / "textures"
    texture_dir.mkdir()
    (texture_dir / "base.png").write_bytes(b"fixture texture")

    source_path = tmp_path / "source.usdc"
    stage = Usd.Stage.CreateNew(str(source_path))
    UsdGeom.Xform.Define(stage, "/World")
    mdl_mesh = _mesh(stage, "/World/MDLMesh")
    UsdGeom.Xformable(mdl_mesh).AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    portable_mesh = _mesh(stage, "/World/PortableMesh")
    subset = UsdGeom.Subset.Define(stage, "/World/PortableMesh/MaterialSubset")
    subset.CreateElementTypeAttr().Set(UsdGeom.Tokens.face)
    subset.CreateFamilyNameAttr().Set(UsdShade.Tokens.materialBind)
    subset.CreateIndicesAttr().Set([0])
    malformed_mesh = _mesh(stage, "/World/MalformedMesh")
    missing_mesh = _mesh(stage, "/World/MissingTextureMesh")
    guide_mesh = _mesh(stage, "/World/GuideMesh")
    UsdGeom.Imageable(guide_mesh).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    UsdGeom.Xformable(camera).AddTranslateOp().Set(Gf.Vec3d(4.0, 5.0, 6.0))

    mdl = _mdl_material(
        stage,
        "/World/Looks/OmniPBR",
        sub_identifier="OmniPBR",
        texture="textures/base.png",
        opacity=0.35,
    )
    portable = _preview_material(stage, "/World/Looks/Portable", "textures/base.png")
    subset_material = _mdl_material(stage, "/World/Looks/Subset", sub_identifier="Aluminum_Brushed")
    malformed = _mdl_material(
        stage,
        "/World/Looks/Malformed",
        sub_identifier="Pure_White",
        misleading_preview_id=True,
    )
    missing = _mdl_material(
        stage,
        "/World/Looks/Missing",
        sub_identifier="OmniPBR",
        texture="textures/missing.png",
    )
    UsdShade.MaterialBindingAPI.Apply(mdl_mesh).Bind(mdl)
    UsdShade.MaterialBindingAPI.Apply(portable_mesh).Bind(portable)
    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(subset_material)
    UsdShade.MaterialBindingAPI.Apply(malformed_mesh).Bind(malformed)
    UsdShade.MaterialBindingAPI.Apply(missing_mesh).Bind(missing)
    UsdShade.MaterialBindingAPI.Apply(guide_mesh).Bind(mdl)
    stage.GetRootLayer().Save()

    diagnostics = create_blender_preview(stage, tmp_path, ["/World/Camera"])

    assert diagnostics["excluded_guide_meshes"] == 1
    assert diagnostics["preserved_materials"] == 1
    assert diagnostics["translated_materials"] == 4
    assert diagnostics["missing_textures"] == [
        {
            "material": "/World/Looks/Missing",
            "input": "diffuse_texture",
            "asset": "textures/missing.png",
        }
    ]

    packaged = Usd.Stage.Open(str(tmp_path / PREVIEW_NAME), load=Usd.Stage.LoadAll)
    assert packaged
    assert not packaged.GetPrimAtPath("/World/GuideMesh")
    for path in ("/World/MDLMesh", "/World/PortableMesh", "/World/MalformedMesh", "/World/MissingTextureMesh"):
        material, _ = UsdShade.MaterialBindingAPI(packaged.GetPrimAtPath(path)).ComputeBoundMaterial()
        output = material.GetSurfaceOutput()
        source = output.GetConnectedSource()
        assert source[0].GetPrim().GetAttribute("info:id").Get() == "UsdPreviewSurface"
        assert source[0].GetPrim().GetAttribute("info:implementationSource").Get() == "id"

    mdl_material, _ = UsdShade.MaterialBindingAPI(packaged.GetPrimAtPath("/World/MDLMesh")).ComputeBoundMaterial()
    mdl_surface = UsdShade.Shader(mdl_material.GetSurfaceOutput().GetConnectedSource()[0].GetPrim())
    assert mdl_surface.GetInput("opacity").Get() == pytest.approx(0.35)
    transform = next(
        UsdShade.Shader(prim)
        for prim in Usd.PrimRange(mdl_material.GetPrim())
        if prim.IsA(UsdShade.Shader) and prim.GetAttribute("info:id").Get() == "UsdTransform2d"
    )
    assert transform.GetInput("scale").Get() == Gf.Vec2f(2.0, 3.0)

    mesh_material, _ = UsdShade.MaterialBindingAPI(packaged.GetPrimAtPath("/World/PortableMesh")).ComputeBoundMaterial()
    bound_subset, _ = UsdShade.MaterialBindingAPI(
        packaged.GetPrimAtPath("/World/PortableMesh/MaterialSubset")
    ).ComputeBoundMaterial()
    assert mesh_material.GetPath() != bound_subset.GetPath()

    with zipfile.ZipFile(tmp_path / PREVIEW_NAME) as archive:
        members = archive.namelist()
    assert any(member.endswith("base.png") for member in members)
    assert not any(member.endswith(".mdl") for member in members)
    assert not (tmp_path / "scene_preview.usdc").exists()


def test_create_blender_preview_expands_instances_without_prototype_mdl(tmp_path: Path):
    (tmp_path / "OmniPBR.mdl").write_text("fixture mdl", encoding="utf-8")
    model_path = tmp_path / "model.usdc"
    model = Usd.Stage.CreateNew(str(model_path))
    root = UsdGeom.Xform.Define(model, "/Robot").GetPrim()
    model.SetDefaultPrim(root)
    visual = _mesh(model, "/Robot/VisualMesh")
    guide = _mesh(model, "/Robot/GuideMesh")
    UsdGeom.Imageable(guide).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    material = _mdl_material(model, "/Robot/Looks/Material", sub_identifier="OmniPBR")
    UsdShade.MaterialBindingAPI.Apply(visual).Bind(material)
    UsdShade.MaterialBindingAPI.Apply(guide).Bind(material)
    model.GetRootLayer().Save()

    composed_path = tmp_path / "composed.usdc"
    composed = Usd.Stage.CreateNew(str(composed_path))
    UsdGeom.Xform.Define(composed, "/World")
    for name, translation in (("RobotA", Gf.Vec3d(1.0, 0.0, 0.0)), ("RobotB", Gf.Vec3d(0.0, 2.0, 0.0))):
        instance = UsdGeom.Xform.Define(composed, f"/World/{name}")
        instance.GetPrim().GetReferences().AddReference("model.usdc")
        instance.GetPrim().SetInstanceable(True)
        instance.AddTranslateOp().Set(translation)
    composed.GetRootLayer().Save()

    flattened_path = tmp_path / "flattened.usdc"
    composed.Flatten().Export(str(flattened_path))
    flattened = Usd.Stage.Open(str(flattened_path), load=Usd.Stage.LoadAll)
    assert flattened.GetPrototypes()

    diagnostics = create_blender_preview(flattened, tmp_path, [])

    assert diagnostics["expanded_instances"] == 2
    assert diagnostics["excluded_guide_meshes"] == 2
    packaged = Usd.Stage.Open(str(tmp_path / PREVIEW_NAME), load=Usd.Stage.LoadAll)
    assert packaged and not packaged.GetPrototypes()
    assert not any(prim.IsInstance() for prim in packaged.Traverse())
    for name in ("RobotA", "RobotB"):
        visual_path = f"/World/{name}/VisualMesh"
        material, _ = UsdShade.MaterialBindingAPI(packaged.GetPrimAtPath(visual_path)).ComputeBoundMaterial()
        assert material.ComputeSurfaceSource("universal")[0].GetIdAttr().Get() == "UsdPreviewSurface"
        assert not packaged.GetPrimAtPath(f"/World/{name}/GuideMesh")

    with zipfile.ZipFile(tmp_path / PREVIEW_NAME) as archive:
        members = archive.namelist()
    assert not any(member.lower().endswith(".mdl") for member in members)
    assert [member for member in members if Path(member).suffix.lower() in {".usd", ".usda", ".usdc", ".usdz"}] == [
        "scene_preview.usdc"
    ]
    assert not (tmp_path / ".scene_preview_source.usdc").exists()


def test_create_blender_preview_materializes_packaged_textures(tmp_path: Path):
    texture_dir = tmp_path / "nested_textures"
    texture_dir.mkdir()
    texture = texture_dir / "base.png"
    texture.write_bytes(b"packaged fixture texture")

    nested_source_path = tmp_path / "nested_source.usdc"
    nested_source = Usd.Stage.CreateNew(str(nested_source_path))
    nested_root = UsdGeom.Xform.Define(nested_source, "/Asset").GetPrim()
    nested_source.SetDefaultPrim(nested_root)
    nested_mesh = _mesh(nested_source, "/Asset/Mesh")
    nested_material = _preview_material(nested_source, "/Asset/Looks/Material", "nested_textures/base.png")
    UsdShade.MaterialBindingAPI.Apply(nested_mesh).Bind(nested_material)
    nested_source.GetRootLayer().Save()
    nested_package = tmp_path / "nested.usdz"
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(nested_source_path)), str(nested_package))

    source_path = tmp_path / "source.usdc"
    stage = Usd.Stage.CreateNew(str(source_path))
    UsdGeom.Xform.Define(stage, "/World")
    mesh = _mesh(stage, "/World/Mesh")
    material = _preview_material(stage, "/World/Looks/Material", "nested.usdz[nested_textures/base.png]")
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material)
    stage.GetRootLayer().Save()

    diagnostics = create_blender_preview(stage, tmp_path, [])

    assert diagnostics["materialized_package_assets"] == 1
    with zipfile.ZipFile(tmp_path / PREVIEW_NAME) as archive:
        members = archive.namelist()
    assert any(member.startswith("preview_dependencies/") and member.endswith("base.png") for member in members)
    assert not any(member.endswith("nested.usdz") for member in members)
    assert not (tmp_path / "preview_dependencies").exists()
