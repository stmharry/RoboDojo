"""Create a portable, Blender-oriented companion for an Isaac USD snapshot."""

from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath
from typing import Any
import zipfile

from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdShade, UsdUtils

from robodojo.sim.scene_export.contracts import package_member_exists, split_package_asset_path

logger = logging.getLogger(__name__)

PREVIEW_NAME = "scene_preview.usdz"
_PREVIEW_ROOT_NAME = "scene_preview.usdc"
_PREVIEW_SOURCE_ROOT_NAME = ".scene_preview_source.usdc"
_PREVIEW_SCOPE = Sdf.Path("/World/RoboDojoPreviewLooks")
_ALLOWED_SHADER_IDS = {
    "UsdPreviewSurface",
    "UsdUVTexture",
    "UsdPrimvarReader_float2",
    "UsdTransform2d",
}
_USD_EXTENSIONS = {".usd", ".usda", ".usdc", ".usdz"}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _float_value(shader: UsdShade.Shader, name: str, default: float) -> float:
    value = shader.GetInput(name).Get()
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _bool_value(shader: UsdShade.Shader, name: str, default: bool = False) -> bool:
    value = shader.GetInput(name).Get()
    return bool(value) if value is not None else default


def _vec2_value(shader: UsdShade.Shader, name: str, default: tuple[float, float]) -> Gf.Vec2f:
    value = shader.GetInput(name).Get()
    try:
        return Gf.Vec2f(float(value[0]), float(value[1])) if value is not None else Gf.Vec2f(*default)
    except (IndexError, TypeError, ValueError):
        return Gf.Vec2f(*default)


def _color_value(
    shader: UsdShade.Shader,
    name: str,
    default: tuple[float, float, float],
) -> Gf.Vec3f:
    value = shader.GetInput(name).Get()
    try:
        return Gf.Vec3f(float(value[0]), float(value[1]), float(value[2])) if value is not None else Gf.Vec3f(*default)
    except (IndexError, TypeError, ValueError):
        return Gf.Vec3f(*default)


def _asset_value(shader: UsdShade.Shader, name: str) -> Sdf.AssetPath | None:
    value = shader.GetInput(name).Get()
    return value if isinstance(value, Sdf.AssetPath) and value.path else None


def _asset_resolves(stage: Usd.Stage, asset: Sdf.AssetPath) -> bool:
    if asset.resolvedPath:
        return True
    path, member = split_package_asset_path(asset.path)
    if not path or "://" in path:
        return False
    candidate = Path(path)
    if not candidate.is_absolute():
        root_path, _ = split_package_asset_path(stage.GetRootLayer().realPath)
        candidate = Path(root_path).parent / candidate
    if not candidate.is_file():
        return False
    return not member or package_member_exists(candidate, member)


def _all_gprims(stage: Usd.Stage, *, instance_proxies: bool = False):
    predicate = Usd.TraverseInstanceProxies() if instance_proxies else Usd.PrimDefaultPredicate
    for prim in Usd.PrimRange.Stage(stage, predicate):
        if prim.IsA(UsdGeom.Gprim):
            yield prim


def _expand_instances(stage: Usd.Stage, preview_root: Path) -> tuple[Usd.Stage, int]:
    """Bake instance proxies into the detached preview without prototype residue."""
    instance_paths = [prim.GetPath() for prim in stage.Traverse() if prim.IsInstance()]
    if not instance_paths:
        stage.GetRootLayer().Export(str(preview_root))
        expanded_stage = Usd.Stage.Open(str(preview_root), load=Usd.Stage.LoadAll)
        if not expanded_stage:
            raise RuntimeError("could not reopen detached preview stage")
        return expanded_stage, 0

    expected_transforms = {
        str(prim.GetPath()): UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for prim in _all_gprims(stage, instance_proxies=True)
    }
    for path in instance_paths:
        stage.GetPrimAtPath(path).SetInstanceable(False)

    expanded_layer = stage.Flatten()
    expanded_stage = Usd.Stage.Open(expanded_layer, load=Usd.Stage.LoadAll)
    if not expanded_stage:
        raise RuntimeError("could not open expanded preview stage")
    for prim_spec in list(expanded_layer.rootPrims):
        if prim_spec.name.startswith("Flattened_Prototype_"):
            expanded_stage.RemovePrim(prim_spec.path)

    output_dir = preview_root.parent.resolve()

    def make_relative(path: str) -> str:
        outer_path, package_member = split_package_asset_path(path)
        candidate = Path(outer_path)
        if not candidate.is_absolute():
            return path
        try:
            relative = candidate.resolve().relative_to(output_dir).as_posix()
        except ValueError:
            return path
        suffix = f"[{package_member}]" if package_member else ""
        return relative + suffix

    UsdUtils.ModifyAssetPaths(expanded_layer, make_relative)
    expanded_layer.Export(str(preview_root))
    expanded_stage = Usd.Stage.Open(str(preview_root), load=Usd.Stage.LoadAll)
    if not expanded_stage:
        raise RuntimeError("could not reopen expanded preview stage")
    if expanded_stage.GetPrototypes() or any(prim.IsInstance() for prim in expanded_stage.Traverse()):
        raise RuntimeError("portable preview retained USD instances or prototypes")
    for path, expected in expected_transforms.items():
        prim = expanded_stage.GetPrimAtPath(path)
        if not prim:
            raise RuntimeError(f"expanded preview is missing instanced geometry: {path}")
        actual = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        if not Gf.IsClose(actual, expected, 1e-7):
            raise RuntimeError(f"expanded preview geometry transform mismatch: {path}")
    return expanded_stage, len(instance_paths)


def _validate_package_members(package_path: Path) -> None:
    """Reject hidden source networks and unsafe paths inside the USDZ archive."""
    try:
        with zipfile.ZipFile(package_path) as archive:
            members = archive.namelist()
    except (OSError, zipfile.BadZipFile) as error:
        raise RuntimeError(f"could not inspect Blender preview USDZ: {error}") from error
    errors = []
    if _PREVIEW_ROOT_NAME not in members:
        errors.append(f"missing-root:{_PREVIEW_ROOT_NAME}")
    for name in members:
        member = PurePosixPath(name)
        if member.is_absolute() or ".." in member.parts or "://" in name:
            errors.append(f"unsafe-path:{name}")
        suffix = member.suffix.lower()
        if suffix == ".mdl":
            errors.append(f"mdl:{name}")
        if suffix in _USD_EXTENSIONS and name != _PREVIEW_ROOT_NAME:
            errors.append(f"external-usd:{name}")
    if errors:
        raise RuntimeError(f"portable preview package validation failed: {errors[:20]}")


def _display_color(prim: Usd.Prim) -> Gf.Vec3f | None:
    if not prim.IsA(UsdGeom.Gprim):
        return None
    values = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
    if not values:
        return None
    return Gf.Vec3f(*[float(component) for component in values[0]])


def _surface_source(material: UsdShade.Material) -> UsdShade.Shader | None:
    output = material.GetSurfaceOutput()
    source = output.GetConnectedSource() if output else None
    return UsdShade.Shader(source[0].GetPrim()) if source else None


def _portable_surface(material: UsdShade.Material) -> UsdShade.Shader | None:
    shader = _surface_source(material)
    if not shader:
        return None
    prim = shader.GetPrim()
    if prim.GetAttribute("info:id").Get() != "UsdPreviewSurface":
        return None
    if prim.GetAttribute("info:implementationSource").Get() != "id":
        return None
    return shader


def _source_asset_shader(material: UsdShade.Material) -> UsdShade.Shader | None:
    for prim in Usd.PrimRange(material.GetPrim()):
        if prim.IsA(UsdShade.Shader) and prim.GetAttribute("info:mdl:sourceAsset").Get():
            return UsdShade.Shader(prim)
    return None


def _create_surface(
    stage: Usd.Stage,
    path: Sdf.Path,
    *,
    diffuse: Gf.Vec3f = Gf.Vec3f(0.18),
    roughness: float = 0.5,
    metallic: float = 0.0,
    opacity: float = 1.0,
    opacity_threshold: float = 0.0,
    ior: float = 1.5,
) -> tuple[UsdShade.Material, UsdShade.Shader]:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendChild("PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(diffuse)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(_clamp(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(_clamp(metallic))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(_clamp(opacity))
    shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(_clamp(opacity_threshold))
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(float(ior))
    shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).Set(Gf.Vec3f(0.0, 0.0, 1.0))
    shader.CreateInput("occlusion", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0))
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material, shader


def _copy_portable_network(
    stage: Usd.Stage,
    source_material: UsdShade.Material,
    target_path: Sdf.Path,
    diagnostics: dict[str, Any],
) -> UsdShade.Material | None:
    shader_prims = [prim for prim in Usd.PrimRange(source_material.GetPrim()) if prim.IsA(UsdShade.Shader)]
    shader_ids = {prim.GetAttribute("info:id").Get() for prim in shader_prims}
    unsupported = sorted(str(shader_id) for shader_id in shader_ids if shader_id not in _ALLOWED_SHADER_IDS)
    if unsupported:
        diagnostics["unsupported_inputs"].append(
            {"material": str(source_material.GetPath()), "shader_ids": unsupported}
        )
        return None

    for prim in shader_prims:
        for shader_input in UsdShade.Shader(prim).GetInputs():
            value = shader_input.Get()
            if isinstance(value, Sdf.AssetPath) and value.path and not _asset_resolves(stage, value):
                diagnostics["missing_textures"].append(
                    {
                        "material": str(source_material.GetPath()),
                        "input": str(shader_input.GetAttr().GetPath()),
                        "asset": value.path,
                    }
                )
                return None

    target_material = UsdShade.Material.Define(stage, target_path)
    source_root = source_material.GetPath()
    shader_map: dict[str, UsdShade.Shader] = {}
    for prim in shader_prims:
        relative = prim.GetPath().MakeRelativePath(source_root)
        target_shader = UsdShade.Shader.Define(stage, target_path.AppendPath(relative))
        target_shader.CreateIdAttr(prim.GetAttribute("info:id").Get())
        source_shader = UsdShade.Shader(prim)
        for output in source_shader.GetOutputs():
            target_shader.CreateOutput(output.GetBaseName(), output.GetTypeName())
        for shader_input in source_shader.GetInputs():
            target_input = target_shader.CreateInput(shader_input.GetBaseName(), shader_input.GetTypeName())
            value = shader_input.Get()
            if value is not None:
                target_input.Set(value)
        shader_map[str(prim.GetPath())] = target_shader

    for prim in shader_prims:
        source_shader = UsdShade.Shader(prim)
        target_shader = shader_map[str(prim.GetPath())]
        for shader_input in source_shader.GetInputs():
            connection = shader_input.GetConnectedSource()
            if not connection:
                continue
            source_path = str(connection[0].GetPrim().GetPath())
            target_source = shader_map.get(source_path)
            if target_source is None:
                diagnostics["unsupported_inputs"].append(
                    {"material": str(source_material.GetPath()), "connection": str(shader_input.GetAttr().GetPath())}
                )
                stage.RemovePrim(target_path)
                return None
            target_shader.GetInput(shader_input.GetBaseName()).ConnectToSource(
                target_source.ConnectableAPI(), connection[1], connection[2]
            )

    source_surface = _portable_surface(source_material)
    target_surface = shader_map[str(source_surface.GetPath())]
    target_material.CreateSurfaceOutput().ConnectToSource(target_surface.ConnectableAPI(), "surface")
    return target_material


def _translate_omnipbr(
    stage: Usd.Stage,
    source_shader: UsdShade.Shader,
    target_path: Sdf.Path,
    display_color: Gf.Vec3f | None,
    diagnostics: dict[str, Any],
) -> UsdShade.Material:
    base = _color_value(source_shader, "diffuse_color_constant", (0.2, 0.2, 0.2))
    tint = _color_value(source_shader, "diffuse_tint", (1.0, 1.0, 1.0))
    brightness = _float_value(source_shader, "albedo_brightness", 1.0)
    add = _float_value(source_shader, "albedo_add", 0.0)
    diffuse = Gf.Vec3f(*[_clamp(float(base[i]) * float(tint[i]) * brightness + add) for i in range(3)])
    roughness = _float_value(source_shader, "reflection_roughness_constant", 0.5)
    metallic = _float_value(source_shader, "metallic_constant", 0.0)
    enable_opacity = _bool_value(source_shader, "enable_opacity")
    opacity = _float_value(source_shader, "opacity_constant", 1.0) if enable_opacity else 1.0
    opacity_threshold = _float_value(source_shader, "opacity_threshold", 0.0) if enable_opacity else 0.0
    material, surface = _create_surface(
        stage,
        target_path,
        diffuse=diffuse,
        roughness=roughness,
        metallic=metallic,
        opacity=opacity,
        opacity_threshold=opacity_threshold,
    )

    reader = UsdShade.Shader.Define(stage, target_path.AppendChild("PrimvarReader"))
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    transform = UsdShade.Shader.Define(stage, target_path.AppendChild("Transform2d"))
    transform.CreateIdAttr("UsdTransform2d")
    transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(
        _vec2_value(source_shader, "texture_scale", (1.0, 1.0))
    )
    transform.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set(
        _vec2_value(source_shader, "texture_translate", (0.0, 0.0))
    )
    transform.CreateInput("rotation", Sdf.ValueTypeNames.Float).Set(_float_value(source_shader, "texture_rotate", 0.0))
    transform.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    texture_index = 0

    def connect_texture(
        source_name: str,
        surface_name: str,
        color_space: str,
        channel: str,
        *,
        scale: Gf.Vec4f = Gf.Vec4f(1.0),
        bias: Gf.Vec4f = Gf.Vec4f(0.0),
        fallback: Any = None,
    ) -> bool:
        nonlocal texture_index
        asset = _asset_value(source_shader, source_name)
        if asset is None:
            return False
        if not _asset_resolves(stage, asset):
            diagnostics["missing_textures"].append(
                {
                    "material": str(source_shader.GetPrim().GetParent().GetPath()),
                    "input": source_name,
                    "asset": asset.path,
                }
            )
            if fallback is not None:
                surface.GetInput(surface_name).Set(fallback)
            return False
        texture_index += 1
        texture = UsdShade.Shader.Define(stage, target_path.AppendChild(f"Texture_{texture_index:02d}"))
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(asset)
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(transform.ConnectableAPI(), "result")
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(scale)
        texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(bias)
        output_type = Sdf.ValueTypeNames.Float3 if channel == "rgb" else Sdf.ValueTypeNames.Float
        texture.CreateOutput(channel, output_type)
        surface.GetInput(surface_name).ConnectToSource(texture.ConnectableAPI(), channel)
        return True

    diffuse_scale = Gf.Vec4f(
        float(tint[0]) * brightness,
        float(tint[1]) * brightness,
        float(tint[2]) * brightness,
        1.0,
    )
    connect_texture(
        "diffuse_texture",
        "diffuseColor",
        "sRGB",
        "rgb",
        scale=diffuse_scale,
        bias=Gf.Vec4f(add, add, add, 0.0),
        fallback=display_color if display_color is not None else Gf.Vec3f(0.18),
    )

    roughness_influence = _clamp(_float_value(source_shader, "reflection_roughness_texture_influence", 0.0))
    connect_texture(
        "reflectionroughness_texture",
        "roughness",
        "raw",
        "r",
        scale=Gf.Vec4f(roughness_influence),
        bias=Gf.Vec4f(roughness * (1.0 - roughness_influence)),
        fallback=0.5,
    )
    metallic_influence = _clamp(_float_value(source_shader, "metallic_texture_influence", 0.0))
    connect_texture(
        "metallic_texture",
        "metallic",
        "raw",
        "r",
        scale=Gf.Vec4f(metallic_influence),
        bias=Gf.Vec4f(metallic * (1.0 - metallic_influence)),
        fallback=0.0,
    )
    normal_strength = _float_value(source_shader, "normalmap_strength", 1.0)
    connect_texture(
        "normalmap_texture",
        "normal",
        "raw",
        "rgb",
        scale=Gf.Vec4f(2.0 * normal_strength, 2.0 * normal_strength, 2.0, 1.0),
        bias=Gf.Vec4f(-normal_strength, -normal_strength, -1.0, 0.0),
        fallback=Gf.Vec3f(0.0, 0.0, 1.0),
    )
    connect_texture("ao_texture", "occlusion", "raw", "r", fallback=1.0)
    if enable_opacity and _bool_value(source_shader, "enable_opacity_texture"):
        connect_texture("opacity_texture", "opacity", "raw", "r", fallback=1.0)
    if _bool_value(source_shader, "enable_emission"):
        emissive = _color_value(source_shader, "emissive_color", (0.0, 0.0, 0.0))
        intensity = _float_value(source_shader, "emissive_intensity", 1.0)
        surface.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*[_clamp(float(component) * intensity) for component in emissive])
        )
        connect_texture("emissive_color_texture", "emissiveColor", "sRGB", "rgb")
    return material


def _translated_material(
    stage: Usd.Stage,
    source_material: UsdShade.Material,
    target_path: Sdf.Path,
    display_color: Gf.Vec3f | None,
    diagnostics: dict[str, Any],
) -> tuple[UsdShade.Material, str]:
    if _portable_surface(source_material):
        copied = _copy_portable_network(stage, source_material, target_path, diagnostics)
        if copied:
            return copied, "preserved"

    source_shader = _source_asset_shader(source_material)
    sub_identifier = ""
    if source_shader:
        sub_identifier = str(source_shader.GetPrim().GetAttribute("info:mdl:sourceAsset:subIdentifier").Get() or "")
    if source_shader and sub_identifier == "OmniPBR":
        return _translate_omnipbr(stage, source_shader, target_path, display_color, diagnostics), "translated"
    if sub_identifier == "Pure_White":
        material, _ = _create_surface(stage, target_path, diffuse=Gf.Vec3f(1.0), roughness=1.0)
        return material, "translated"
    if sub_identifier == "OmniGlass":
        roughness = _float_value(source_shader, "frosting_roughness", 0.0)
        ior = _float_value(source_shader, "glass_ior", 1.491)
        color = _color_value(source_shader, "glass_color", (1.0, 1.0, 1.0))
        material, _ = _create_surface(stage, target_path, diffuse=color, roughness=roughness, opacity=0.2, ior=ior)
        return material, "translated"
    if sub_identifier.startswith("Aluminum_"):
        material, _ = _create_surface(stage, target_path, diffuse=Gf.Vec3f(0.6), roughness=0.35, metallic=1.0)
        return material, "translated"
    if sub_identifier.startswith("Plastic_"):
        material, _ = _create_surface(stage, target_path, diffuse=Gf.Vec3f(0.05), roughness=0.45)
        return material, "translated"

    fallback = display_color if display_color is not None else Gf.Vec3f(0.18)
    material, _ = _create_surface(stage, target_path, diffuse=fallback)
    diagnostics["unsupported_inputs"].append(
        {"material": str(source_material.GetPath()), "source_asset_subidentifier": sub_identifier or None}
    )
    return material, "fallback"


def _remove_guide_geometry(stage: Usd.Stage) -> list[str]:
    paths = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Gprim) and UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide:
            paths.append(prim.GetPath())
    for path in sorted(paths, key=lambda item: item.pathElementCount, reverse=True):
        stage.RemovePrim(path)
    return [str(path) for path in paths]


def _remove_original_shading(stage: Usd.Stage) -> None:
    paths = []
    for prim in stage.Traverse():
        if prim.GetPath().HasPrefix(_PREVIEW_SCOPE):
            continue
        if prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader):
            paths.append(prim.GetPath())
    for path in sorted(paths, key=lambda item: item.pathElementCount):
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
    for prim in stage.Traverse():
        if prim.GetPath().HasPrefix(_PREVIEW_SCOPE):
            continue
        for relationship in list(prim.GetRelationships()):
            if relationship.GetName().startswith("material:binding"):
                targets = relationship.GetTargets()
                if not any(target.HasPrefix(_PREVIEW_SCOPE) for target in targets):
                    prim.RemoveProperty(relationship.GetName())


def _is_renderable(prim: Usd.Prim) -> bool:
    if not prim.IsA(UsdGeom.Gprim):
        return False
    imageable = UsdGeom.Imageable(prim)
    return imageable.ComputeVisibility() != UsdGeom.Tokens.invisible


def _validate_preview(stage: Usd.Stage) -> None:
    errors = []
    if stage.GetPrototypes():
        errors.append(f"prototypes:{len(stage.GetPrototypes())}")
    for prim in stage.Traverse():
        if prim.IsInstance():
            errors.append(f"instance:{prim.GetPath()}")
        if prim.IsA(UsdGeom.Gprim):
            if UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide:
                errors.append(f"guide:{prim.GetPath()}")
            if _is_renderable(prim):
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                if not material or not _portable_surface(material):
                    errors.append(f"material:{prim.GetPath()}")
        if prim.IsA(UsdShade.Shader):
            shader_id = prim.GetAttribute("info:id").Get()
            if shader_id not in _ALLOWED_SHADER_IDS:
                errors.append(f"shader:{prim.GetPath()}:{shader_id}")
        for attribute in prim.GetAttributes():
            value = attribute.Get()
            if not isinstance(value, Sdf.AssetPath) or not value.path:
                continue
            path, _ = split_package_asset_path(value.path)
            if path.lower().endswith(".mdl") or Path(path).is_absolute() or "://" in path:
                errors.append(f"asset:{attribute.GetPath()}:{value.path}")
            elif not _asset_resolves(stage, value):
                errors.append(f"unresolved:{attribute.GetPath()}:{value.path}")
    if errors:
        raise RuntimeError(f"portable preview validation failed: {errors[:20]}")


def create_blender_preview(
    source_stage: Usd.Stage,
    output_dir: Path,
    camera_paths: list[str],
) -> dict[str, Any]:
    """Create and validate ``scene_preview.usdz`` beside the canonical export."""
    source_root = output_dir / _PREVIEW_SOURCE_ROOT_NAME
    preview_root = output_dir / _PREVIEW_ROOT_NAME
    preview_path = output_dir / PREVIEW_NAME
    source_stage.GetRootLayer().Export(str(source_root))
    stage = Usd.Stage.Open(str(source_root), load=Usd.Stage.LoadAll)
    if not stage:
        raise RuntimeError("could not open detached preview stage")
    stage, expanded_instances = _expand_instances(stage, preview_root)

    source_camera_transforms = {}
    source_gprim_transforms = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Gprim) and UsdGeom.Imageable(prim).ComputePurpose() != UsdGeom.Tokens.guide:
            source_gprim_transforms[str(prim.GetPath())] = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
    for path in camera_paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsA(UsdGeom.Camera):
            source_camera_transforms[path] = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )

    excluded = _remove_guide_geometry(stage)
    UsdGeom.Scope.Define(stage, _PREVIEW_SCOPE)
    diagnostics: dict[str, Any] = {
        "preserved_materials": 0,
        "translated_materials": 0,
        "fallback_materials": 0,
        "missing_textures": [],
        "unsupported_inputs": [],
        "expanded_instances": expanded_instances,
        "excluded_guide_meshes": len(excluded),
        "excluded_guide_paths": excluded,
    }
    material_map: dict[str, UsdShade.Material] = {}
    targets: list[tuple[Usd.Prim, str, UsdShade.Material | None, Gf.Vec3f | None]] = []
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdGeom.Gprim):
            if not _is_renderable(prim):
                continue
        elif not prim.IsA(UsdGeom.Subset):
            continue
        bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        key = str(bound.GetPath()) if bound else f"__unbound__:{prim.GetPath()}"
        display_prim = prim if prim.IsA(UsdGeom.Gprim) else prim.GetParent()
        targets.append((prim, key, bound if bound else None, _display_color(display_prim)))

    for index, (_, key, source_material, display_color) in enumerate(targets):
        if key in material_map:
            continue
        source_name = source_material.GetPath().name if source_material else "Unbound"
        name = Tf.MakeValidIdentifier(f"M_{index:04d}_{source_name}")
        target_path = _PREVIEW_SCOPE.AppendChild(name)
        if source_material:
            material, status = _translated_material(stage, source_material, target_path, display_color, diagnostics)
        else:
            fallback = display_color if display_color is not None else Gf.Vec3f(0.18)
            material, _ = _create_surface(stage, target_path, diffuse=fallback)
            status = "fallback"
        diagnostics[f"{status}_materials"] += 1
        material_map[key] = material

    for prim, key, _, _ in targets:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material_map[key])
    _remove_original_shading(stage)
    stage.GetRootLayer().Save()
    _validate_preview(stage)

    created = UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(preview_root)), str(preview_path))
    if not created or not preview_path.is_file():
        raise RuntimeError("failed to create Blender preview USDZ package")
    _validate_package_members(preview_path)
    packaged = Usd.Stage.Open(str(preview_path), load=Usd.Stage.LoadAll)
    if not packaged:
        raise RuntimeError("could not reopen Blender preview USDZ package")
    _validate_preview(packaged)

    for path, expected in source_gprim_transforms.items():
        prim = packaged.GetPrimAtPath(path)
        if not prim:
            raise RuntimeError(f"portable preview is missing renderable geometry: {path}")
        actual = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        if not Gf.IsClose(actual, expected, 1e-7):
            raise RuntimeError(f"portable preview geometry transform mismatch: {path}")
    for path, expected in source_camera_transforms.items():
        prim = packaged.GetPrimAtPath(path)
        if not prim or not prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"portable preview is missing camera: {path}")
        actual = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        if not Gf.IsClose(actual, expected, 1e-7):
            raise RuntimeError(f"portable preview camera transform mismatch: {path}")

    preview_root.unlink(missing_ok=True)
    source_root.unlink(missing_ok=True)
    diagnostics["material_count"] = (
        diagnostics["preserved_materials"] + diagnostics["translated_materials"] + diagnostics["fallback_materials"]
    )
    diagnostics["approximation"] = (
        "UsdPreviewSurface materials are portable approximations and are not MDL-faithful simulation materials."
    )
    return diagnostics
