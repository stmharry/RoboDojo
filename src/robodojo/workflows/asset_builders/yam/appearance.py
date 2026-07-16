"""Build a provenance-preserving Isaac USD for the I2RT YAM arm."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

ARM_MESH_NAMES = ("base.stl", "link1.stl", "link2.stl", "link3.stl", "link4.stl", "link5.stl")
GRIPPER_MESH_NAMES = ("gripper.stl", "tip_left.stl", "tip_right.stl")
ARM_JOINT_NAMES = tuple(f"dof_joint{index}" for index in range(1, 7))
GRIPPER_JOINT_NAMES = ("dof_joint7", "dof_joint8")
FINGER_LOWER_LIMIT_M = -0.0475
PREVIEW_MATERIAL_KEYS = ("diffuse_color", "roughness", "metallic", "opacity")

logger = logging.getLogger(__name__)


from robodojo.workflows.asset_builders.yam.common import sha256


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _appearance_contract(
    build_manifest: dict,
    visual_links: list[str],
    setup_name: str | None = None,
) -> dict:
    """Validate and normalize the deterministic YAM render-material contract."""

    source = build_manifest.get("appearance")
    if not isinstance(source, dict):
        raise ValueError("YAM tooling must declare an appearance contract")
    if source.get("shader") != "UsdPreviewSurface":
        raise ValueError("YAM appearance supports only UsdPreviewSurface")
    if source.get("color_space") != "linear_rgb":
        raise ValueError("YAM appearance colors must be declared in linear_rgb")
    material_scope = source.get("material_scope")
    if not isinstance(material_scope, str) or not material_scope or "/" in material_scope:
        raise ValueError(f"invalid YAM material scope: {material_scope!r}")

    palette_source = source.get("palette")
    if not isinstance(palette_source, dict) or not palette_source:
        raise ValueError("YAM appearance palette cannot be empty")
    palette = {}
    for name in sorted(palette_source):
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError(f"invalid YAM material name: {name!r}")
        raw = palette_source[name]
        if not isinstance(raw, dict) or set(raw) != set(PREVIEW_MATERIAL_KEYS):
            raise ValueError(f"YAM material {name!r} must define exactly {PREVIEW_MATERIAL_KEYS}")
        diffuse_color = raw["diffuse_color"]
        if not isinstance(diffuse_color, list) or len(diffuse_color) != 3:
            raise ValueError(f"YAM material {name!r} must declare a 3D diffuse_color")
        material = {
            "diffuse_color": [float(value) for value in diffuse_color],
            "roughness": float(raw["roughness"]),
            "metallic": float(raw["metallic"]),
            "opacity": float(raw["opacity"]),
        }
        scalar_values = (*material["diffuse_color"], material["roughness"], material["metallic"], material["opacity"])
        if any(value < 0.0 or value > 1.0 for value in scalar_values):
            raise ValueError(f"YAM material {name!r} values must be normalized to [0,1]")
        material["sha256"] = _canonical_sha256(
            {
                "color_space": source["color_space"],
                "name": name,
                "shader": source["shader"],
                **material,
            }
        )
        palette[name] = material

    setups = source.get("setups")
    expected_setups = set(_setup_asset_outputs(build_manifest))
    if not isinstance(setups, dict) or set(setups) != expected_setups:
        raise ValueError(f"YAM appearance must declare setup mappings for {sorted(expected_setups)}")
    default_setup = source.get("default_setup")
    if default_setup not in setups:
        raise ValueError(f"YAM appearance default setup is invalid: {default_setup!r}")

    normalized_setups = {}
    used_materials = set()
    for name in sorted(setups):
        setup = setups[name]
        if not isinstance(setup, dict) or set(setup) != {"derivation_source", "link_materials"}:
            raise ValueError(f"YAM appearance setup {name!r} must declare derivation_source and link_materials")
        derivation_source = setup["derivation_source"]
        provenance = build_manifest.get("sources", {}).get(derivation_source)
        if not isinstance(provenance, dict) or provenance.get("usage") != "reference_only":
            raise ValueError(f"YAM appearance setup {name!r} must cite reference-only provenance")
        if provenance.get("license") in {None, "", "undeclared"}:
            raise ValueError(f"YAM appearance setup {name!r} provenance must declare a public dataset license")
        link_materials = setup["link_materials"]
        if not isinstance(link_materials, dict) or set(link_materials) != set(visual_links):
            missing = sorted(set(visual_links) - set(link_materials or {}))
            extra = sorted(set(link_materials or {}) - set(visual_links))
            raise ValueError(
                f"YAM appearance setup {name!r} link mapping differs from visual links: "
                f"missing={missing}, extra={extra}"
            )
        unknown = sorted(set(link_materials.values()) - set(palette))
        if unknown:
            raise ValueError(f"YAM appearance setup {name!r} references unknown materials: {unknown}")
        used_materials.update(link_materials.values())
        normalized_setups[name] = {
            "derivation_source": derivation_source,
            "link_materials": {link: link_materials[link] for link in sorted(link_materials)},
        }
    unused = sorted(set(palette) - used_materials)
    if unused:
        raise ValueError(f"YAM appearance contains unused materials: {unused}")

    selected_setup = default_setup if setup_name is None else setup_name
    if selected_setup not in normalized_setups:
        raise ValueError(f"unknown YAM appearance setup: {selected_setup!r}")
    selected = normalized_setups[selected_setup]

    return {
        "setup": selected_setup,
        "derivation_source": selected["derivation_source"],
        "shader": source["shader"],
        "color_space": source["color_space"],
        "material_scope": material_scope,
        "palette": palette,
        "link_materials": selected["link_materials"],
    }


def _visual_proxy_contracts(build_manifest: dict, appearance: dict) -> dict:
    """Validate standalone render-only hardware proxies generated with YAM."""

    proxies = build_manifest.get("asset", {}).get("visual_proxies")
    expected = {"molmoact2", "moonlake_office"}
    if not isinstance(proxies, dict) or set(proxies) != expected:
        raise ValueError(f"YAM tooling must declare D405 proxies for {sorted(expected)}")
    contracts = {}
    outputs = set()
    for setup_name in sorted(proxies):
        source = proxies[setup_name]
        if not isinstance(source, dict):
            raise ValueError(f"YAM {setup_name} D405 visual proxy must be a mapping")
        output = source.get("output")
        if not isinstance(output, str) or Path(output).name != output or Path(output).suffix.lower() != ".usd":
            raise ValueError(f"invalid D405 visual proxy output: {output!r}")
        if output in outputs:
            raise ValueError(f"duplicate D405 visual proxy output: {output}")
        outputs.add(output)
        if source.get("default_prim") != "D405" or source.get("optical_frame") != "OpticalFrame":
            raise ValueError("D405 visual proxy must publish an identity OpticalFrame below its D405 default prim")
        if source.get("physical") is not False:
            raise ValueError("D405 visual proxy must be explicitly non-physical")
        provenance_source = source.get("provenance_source")
        provenance = build_manifest.get("sources", {}).get(provenance_source)
        if not isinstance(provenance, dict) or provenance.get("usage") != "geometry_reference":
            raise ValueError("D405 visual proxy must cite geometry-reference provenance")
        dimensions = [float(value) for value in source.get("dimensions_m", [])]
        if dimensions != [0.042, 0.042, 0.023]:
            raise ValueError(f"D405 visual proxy must use nominal [width,height,depth] dimensions, got {dimensions}")
        documented_dimensions = provenance.get("nominal_dimensions_m", {})
        if [documented_dimensions.get(key) for key in ("width", "height", "depth")] != dimensions:
            raise ValueError("D405 visual proxy dimensions differ from their cited provenance")
        materials = {"housing": source.get("housing_material"), "detail": source.get("detail_material")}
        unknown = sorted(set(materials.values()) - set(appearance["palette"]))
        if unknown:
            raise ValueError(f"D405 visual proxy references unknown materials: {unknown}")
        contract = {
            "setup": setup_name,
            "output": output,
            "default_prim": source["default_prim"],
            "optical_frame": source["optical_frame"],
            "provenance_source": provenance_source,
            "dimensions_m": dimensions,
            "materials": materials,
            "physical": False,
        }
        contract["contract_sha256"] = _canonical_sha256(contract)
        contracts[setup_name] = contract
    return contracts


def _setup_asset_outputs(build_manifest: dict) -> dict[str, str]:
    source = build_manifest.get("asset", {}).get("setup_outputs")
    expected = {"molmoact2", "moonlake_office"}
    if not isinstance(source, dict) or set(source) != expected:
        raise ValueError(f"YAM tooling must declare robot outputs for {sorted(expected)}")
    outputs = {str(name): str(output) for name, output in source.items()}
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("YAM setup robot outputs must be unique")
    for output in outputs.values():
        if Path(output).name != output or Path(output).suffix.lower() != ".usd":
            raise ValueError(f"invalid YAM setup robot output: {output!r}")
    return outputs


def _stage_physics_digest(stage) -> str:
    """Hash authored rigid-body, collision, and joint state for mutation guards."""

    from pxr import Usd, UsdPhysics

    records = []
    for prim in stage.Traverse():
        is_physics = (
            prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or prim.HasAPI(UsdPhysics.CollisionAPI)
            or prim.IsA(UsdPhysics.Joint)
            or "/collisions" in str(prim.GetPath())
        )
        if not is_physics:
            continue
        records.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "schemas": sorted(prim.GetAppliedSchemas()),
                "attributes": [
                    {
                        "name": attribute.GetName(),
                        "type": str(attribute.GetTypeName()),
                        "value": repr(attribute.Get(Usd.TimeCode.Default())),
                    }
                    for attribute in sorted(prim.GetAttributes(), key=lambda item: item.GetName())
                ],
                "relationships": [
                    {
                        "name": relationship.GetName(),
                        "targets": sorted(str(target) for target in relationship.GetTargets()),
                    }
                    for relationship in sorted(prim.GetRelationships(), key=lambda item: item.GetName())
                ],
            }
        )
    return _canonical_sha256(records)


def _author_preview_appearance(stage, appearance: dict, validated_visual_paths: list[str]) -> dict:
    """Author canonical YAM hardware materials for validated renderable Gprims."""

    from pxr import Gf, Sdf, UsdGeom, UsdShade

    default_path = stage.GetDefaultPrim().GetPath()
    looks_path = default_path.AppendChild(appearance["material_scope"])
    UsdGeom.Scope.Define(stage, looks_path)
    materials = {}
    material_contracts = {}
    for name, parameters in appearance["palette"].items():
        material_path = looks_path.AppendChild(name)
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path.AppendChild("PreviewSurface"))
        shader.CreateIdAttr(appearance["shader"])
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*parameters["diffuse_color"]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(parameters["roughness"])
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(parameters["metallic"])
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(parameters["opacity"])
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        materials[name] = material
        material_contracts[name] = {
            **parameters,
            "material_path": str(material_path),
            "shader_path": str(shader.GetPath()),
        }

    validated = set(validated_visual_paths)
    renderables_by_link = {link: [] for link in appearance["link_materials"]}
    for path in sorted(validated):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Gprim):
            raise RuntimeError(f"appearance target is not a renderable Gprim: {path}")
        relative = prim.GetPath().MakeRelativePath(default_path)
        path_elements = str(relative).split("/")
        if len(path_elements) < 3 or path_elements[1] != "visuals":
            raise RuntimeError(f"appearance target is outside a link visual hierarchy: {path}")
        link = path_elements[0]
        if link not in renderables_by_link:
            raise RuntimeError(f"appearance target has no link material mapping: {path}")
        renderables_by_link[link].append(path)

    bindings = []
    for link, material_name in appearance["link_materials"].items():
        renderable_paths = renderables_by_link[link]
        if not renderable_paths:
            raise RuntimeError(f"appearance link has no validated renderable geometry: {link}")
        target_path = default_path.AppendChild(link).AppendChild("visuals")
        target = stage.GetPrimAtPath(target_path)
        if not target.IsValid() or not target.IsA(UsdGeom.Xform):
            raise RuntimeError(f"appearance visual root is not an editable Xform: {target_path}")
        parameters = appearance["palette"][material_name]
        UsdShade.MaterialBindingAPI.Apply(target).Bind(
            materials[material_name],
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        UsdGeom.PrimvarsAPI(target).CreatePrimvar(
            "displayColor",
            Sdf.ValueTypeNames.Color3fArray,
            UsdGeom.Tokens.constant,
        ).Set([Gf.Vec3f(*parameters["diffuse_color"])])
        for renderable_path in renderable_paths:
            renderable = stage.GetPrimAtPath(renderable_path)
            bound = UsdShade.MaterialBindingAPI(renderable).ComputeBoundMaterial()[0]
            if not bound or bound.GetPath() != materials[material_name].GetPath():
                raise RuntimeError(f"appearance material did not bind to {renderable_path}")
            display_color = UsdGeom.PrimvarsAPI(renderable).FindPrimvarWithInheritance("displayColor").Get()
            if not display_color or any(
                abs(float(actual) - expected) > 1e-6
                for actual, expected in zip(display_color[0], parameters["diffuse_color"])
            ):
                raise RuntimeError(f"appearance displayColor did not reach {renderable_path}")
        bindings.append(
            {
                "link": link,
                "target": str(target_path),
                "renderable_paths": renderable_paths,
                "material": material_name,
            }
        )

    if {binding["link"] for binding in bindings} != set(appearance["link_materials"]):
        raise RuntimeError("appearance bindings do not cover every mapped visual link")
    return {
        **appearance,
        "palette": material_contracts,
        "bindings": bindings,
    }


def _author_d405_visual_proxy(output_root: Path, contract: dict, appearance: dict) -> dict:
    """Author a deterministic D405 proxy with a named identity optical frame."""

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    root = UsdGeom.Xform.Define(stage, "/D405")
    stage.SetDefaultPrim(root.GetPrim())
    optical_frame = UsdGeom.Xform.Define(stage, "/D405/OpticalFrame")
    looks_path = Sdf.Path("/D405/Looks")
    UsdGeom.Scope.Define(stage, looks_path)
    materials = {}
    for role, material_name in sorted(contract["materials"].items()):
        parameters = appearance["palette"][material_name]
        material_path = looks_path.AppendChild(role)
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path.AppendChild("PreviewSurface"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*parameters["diffuse_color"]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(parameters["roughness"])
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(parameters["metallic"])
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(parameters["opacity"])
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        materials[role] = material

    visual_paths = []

    def bind_visual(gprim, role: str) -> None:
        parameters = appearance["palette"][contract["materials"][role]]
        prim = gprim.GetPrim()
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[role])
        UsdGeom.Gprim(prim).CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(
            [Gf.Vec3f(*parameters["diffuse_color"])]
        )
        visual_paths.append(str(prim.GetPath()))

    width, height, depth = contract["dimensions_m"]
    housing = UsdGeom.Cube.Define(stage, "/D405/Housing")
    housing.CreateSizeAttr(1.0)
    housing_xform = UsdGeom.Xformable(housing.GetPrim())
    housing_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, depth / 2.0))
    housing_xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(width, height, depth))
    bind_visual(housing, "housing")

    front = UsdGeom.Cube.Define(stage, "/D405/FrontPanel")
    front.CreateSizeAttr(1.0)
    front_xform = UsdGeom.Xformable(front.GetPrim())
    front_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.00025))
    front_xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.034, 0.014, 0.0005))
    bind_visual(front, "detail")

    for side, x_position in (("Left", -0.011), ("Right", 0.011)):
        lens = UsdGeom.Cylinder.Define(stage, f"/D405/{side}Lens")
        lens.CreateAxisAttr(UsdGeom.Tokens.z)
        lens.CreateRadiusAttr(0.004)
        lens.CreateHeightAttr(0.0002)
        lens_xform = UsdGeom.Xformable(lens.GetPrim())
        lens_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(x_position, 0.0, 0.0001))
        bind_visual(lens, "detail")

    forbidden_physics_paths = []
    for prim in stage.Traverse():
        if (
            prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or prim.HasAPI(UsdPhysics.CollisionAPI)
            or prim.HasAPI(UsdPhysics.MassAPI)
            or prim.IsA(UsdPhysics.Joint)
        ):
            forbidden_physics_paths.append(str(prim.GetPath()))
    if forbidden_physics_paths:
        raise RuntimeError(f"D405 visual proxy unexpectedly contains physics: {forbidden_physics_paths}")
    if UsdGeom.Xformable(optical_frame.GetPrim()).GetOrderedXformOps():
        raise RuntimeError("D405 OpticalFrame must remain identity")

    output = output_root / contract["output"]
    stage.GetRootLayer().Export(str(output), args={"format": "usda"})
    reopened = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
    if reopened is None or str(reopened.GetDefaultPrim().GetPath()) != "/D405":
        raise RuntimeError(f"could not reopen generated D405 visual proxy: {output}")
    reopened_optical_frame = reopened.GetPrimAtPath("/D405/OpticalFrame")
    if not reopened_optical_frame.IsValid() or UsdGeom.Xformable(reopened_optical_frame).GetOrderedXformOps():
        raise RuntimeError("saved D405 proxy must publish an identity OpticalFrame child")
    for prim in reopened.Traverse():
        if (
            prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or prim.HasAPI(UsdPhysics.CollisionAPI)
            or prim.HasAPI(UsdPhysics.MassAPI)
            or prim.IsA(UsdPhysics.Joint)
        ):
            raise RuntimeError(f"saved D405 visual proxy unexpectedly contains physics: {prim.GetPath()}")
    return {
        **contract,
        "visual_paths": sorted(visual_paths),
        "sha256": sha256(output),
    }
