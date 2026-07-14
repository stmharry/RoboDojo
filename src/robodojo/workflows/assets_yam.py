"""Build a provenance-preserving Isaac USD for the I2RT YAM arm."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
import shutil
import traceback
import xml.etree.ElementTree as ET

import yaml

ARM_MESH_NAMES = ("base.stl", "link1.stl", "link2.stl", "link3.stl", "link4.stl", "link5.stl")
GRIPPER_MESH_NAMES = ("gripper.stl", "tip_left.stl", "tip_right.stl")
ARM_JOINT_NAMES = tuple(f"dof_joint{index}" for index in range(1, 7))
GRIPPER_JOINT_NAMES = ("dof_joint7", "dof_joint8")
FINGER_LOWER_LIMIT_M = -0.0475
PREVIEW_MATERIAL_KEYS = ("diffuse_color", "roughness", "metallic", "opacity")

logger = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _appearance_contract(build_manifest: dict, visual_links: list[str]) -> dict:
    """Validate and normalize the deterministic YAM render-material contract."""

    source = build_manifest.get("appearance")
    if not isinstance(source, dict):
        raise ValueError("YAM tooling must declare an appearance contract")
    derivation_source = source.get("derivation_source")
    provenance = build_manifest.get("sources", {}).get(derivation_source)
    if not isinstance(provenance, dict) or provenance.get("usage") != "reference_only":
        raise ValueError("YAM appearance must cite a reference-only provenance source")
    if provenance.get("license") in {None, "", "undeclared"}:
        raise ValueError("YAM appearance provenance must declare a public dataset license")
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

    link_materials = source.get("link_materials")
    if not isinstance(link_materials, dict) or set(link_materials) != set(visual_links):
        missing = sorted(set(visual_links) - set(link_materials or {}))
        extra = sorted(set(link_materials or {}) - set(visual_links))
        raise ValueError(f"YAM appearance link mapping differs from visual links: missing={missing}, extra={extra}")
    unknown = sorted(set(link_materials.values()) - set(palette))
    if unknown:
        raise ValueError(f"YAM appearance references unknown materials: {unknown}")
    unused = sorted(set(palette) - set(link_materials.values()))
    if unused:
        raise ValueError(f"YAM appearance contains unused materials: {unused}")

    return {
        "derivation_source": derivation_source,
        "shader": source["shader"],
        "color_space": source["color_space"],
        "material_scope": material_scope,
        "palette": palette,
        "link_materials": {link: link_materials[link] for link in sorted(link_materials)},
    }


def _visual_proxy_contracts(build_manifest: dict, appearance: dict) -> dict:
    """Validate standalone render-only hardware proxies generated with YAM."""

    proxies = build_manifest.get("asset", {}).get("visual_proxies")
    if not isinstance(proxies, dict) or set(proxies) != {"d405"}:
        raise ValueError("YAM tooling must declare exactly one d405 visual proxy")
    source = proxies["d405"]
    if not isinstance(source, dict):
        raise ValueError("YAM d405 visual proxy must be a mapping")
    output = source.get("output")
    if not isinstance(output, str) or Path(output).name != output or Path(output).suffix.lower() != ".usd":
        raise ValueError(f"invalid D405 visual proxy output: {output!r}")
    if source.get("default_prim") != "OpticalFrame" or source.get("optical_frame") != "OpticalFrame":
        raise ValueError("D405 visual proxy must use OpticalFrame as its identity default prim")
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
    materials = {
        "housing": source.get("housing_material"),
        "detail": source.get("detail_material"),
    }
    unknown = sorted(set(materials.values()) - set(appearance["palette"]))
    if unknown:
        raise ValueError(f"D405 visual proxy references unknown materials: {unknown}")
    contract = {
        "output": output,
        "default_prim": source["default_prim"],
        "optical_frame": source["optical_frame"],
        "provenance_source": provenance_source,
        "dimensions_m": dimensions,
        "materials": materials,
        "physical": False,
    }
    contract["contract_sha256"] = _canonical_sha256(contract)
    return {"d405": contract}


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
    """Author Molmo-style preview materials for validated renderable Gprims."""

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
    """Author a deterministic D405 render proxy rooted at its optical plane."""

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    optical_frame = UsdGeom.Xform.Define(stage, "/OpticalFrame")
    stage.SetDefaultPrim(optical_frame.GetPrim())
    looks_path = Sdf.Path("/OpticalFrame/Looks")
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
    housing = UsdGeom.Cube.Define(stage, "/OpticalFrame/Housing")
    housing.CreateSizeAttr(1.0)
    housing_xform = UsdGeom.Xformable(housing.GetPrim())
    housing_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, depth / 2.0))
    housing_xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(width, height, depth))
    bind_visual(housing, "housing")

    front = UsdGeom.Cube.Define(stage, "/OpticalFrame/FrontPanel")
    front.CreateSizeAttr(1.0)
    front_xform = UsdGeom.Xformable(front.GetPrim())
    front_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.00025))
    front_xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.034, 0.014, 0.0005))
    bind_visual(front, "detail")

    for side, x_position in (("Left", -0.011), ("Right", 0.011)):
        lens = UsdGeom.Cylinder.Define(stage, f"/OpticalFrame/{side}Lens")
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
    if reopened is None or str(reopened.GetDefaultPrim().GetPath()) != "/OpticalFrame":
        raise RuntimeError(f"could not reopen generated D405 visual proxy: {output}")
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


def _required_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.exists():
        raise FileNotFoundError(path)
    return path


def _joint_by_name(robot: ET.Element, name: str) -> ET.Element:
    matches = [joint for joint in robot.findall("joint") if joint.get("name") == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name}, found {len(matches)}")
    return matches[0]


def _add_collisions_from_visuals(robot: ET.Element) -> int:
    count = 0
    for link in robot.findall("link"):
        if link.findall("collision"):
            raise RuntimeError(f"source link {link.get('name')} already defines collision geometry")
        for visual in link.findall("visual"):
            collision = ET.Element("collision")
            origin = visual.find("origin")
            geometry = visual.find("geometry")
            if geometry is None:
                raise RuntimeError(f"visual on {link.get('name')} has no geometry")
            if origin is not None:
                collision.append(deepcopy(origin))
            collision.append(deepcopy(geometry))
            link.append(collision)
            count += 1
    return count


def _fixed_camera_frame_contract(build_manifest: dict) -> list[dict]:
    """Validate and normalize non-physical camera frames authored into USD."""
    frames = []
    seen_paths = set()
    for source in build_manifest["asset"].get("fixed_frames", []):
        frame = deepcopy(source)
        name = frame.get("name")
        parent = frame.get("parent")
        if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
            raise ValueError(f"invalid fixed camera frame name: {name!r}")
        if not isinstance(parent, str) or not parent or parent.startswith("/") or ".." in parent.split("/"):
            raise ValueError(f"invalid fixed camera frame parent: {parent!r}")
        if len(frame.get("position", [])) != 3 or len(frame.get("orientation", [])) != 4:
            raise ValueError(f"fixed camera frame {name!r} must declare a 3D position and scalar-first quaternion")
        if frame.get("physical") is not False:
            raise ValueError(f"fixed camera frame {name!r} must be explicitly non-physical")
        derivation_source = frame.get("derivation_source")
        reference = build_manifest["sources"].get(derivation_source)
        if not isinstance(reference, dict) or reference.get("usage") != "reference_only":
            raise ValueError(f"fixed camera frame {name!r} must cite a reference-only derivation source")
        relative_path = f"{parent.rstrip('/')}/{name}"
        if relative_path in seen_paths:
            raise ValueError(f"duplicate fixed camera frame path: {relative_path}")
        seen_paths.add(relative_path)
        frame["path"] = relative_path
        frames.append(frame)
    return frames


def _visual_link_contract(robot: ET.Element) -> tuple[list[str], list[str]]:
    """Return deterministic URDF link sets with and without visual geometry."""
    link_names = []
    visual_links = []
    for link in robot.findall("link"):
        name = link.get("name")
        if not name:
            raise RuntimeError("URDF link is missing its name")
        if name in link_names:
            raise RuntimeError(f"duplicate URDF link name: {name}")
        link_names.append(name)
        if link.findall("visual"):
            visual_links.append(name)
    links_without_visuals = sorted(set(link_names) - set(visual_links))
    return sorted(visual_links), links_without_visuals


def _generated_usd_layers(output_root: Path):
    """Open every generated USD layer without composing their scene arcs."""
    from pxr import Sdf

    output_root = output_root.resolve()
    layers = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
            continue
        layer = Sdf.Layer.FindOrOpen(str(path))
        if layer is None:
            raise RuntimeError(f"could not open generated USD layer {path}")
        real_path = Path(layer.realPath).resolve() if layer.realPath else None
        if real_path is None or not real_path.is_relative_to(output_root):
            raise RuntimeError(f"generated USD layer resolves outside the asset directory: {layer.identifier}")
        layers.append(layer)
    if not layers:
        raise RuntimeError(f"no generated USD layers found below {output_root}")
    return layers


def _remove_empty_generated_visual_prims(
    output: Path,
    output_root: Path,
    links_without_visuals: list[str],
) -> list[str]:
    """Remove importer residue for URDF links that intentionally have no visuals."""
    from pxr import Sdf

    output_root = output_root.resolve()
    root_layer = Sdf.Layer.FindOrOpen(str(output))
    if root_layer is None or not root_layer.defaultPrim:
        raise RuntimeError(f"generated YAM root layer has no default prim: {output}")
    layers = _generated_usd_layers(output_root)
    removed_paths = []
    changed_layers = set()
    for link_name in links_without_visuals:
        visual_path = Sdf.Path(f"/{root_layer.defaultPrim}/{link_name}/visuals")
        specs = [layer.GetPrimAtPath(visual_path) for layer in layers]
        specs = [spec for spec in specs if spec is not None]
        if not specs:
            continue
        for spec in specs:
            layer_path = Path(spec.layer.realPath).resolve() if spec.layer.realPath else None
            if layer_path is None or not layer_path.is_relative_to(output_root):
                raise RuntimeError(f"refusing to edit visual prim outside the generated asset: {spec.path}")
            if spec.typeName not in {"", "Xform"} or spec.nameChildren or spec.properties:
                raise RuntimeError(f"refusing to remove non-empty visual prim for {link_name}: {spec.path}")
            if (
                spec.payloadList.GetAppliedItems()
                or spec.inheritPathList.GetAppliedItems()
                or spec.specializesList.GetAppliedItems()
            ):
                raise RuntimeError(f"refusing to remove composed visual prim for {link_name}: {spec.path}")
            if dict(spec.variantSelections) or spec.variantSetNameList.GetAppliedItems():
                raise RuntimeError(f"refusing to remove variant-bearing visual prim for {link_name}: {spec.path}")
            for reference in spec.referenceList.GetAppliedItems():
                if not reference.primPath:
                    raise RuntimeError(f"visual reference for {link_name} has no explicit target: {reference}")
                if reference.assetPath:
                    resolved = Path(Sdf.ComputeAssetPathRelativeToLayer(spec.layer, reference.assetPath)).resolve()
                    if not resolved.is_relative_to(output_root):
                        raise RuntimeError(f"visual reference for {link_name} resolves outside the asset: {resolved}")
                    target_layer = Sdf.Layer.FindOrOpen(str(resolved))
                    if target_layer is not None and target_layer.GetPrimAtPath(reference.primPath) is not None:
                        raise RuntimeError(f"refusing to remove resolved visual reference for {link_name}: {reference}")
                elif any(layer.GetPrimAtPath(reference.primPath) is not None for layer in layers):
                    raise RuntimeError(f"refusing to remove resolved visual reference for {link_name}: {reference}")
            parent = spec.nameParent
            if parent is None or parent.nameChildren.get(spec.name) is None:
                raise RuntimeError(f"could not locate authored visual prim for removal: {spec.path}")
            authored_layer = spec.layer
            del parent.nameChildren[spec.name]
            changed_layers.add(authored_layer)
        removed_paths.append(str(visual_path))
    for layer in changed_layers:
        layer.Save()
    return sorted(removed_paths)


def _validate_generated_visuals(stage, visual_links: list[str], links_without_visuals: list[str]) -> list[str]:
    """Validate that generated visuals match the source URDF visual contract."""
    from pxr import Sdf, Usd, UsdGeom

    default_path = stage.GetDefaultPrim().GetPath()
    gprims = [prim for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()) if prim.IsA(UsdGeom.Gprim)]
    validated_paths = []
    for link_name in visual_links:
        visual_path = default_path.AppendChild(link_name).AppendChild("visuals")
        link_gprims = [prim for prim in gprims if prim.GetPath().HasPrefix(visual_path)]
        if not link_gprims:
            raise RuntimeError(f"generated visual contract is missing renderable geometry below {visual_path}")
        validated_paths.extend(str(prim.GetPath()) for prim in link_gprims)
    for link_name in links_without_visuals:
        visual_path = default_path.AppendChild(link_name).AppendChild("visuals")
        if stage.GetPrimAtPath(Sdf.Path(visual_path)).IsValid():
            raise RuntimeError(f"generated nonvisual link still has a visual prim: {visual_path}")
    return sorted(validated_paths)


def derive_yam_urdf(source_root: Path, output_root: Path, build_manifest: dict) -> dict:
    """Create the normalized runtime URDF and source snapshot without importing Isaac."""
    asset = build_manifest["asset"]
    source_root = source_root.resolve()
    arm_urdf = _required_path(source_root, asset["arm_urdf"])
    arm_mesh_dir = _required_path(source_root, asset["arm_meshes"])
    gripper_mesh_dir = _required_path(source_root, asset["gripper_meshes"])
    license_path = _required_path(source_root, build_manifest["sources"]["i2rt"]["license_path"])
    visual_proxy_outputs = [
        proxy.get("output") for proxy in asset.get("visual_proxies", {}).values() if isinstance(proxy, dict)
    ]
    if any(not isinstance(name, str) or Path(name).name != name for name in visual_proxy_outputs):
        raise ValueError(f"invalid visual proxy outputs: {visual_proxy_outputs}")

    output_root.mkdir(parents=True, exist_ok=True)
    for relative in ("source", "meshes", "configuration"):
        destination = output_root / relative
        if destination.exists():
            shutil.rmtree(destination)
    for relative in ("YAM.usd", "config.yaml", ".asset_hash", "manifest.json", *visual_proxy_outputs):
        (output_root / relative).unlink(missing_ok=True)
    source_snapshot = output_root / "source"
    (source_snapshot / "arm").mkdir(parents=True)
    (source_snapshot / "gripper").mkdir(parents=True)
    shutil.copy2(arm_urdf, source_snapshot / "arm" / "yam.urdf")
    shutil.copytree(arm_mesh_dir, source_snapshot / "arm" / "assets")
    shutil.copytree(gripper_mesh_dir, source_snapshot / "gripper" / "assets")
    shutil.copy2(license_path, output_root / "LICENSE-I2RT")

    mesh_output = output_root / "meshes"
    mesh_output.mkdir()
    mesh_sources: dict[str, Path] = {}
    for name in ARM_MESH_NAMES:
        source = _required_path(arm_mesh_dir, name)
        mesh_sources[name] = source
        shutil.copy2(source, mesh_output / name)
    for name in GRIPPER_MESH_NAMES:
        source = _required_path(gripper_mesh_dir, name)
        mesh_sources[name] = source
        shutil.copy2(source, mesh_output / name)

    tree = ET.parse(arm_urdf)
    robot = tree.getroot()
    if robot.tag != "robot":
        raise RuntimeError(f"unexpected URDF root {robot.tag!r}")
    robot.set("name", "yam")

    referenced_meshes = []
    for mesh in robot.findall(".//mesh"):
        source_name = Path(mesh.get("filename", "")).name.lower()
        if source_name not in mesh_sources:
            raise RuntimeError(f"unmapped source mesh {mesh.get('filename')!r}")
        mesh.set("filename", f"meshes/{source_name}")
        referenced_meshes.append(source_name)
    if sorted(referenced_meshes) != sorted(mesh_sources):
        raise RuntimeError(f"URDF mesh set differs from expected I2RT contract: {referenced_meshes}")
    visual_links, links_without_visuals = _visual_link_contract(robot)
    appearance = _appearance_contract(build_manifest, visual_links)
    visual_proxies = _visual_proxy_contracts(build_manifest, appearance)

    base_joint = _joint_by_name(robot, "dof_joint0")
    limit = base_joint.find("limit")
    if limit is None or float(limit.get("lower", "nan")) != 0.0 or float(limit.get("upper", "nan")) != 0.0:
        raise RuntimeError("dof_joint0 is no longer the expected zero-range source joint")
    base_joint.set("type", "fixed")
    for child_name in ("axis", "limit"):
        child = base_joint.find(child_name)
        if child is not None:
            base_joint.remove(child)

    finger_limits = {}
    for name in GRIPPER_JOINT_NAMES:
        joint = _joint_by_name(robot, name)
        limit = joint.find("limit")
        if limit is None:
            raise RuntimeError(f"{name} has no limit")
        limit.set("lower", str(FINGER_LOWER_LIMIT_M))
        limit.set("upper", "0.0")
        limit.set("effort", "40.0")
        finger_limits[name] = [float(limit.get("lower")), float(limit.get("upper"))]

    collision_count = _add_collisions_from_visuals(robot)
    ET.indent(tree, space="  ")
    derived_urdf = output_root / asset["derived_urdf"]
    tree.write(derived_urdf, encoding="utf-8", xml_declaration=True)
    (output_root / "robot_config.yml").write_text(
        yaml.safe_dump(build_manifest["robot_config"], sort_keys=False), encoding="utf-8"
    )
    return {
        "derived_urdf": derived_urdf,
        "base_joint": {"name": "dof_joint0", "type": "fixed", "origin_preserved": True},
        "arm_joints": list(ARM_JOINT_NAMES),
        "gripper_joints": list(GRIPPER_JOINT_NAMES),
        "finger_limits_m": finger_limits,
        "collision_geometry_count": collision_count,
        "visual_links": visual_links,
        "links_without_visuals": links_without_visuals,
        "appearance": appearance,
        "visual_proxies": visual_proxies,
        "mesh_sources": {name: sha256(path) for name, path in sorted(mesh_sources.items())},
        "source_urdf_sha256": sha256(arm_urdf),
        "license_sha256": sha256(license_path),
    }


def _convert_to_usd(
    derived_urdf: Path,
    output_root: Path,
    build_manifest: dict,
    visual_links: list[str],
    links_without_visuals: list[str],
) -> dict:
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

        class _NoVersionSwitchUrdfConverter(UrdfConverter):
            """Use Isaac Sim's installed importer when merge_fixed_joints is disabled.

            IsaacLab 0.54.3 attempts to enable importer 2.4.31 on Isaac Sim 5.1
            to retain legacy fixed-joint merge behavior. YAM explicitly disables
            that merge, and the 5.1 wheel currently ships 2.4.30, so requesting
            2.4.31 makes an otherwise supported conversion fail dependency
            resolution.
            """

            def __init__(self, cfg):
                from isaacsim.asset.importer.urdf._urdf import acquire_urdf_interface

                self._urdf_interface = acquire_urdf_interface()
                AssetConverterBase.__init__(self, cfg=cfg)

            def _get_urdf_import_config(self):
                import omni.kit.commands

                _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
                import_config.set_distance_scale(1.0)
                import_config.set_make_default_prim(True)
                import_config.set_create_physics_scene(False)
                import_config.set_density(self.cfg.link_density)
                import_config.set_convex_decomp(self.cfg.collider_type == "convex_decomposition")
                import_config.set_collision_from_visuals(self.cfg.collision_from_visuals)
                import_config.set_merge_fixed_joints(self.cfg.merge_fixed_joints)
                if hasattr(import_config, "set_merge_fixed_ignore_inertia"):
                    import_config.set_merge_fixed_ignore_inertia(self.cfg.merge_fixed_joints)
                import_config.set_fix_base(self.cfg.fix_base)
                import_config.set_self_collision(self.cfg.self_collision)
                import_config.set_parse_mimic(self.cfg.convert_mimic_joints_to_normal_joints)
                import_config.set_replace_cylinders_with_capsules(self.cfg.replace_cylinders_with_capsules)
                return import_config

        converter_contract = build_manifest["asset"]["converter"]
        converter = _NoVersionSwitchUrdfConverter(
            UrdfConverterCfg(
                asset_path=str(derived_urdf),
                usd_dir=str(output_root),
                usd_file_name=build_manifest["asset"]["output"],
                fix_base=bool(converter_contract["fix_base"]),
                merge_fixed_joints=bool(converter_contract["merge_fixed_joints"]),
                make_instanceable=bool(converter_contract["make_instanceable"]),
                force_usd_conversion=True,
                collision_from_visuals=bool(converter_contract["collision_from_visuals"]),
                collider_type=str(converter_contract["collider_type"]),
                self_collision=bool(converter_contract["self_collision"]),
                joint_drive=UrdfConverterCfg.JointDriveCfg(
                    target_type="position",
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
                ),
            )
        )
        output = Path(converter.usd_path)
        if output.resolve() != (output_root / build_manifest["asset"]["output"]).resolve():
            raise RuntimeError(f"converter wrote unexpected output {output}")

        removed_empty_visual_prims = _remove_empty_generated_visual_prims(
            output,
            output_root,
            links_without_visuals,
        )
        stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
        if stage is None or not stage.GetDefaultPrim().IsValid():
            raise RuntimeError(f"could not open generated YAM stage {output}")
        validated_visual_paths = _validate_generated_visuals(stage, visual_links, links_without_visuals)
        default_path = str(stage.GetDefaultPrim().GetPath())
        fixed_camera_frames = []
        for frame in _fixed_camera_frame_contract(build_manifest):
            parent_path = f"{default_path}/{frame['parent'].strip('/')}"
            if not stage.GetPrimAtPath(parent_path).IsValid():
                raise RuntimeError(f"fixed camera frame parent does not exist: {parent_path}")
            frame_path = f"{default_path}/{frame['path']}"
            frame_xform = UsdGeom.Xform.Define(stage, frame_path)
            xformable = UsdGeom.Xformable(frame_xform.GetPrim())
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Vec3d(*[float(value) for value in frame["position"]])
            )
            orientation = frame["orientation"]
            xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Quatd(float(orientation[0]), *[float(value) for value in orientation[1:]])
            )
            prim = frame_xform.GetPrim()
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
                raise RuntimeError(f"fixed camera frame unexpectedly has physics APIs: {frame_path}")
            fixed_camera_frames.append(
                {
                    "path": frame_path,
                    "parent": parent_path,
                    "position": list(frame["position"]),
                    "orientation": list(frame["orientation"]),
                    "derivation_source": frame["derivation_source"],
                    "physical": False,
                }
            )
        material = UsdShade.Material.Define(stage, f"{default_path}/fingerPhysicsMaterial")
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr(3.0)
        physics_material.CreateDynamicFrictionAttr(2.5)
        physics_material.CreateRestitutionAttr(0.0)
        finger_material_targets = []
        for side in ("tip_left", "tip_right"):
            target = stage.GetPrimAtPath(f"{default_path}/{side}/collisions")
            if not target.IsValid():
                raise RuntimeError(f"missing editable collision root for {side}")
            UsdShade.MaterialBindingAPI.Apply(target).Bind(material, materialPurpose="physics")
            finger_material_targets.append(str(target.GetPath()))

        collision_paths = []
        finger_collision_paths = []
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            path = str(prim.GetPath())
            collision_paths.append(path)
            if any(f"/{side}/collisions/" in path for side in ("tip_left", "tip_right")):
                bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")[0]
                if not bound or bound.GetPath() != material.GetPath():
                    raise RuntimeError(f"finger collision {path} did not inherit its physics material")
                finger_collision_paths.append(path)
        if len(collision_paths) != 9 or len(finger_collision_paths) != 2:
            raise RuntimeError(
                f"generated collision contract mismatch: total={collision_paths}, fingers={finger_collision_paths}"
            )

        joint_names = sorted(
            prim.GetName()
            for prim in stage.Traverse()
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)
        )
        expected = sorted((*ARM_JOINT_NAMES, *GRIPPER_JOINT_NAMES))
        if joint_names != expected:
            raise RuntimeError(f"generated joint contract mismatch: {joint_names} != {expected}")
        link_names = {prim.GetName() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
        required_links = {"base", "gripper", "tip_left", "tip_right"}
        if not required_links.issubset(link_names):
            raise RuntimeError(f"generated link contract is missing {sorted(required_links - link_names)}")
        appearance = _appearance_contract(build_manifest, visual_links)
        physics_digest_before_appearance = _stage_physics_digest(stage)
        generated_appearance = _author_preview_appearance(stage, appearance, validated_visual_paths)
        physics_digest_after_appearance = _stage_physics_digest(stage)
        if physics_digest_after_appearance != physics_digest_before_appearance:
            raise RuntimeError("authoring YAM appearance changed the physics or collision contract")
        stage.GetRootLayer().Save()
        completed_stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
        if completed_stage is None or not completed_stage.GetDefaultPrim().IsValid():
            raise RuntimeError(f"could not reopen completed YAM stage {output}")
        validated_visual_paths = _validate_generated_visuals(
            completed_stage,
            visual_links,
            links_without_visuals,
        )
        if _stage_physics_digest(completed_stage) != physics_digest_before_appearance:
            raise RuntimeError("saved YAM appearance changed the physics or collision contract")
        for binding in generated_appearance["bindings"]:
            expected_material = f"{default_path}/{appearance['material_scope']}/{binding['material']}"
            for renderable_path in binding["renderable_paths"]:
                prim = completed_stage.GetPrimAtPath(renderable_path)
                bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
                if not bound or str(bound.GetPath()) != expected_material:
                    raise RuntimeError(f"saved YAM appearance binding is invalid: {binding}")
        generated_visual_proxies = {
            name: _author_d405_visual_proxy(output_root, contract, appearance)
            for name, contract in _visual_proxy_contracts(build_manifest, appearance).items()
        }
        generated_contract = {
            "output": output.name,
            "default_prim": default_path,
            "joint_names": joint_names,
            "required_links": sorted(required_links),
            "finger_material_targets": sorted(finger_material_targets),
            "collision_paths": sorted(collision_paths),
            "finger_collision_paths": sorted(finger_collision_paths),
            "finger_physics_material": {"static_friction": 3.0, "dynamic_friction": 2.5},
            "fixed_camera_frames": fixed_camera_frames,
            "appearance": {
                **generated_appearance,
                "physics_contract_sha256": physics_digest_before_appearance,
            },
            "visual_proxies": generated_visual_proxies,
            "validated_visual_paths": validated_visual_paths,
            "removed_empty_visual_prims": removed_empty_visual_prims,
        }
        # IsaacLab's cache/config files are build-time implementation details;
        # the manifest below records the stable converter contract instead.
        for auxiliary in (output_root / ".asset_hash", output_root / "config.yaml"):
            auxiliary.unlink(missing_ok=True)
    except Exception:
        traceback.print_exc()
        simulation_app.close()
        raise
    return generated_contract, simulation_app


def _output_checksums(output_root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output_root)): sha256(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def build(source_root: Path, output_root: Path, manifest_path: Path) -> dict:
    build_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    derived = derive_yam_urdf(source_root, output_root, build_manifest)
    generated, simulation_app = _convert_to_usd(
        derived["derived_urdf"],
        output_root,
        build_manifest,
        derived["visual_links"],
        derived["links_without_visuals"],
    )
    source = build_manifest["sources"]["i2rt"]
    reference_sources = {key: value for key, value in build_manifest["sources"].items() if key != "i2rt"}
    result = {
        "format": 1,
        "asset": "yam",
        "provenance": {
            "repository": source["repository"],
            "revision": source["revision"],
            "license": source["license"],
            "source_urdf_sha256": derived["source_urdf_sha256"],
            "source_mesh_sha256": derived["mesh_sources"],
            "license_sha256": derived["license_sha256"],
            "build_manifest_sha256": sha256(manifest_path),
        },
        "reference_provenance": reference_sources,
        "transformations": list(build_manifest["asset"]["transformations"]),
        "derived_contract": {key: value for key, value in derived.items() if key != "derived_urdf"},
        "converter": dict(build_manifest["asset"]["converter"]),
        "generated_contract": generated,
        "robot_contract": build_manifest["robot_config"],
        "physics_contract": build_manifest["physics_contract"],
        "outputs": _output_checksums(output_root),
    }
    (output_root / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    simulation_app.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source_root, args.output_root, args.manifest)
    logger.info("Built YAM asset:\n%s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
