"""Build a provenance-preserving Isaac USD for the I2RT YAM arm."""

from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import yaml

ARM_MESH_NAMES = ("base.stl", "link1.stl", "link2.stl", "link3.stl", "link4.stl", "link5.stl")
GRIPPER_MESH_NAMES = ("gripper.stl", "tip_left.stl", "tip_right.stl")
ARM_JOINT_NAMES = tuple(f"dof_joint{index}" for index in range(1, 7))
GRIPPER_JOINT_NAMES = ("dof_joint7", "dof_joint8")
FINGER_LOWER_LIMIT_M = -0.0475
PREVIEW_MATERIAL_KEYS = ("diffuse_color", "roughness", "metallic", "opacity")

logger = logging.getLogger(__name__)


from robodojo.workflows.asset_builders.yam.appearance import (
    _appearance_contract,
    _setup_asset_outputs,
    _visual_proxy_contracts,
)
from robodojo.workflows.asset_builders.yam.common import sha256


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


def _finger_collider_contract(build_manifest: dict) -> dict[str, list[dict]]:
    """Validate the historical YAM finger-pad geometry in I2RT link frames."""

    source = build_manifest.get("physics_contract", {}).get("gripper", {}).get("finger_colliders")
    if not isinstance(source, dict) or source.get("frame") != "i2rt_tip_link":
        raise ValueError("YAM gripper must declare finger colliders in i2rt_tip_link frames")
    derivation_source = source.get("derivation_source")
    provenance = build_manifest.get("sources", {}).get(derivation_source)
    if not isinstance(provenance, dict) or provenance.get("usage") != "reference_only":
        raise ValueError("YAM finger colliders must cite reference-only simulator geometry")

    links = source.get("links")
    if not isinstance(links, dict) or set(links) != {"tip_left", "tip_right"}:
        raise ValueError("YAM finger colliders must cover tip_left and tip_right")
    normalized = {}
    for link_name in ("tip_left", "tip_right"):
        colliders = links[link_name]
        if not isinstance(colliders, list) or len(colliders) != 3:
            raise ValueError(f"{link_name} must declare exactly three finger-pad colliders")
        names = set()
        normalized[link_name] = []
        for collider in colliders:
            if not isinstance(collider, dict) or set(collider) != {
                "name",
                "origin_xyz",
                "origin_rpy",
                "size",
            }:
                raise ValueError(f"invalid {link_name} finger collider: {collider!r}")
            name = collider["name"]
            if not isinstance(name, str) or not name or name in names or "/" in name:
                raise ValueError(f"invalid or duplicate {link_name} finger collider name: {name!r}")
            names.add(name)
            origin_xyz = [float(value) for value in collider["origin_xyz"]]
            origin_rpy = [float(value) for value in collider["origin_rpy"]]
            size = [float(value) for value in collider["size"]]
            if len(origin_xyz) != 3 or len(origin_rpy) != 3 or len(size) != 3 or any(value <= 0 for value in size):
                raise ValueError(f"invalid dimensions for {link_name}/{name}")
            normalized[link_name].append(
                {"name": name, "origin_xyz": origin_xyz, "origin_rpy": origin_rpy, "size": size}
            )
    return normalized


def _add_collision_geometry(robot: ET.Element, finger_colliders: dict[str, list[dict]]) -> int:
    count = 0
    for link in robot.findall("link"):
        if link.findall("collision"):
            raise RuntimeError(f"source link {link.get('name')} already defines collision geometry")
        link_name = link.get("name")
        if link_name in finger_colliders:
            for pad in finger_colliders[link_name]:
                collision = ET.SubElement(link, "collision", {"name": pad["name"]})
                ET.SubElement(
                    collision,
                    "origin",
                    {
                        "xyz": " ".join(str(value) for value in pad["origin_xyz"]),
                        "rpy": " ".join(str(value) for value in pad["origin_rpy"]),
                    },
                )
                geometry = ET.SubElement(collision, "geometry")
                ET.SubElement(geometry, "box", {"size": " ".join(str(value) for value in pad["size"])})
                count += 1
            continue
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
    setup_asset_outputs = list(_setup_asset_outputs(build_manifest).values())

    output_root.mkdir(parents=True, exist_ok=True)
    for relative in ("source", "meshes", "configuration"):
        destination = output_root / relative
        if destination.exists():
            shutil.rmtree(destination)
    for relative in (
        "YAM.usd",
        "config.yaml",
        ".asset_hash",
        "manifest.json",
        *setup_asset_outputs,
        *visual_proxy_outputs,
    ):
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

    finger_colliders = _finger_collider_contract(build_manifest)
    collision_count = _add_collision_geometry(robot, finger_colliders)
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
