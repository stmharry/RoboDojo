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

logger = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def derive_yam_urdf(source_root: Path, output_root: Path, build_manifest: dict) -> dict:
    """Create the normalized runtime URDF and source snapshot without importing Isaac."""
    asset = build_manifest["asset"]
    source_root = source_root.resolve()
    arm_urdf = _required_path(source_root, asset["arm_urdf"])
    arm_mesh_dir = _required_path(source_root, asset["arm_meshes"])
    gripper_mesh_dir = _required_path(source_root, asset["gripper_meshes"])
    license_path = _required_path(source_root, build_manifest["sources"]["i2rt"]["license_path"])

    output_root.mkdir(parents=True, exist_ok=True)
    for relative in ("source", "meshes", "configuration"):
        destination = output_root / relative
        if destination.exists():
            shutil.rmtree(destination)
    for relative in ("YAM.usd", "config.yaml", ".asset_hash", "manifest.json"):
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
        "mesh_sources": {name: sha256(path) for name, path in sorted(mesh_sources.items())},
        "source_urdf_sha256": sha256(arm_urdf),
        "license_sha256": sha256(license_path),
    }


def _convert_to_usd(derived_urdf: Path, output_root: Path, build_manifest: dict) -> dict:
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
        from pxr import Usd, UsdPhysics, UsdShade

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

        stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
        if stage is None or not stage.GetDefaultPrim().IsValid():
            raise RuntimeError(f"could not open generated YAM stage {output}")
        default_path = str(stage.GetDefaultPrim().GetPath())
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
        stage.GetRootLayer().Save()
        generated_contract = {
            "output": output.name,
            "default_prim": default_path,
            "joint_names": joint_names,
            "required_links": sorted(required_links),
            "finger_material_targets": sorted(finger_material_targets),
            "collision_paths": sorted(collision_paths),
            "finger_collision_paths": sorted(finger_collision_paths),
            "finger_physics_material": {"static_friction": 3.0, "dynamic_friction": 2.5},
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
    generated, simulation_app = _convert_to_usd(derived["derived_urdf"], output_root, build_manifest)
    source = build_manifest["sources"]["i2rt"]
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
        },
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
