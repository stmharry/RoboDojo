"""Pinned AgileX PiPER URDF-to-USD asset builder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_source(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise FileNotFoundError(path)
    return path


def derive_piper_urdf(source_root: Path, output_root: Path, manifest: dict) -> dict:
    source = manifest["sources"]["piper_ros"]
    urdf_source = _checked_source(source_root, source["urdf_path"])
    license_source = _checked_source(source_root, source["license_path"])
    if _sha256(urdf_source) != source["urdf_sha256"]:
        raise RuntimeError("pinned PiPER URDF checksum mismatch")
    if _sha256(license_source) != source["license_sha256"]:
        raise RuntimeError("pinned piper_ros license checksum mismatch")

    output_root.mkdir(parents=True, exist_ok=True)
    mesh_output = output_root / "meshes"
    if mesh_output.exists():
        shutil.rmtree(mesh_output)
    mesh_output.mkdir()
    mesh_source = (source_root / source["mesh_path"]).resolve()
    for name, expected in source["mesh_sha256"].items():
        path = _checked_source(mesh_source, name)
        if _sha256(path) != expected:
            raise RuntimeError(f"pinned PiPER mesh checksum mismatch: {name}")
        shutil.copy2(path, mesh_output / name)

    tree = ET.parse(urdf_source)
    robot = tree.getroot()
    referenced = []
    for mesh in robot.findall(".//mesh"):
        name = Path(mesh.get("filename", "")).name
        if name not in source["mesh_sha256"]:
            raise RuntimeError(f"unexpected PiPER mesh reference: {mesh.get('filename')!r}")
        mesh.set("filename", f"meshes/{name}")
        referenced.append(name)
    if set(referenced) != set(source["mesh_sha256"]):
        raise RuntimeError("PiPER URDF mesh inventory changed")

    limits = {}
    expected_limits = manifest["joint_limits"]
    for joint in robot.findall("joint"):
        name = joint.get("name")
        if name not in expected_limits:
            continue
        limit = joint.find("limit")
        actual = [float(limit.get("lower")), float(limit.get("upper"))]
        expected = [float(value) for value in expected_limits[name]]
        if actual != expected:
            raise RuntimeError(f"PiPER {name} limits changed: {actual} != {expected}")
        limits[name] = actual
    if set(limits) != set(expected_limits):
        raise RuntimeError("PiPER joint inventory changed")

    ET.indent(tree, space="  ")
    derived = output_root / manifest["asset"]["derived_urdf"]
    tree.write(derived, encoding="utf-8", xml_declaration=True)
    shutil.copy2(license_source, output_root / "LICENSE-piper_ros")
    (output_root / "robot_config.yml").write_text(
        yaml.safe_dump(manifest["robot_config"], sort_keys=False), encoding="utf-8"
    )
    return {"derived_urdf": derived, "joint_limits": limits}


def _convert(urdf: Path, output_root: Path, manifest: dict):
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import isaaclab

        bundled = Path(isaaclab.__file__).resolve().parent / "source" / "isaaclab" / "isaaclab"
        if bundled.is_dir() and str(bundled) not in isaaclab.__path__:
            isaaclab.__path__.append(str(bundled))
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
        from pxr import Usd, UsdPhysics

        from robodojo.workflows.asset_builders.yam.geometry import _remove_empty_generated_visual_prims

        class _InstalledUrdfConverter(UrdfConverter):
            """Use Isaac Sim's bundled importer for unmerged fixed joints.

            IsaacLab requests importer 2.4.31 for its legacy fixed-joint path,
            while the pinned Isaac Sim 5.1 wheel supplies 2.4.30. PiPER keeps
            fixed joints, so the optional version switch is unnecessary.
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

        config = manifest["asset"]["converter"]
        converter = _InstalledUrdfConverter(
            UrdfConverterCfg(
                asset_path=str(urdf),
                usd_dir=str(output_root),
                usd_file_name=manifest["asset"]["output"],
                fix_base=bool(config["fix_base"]),
                merge_fixed_joints=bool(config["merge_fixed_joints"]),
                make_instanceable=bool(config["make_instanceable"]),
                force_usd_conversion=True,
                collision_from_visuals=bool(config["collision_from_visuals"]),
                collider_type=str(config["collider_type"]),
                self_collision=bool(config["self_collision"]),
                joint_drive=UrdfConverterCfg.JointDriveCfg(
                    target_type="position",
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
                ),
            )
        )
        output = Path(converter.usd_path)
        if output.resolve() != (output_root / manifest["asset"]["output"]).resolve():
            raise RuntimeError(f"converter wrote unexpected output {output}")
        removed_empty_visual_prims = _remove_empty_generated_visual_prims(
            output,
            output_root,
            ["dummy_link"],
        )
        stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
        if stage is None or not stage.GetDefaultPrim().IsValid():
            raise RuntimeError(f"could not reopen generated PiPER stage {output}")
        joints = sorted(
            prim.GetName()
            for prim in stage.Traverse()
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)
        )
        expected_joints = [f"joint{index}" for index in range(1, 9)]
        if joints != expected_joints:
            raise RuntimeError(f"generated PiPER joint contract mismatch: {joints} != {expected_joints}")
        rigid_bodies = sorted(
            prim.GetName() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        )
        required_bodies = {"base_link", "gripper_base", *(f"link{index}" for index in range(1, 9))}
        missing_bodies = sorted(required_bodies.difference(rigid_bodies))
        if missing_bodies:
            raise RuntimeError(f"generated PiPER stage is missing rigid bodies: {missing_bodies}")
        collision_count = sum(
            1
            for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        )
        if collision_count < len(required_bodies):
            raise RuntimeError(f"generated PiPER stage has too few colliders: {collision_count}")
        generated = {
            "output": output.name,
            "default_prim": str(stage.GetDefaultPrim().GetPath()),
            "joint_names": joints,
            "rigid_bodies": rigid_bodies,
            "collision_count": collision_count,
            "removed_empty_visual_prims": removed_empty_visual_prims,
        }
        return app, generated
    except Exception:
        import traceback

        traceback.print_exc()
        app.close()
        raise


def build(source_root: Path, output_root: Path, manifest_path: Path) -> dict:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    derived = derive_piper_urdf(source_root, output_root, manifest)
    app, generated = _convert(derived["derived_urdf"], output_root, manifest)
    try:
        outputs = {
            str(path.relative_to(output_root)): _sha256(path)
            for path in sorted(output_root.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        result = {
            "format": 1,
            "asset": "piper",
            "provenance": {
                "repository": manifest["sources"]["piper_ros"]["repository"],
                "revision": manifest["sources"]["piper_ros"]["revision"],
                "license": manifest["sources"]["piper_ros"]["license"],
                "source_urdf_sha256": manifest["sources"]["piper_ros"]["urdf_sha256"],
                "source_mesh_sha256": manifest["sources"]["piper_ros"]["mesh_sha256"],
                "license_sha256": manifest["sources"]["piper_ros"]["license_sha256"],
                "build_manifest_sha256": _sha256(manifest_path),
            },
            "transformations": manifest["asset"]["transformations"],
            "converter": manifest["asset"]["converter"],
            "generated_contract": generated,
            "joint_limits": derived["joint_limits"],
            "robot_contract": manifest["robot_config"],
            "outputs": outputs,
        }
        (output_root / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    except BaseException:
        # SimulationApp.close() terminates the standalone process before Python
        # can render an active exception, so emit it while the app is live.
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    build(args.source_root, args.output_root, args.manifest)


if __name__ == "__main__":
    main()
