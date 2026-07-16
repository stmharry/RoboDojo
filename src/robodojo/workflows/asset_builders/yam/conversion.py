"""Build a provenance-preserving Isaac USD for the I2RT YAM arm."""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
import traceback

ARM_MESH_NAMES = ("base.stl", "link1.stl", "link2.stl", "link3.stl", "link4.stl", "link5.stl")
GRIPPER_MESH_NAMES = ("gripper.stl", "tip_left.stl", "tip_right.stl")
ARM_JOINT_NAMES = tuple(f"dof_joint{index}" for index in range(1, 7))
GRIPPER_JOINT_NAMES = ("dof_joint7", "dof_joint8")
FINGER_LOWER_LIMIT_M = -0.0475
PREVIEW_MATERIAL_KEYS = ("diffuse_color", "roughness", "metallic", "opacity")

logger = logging.getLogger(__name__)


from robodojo.workflows.asset_builders.yam.appearance import (
    _appearance_contract,
    _author_d405_visual_proxy,
    _author_preview_appearance,
    _setup_asset_outputs,
    _stage_physics_digest,
    _visual_proxy_contracts,
)
from robodojo.workflows.asset_builders.yam.common import sha256
from robodojo.workflows.asset_builders.yam.geometry import (
    _fixed_camera_frame_contract,
    _remove_empty_generated_visual_prims,
    _validate_generated_visuals,
)


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
        # The official IsaacLab wheel keeps its implementation below
        # ``isaaclab/source/isaaclab/isaaclab`` while exposing only the app
        # launcher from the top-level package. AppLauncher normally extends
        # this path, but this standalone asset converter starts SimulationApp
        # directly and must make the bundled modules visible itself.
        import isaaclab

        bundled_isaaclab = Path(isaaclab.__file__).resolve().parent / "source" / "isaaclab" / "isaaclab"
        if bundled_isaaclab.is_dir() and str(bundled_isaaclab) not in isaaclab.__path__:
            isaaclab.__path__.append(str(bundled_isaaclab))
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
        if len(collision_paths) != 13 or len(finger_collision_paths) != 6:
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
        setup_assets = {}
        for setup_name, setup_output in sorted(_setup_asset_outputs(build_manifest).items()):
            destination = output_root / setup_output
            shutil.copy2(output, destination)
            setup_stage = Usd.Stage.Open(str(destination), load=Usd.Stage.LoadAll)
            setup_appearance = _appearance_contract(build_manifest, visual_links, setup_name)
            setup_generated_appearance = _author_preview_appearance(
                setup_stage,
                setup_appearance,
                validated_visual_paths,
            )
            setup_stage.GetRootLayer().Save()
            setup_physics = _stage_physics_digest(setup_stage)
            if setup_physics != physics_digest_before_appearance:
                raise RuntimeError(f"YAM setup asset {setup_name} changed the shared physics contract")
            setup_assets[setup_name] = {
                "output": setup_output,
                "sha256": sha256(destination),
                "physics_contract_sha256": setup_physics,
                "appearance": setup_generated_appearance,
            }
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
            "setup_assets": setup_assets,
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
