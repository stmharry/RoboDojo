"""Moonlake packing USD and metadata validation."""

from pathlib import Path

from robodojo.workflows.asset_builders.moonlake_packing import geometry


def _validate_stage(output: Path, spec: dict) -> dict:
    stage = geometry.Usd.Stage.Open(str(output))
    if stage is None or not stage.GetDefaultPrim().IsValid():
        raise RuntimeError(f"generated asset could not be reopened: {output}")
    inventory = {"articulations": 0, "rigid_bodies": 0, "joints": 0, "collisions": 0, "cameras": 0}
    max_joint_anchor_error = 0.0
    parent_bodies = set()
    child_bodies = set()
    for prim in stage.Traverse():
        inventory["articulations"] += int(prim.HasAPI(geometry.UsdPhysics.ArticulationRootAPI))
        inventory["rigid_bodies"] += int(prim.HasAPI(geometry.UsdPhysics.RigidBodyAPI))
        inventory["joints"] += int(prim.IsA(geometry.UsdPhysics.Joint))
        inventory["collisions"] += int(prim.HasAPI(geometry.UsdPhysics.CollisionAPI))
        inventory["cameras"] += int(prim.IsA(geometry.UsdGeom.Camera))
        if prim.IsA(geometry.UsdPhysics.Joint):
            joint = geometry.UsdPhysics.Joint(prim)
            body0 = joint.GetBody0Rel().GetTargets()
            body1 = joint.GetBody1Rel().GetTargets()
            if len(body0) != 1 or len(body1) != 1:
                raise RuntimeError(f"joint {prim.GetPath()} must connect exactly two bodies")
            parent_bodies.add(body0[0])
            child_bodies.add(body1[0])
            world0 = geometry.UsdGeom.Xformable(stage.GetPrimAtPath(body0[0])).ComputeLocalToWorldTransform(
                geometry.Usd.TimeCode.Default()
            )
            world1 = geometry.UsdGeom.Xformable(stage.GetPrimAtPath(body1[0])).ComputeLocalToWorldTransform(
                geometry.Usd.TimeCode.Default()
            )
            anchor0 = world0.Transform(geometry.Gf.Vec3d(joint.GetLocalPos0Attr().Get()))
            anchor1 = world1.Transform(geometry.Gf.Vec3d(joint.GetLocalPos1Attr().Get()))
            max_joint_anchor_error = max(max_joint_anchor_error, float((anchor0 - anchor1).GetLength()))
    expected_type = spec["object_type"]
    if inventory["cameras"] or inventory["collisions"] == 0:
        raise RuntimeError(f"invalid generated asset inventory for {spec['category']}: {inventory}")
    if expected_type == "Rigid" and (inventory["articulations"] != 0 or inventory["rigid_bodies"] != 1):
        raise RuntimeError(f"invalid rigid asset inventory for {spec['category']}: {inventory}")
    if expected_type == "Articulation" and inventory["articulations"] != 1:
        raise RuntimeError(f"invalid articulation inventory for {spec['category']}: {inventory}")
    if spec["category"] == "moonlake_magnetic_gift_box" and (
        inventory["rigid_bodies"] != 2 or inventory["joints"] != 1
    ):
        raise RuntimeError(f"invalid gift-box articulation inventory: {inventory}")
    if spec["category"] == "moonlake_anker_cable" and (
        inventory["rigid_bodies"] != int(spec["segment_count"]) + 2
        or inventory["joints"] != int(spec["segment_count"]) + 1
    ):
        raise RuntimeError(f"invalid cable articulation inventory: {inventory}")
    if max_joint_anchor_error > 1e-6:
        raise RuntimeError(
            f"generated articulation {spec['category']} has a {max_joint_anchor_error:.6g} m joint anchor gap"
        )
    if spec["object_type"] == "Articulation":
        roots = parent_bodies - child_bodies
        if len(roots) != 1:
            raise RuntimeError(f"generated articulation {spec['category']} must have one directed root, got {roots}")
        if spec["category"] == "moonlake_anker_cable" and next(iter(roots)).name != "segment_00":
            raise RuntimeError(f"generated cable must be rooted at segment_00, got {roots}")
    return inventory


def _author_asset(key: str, spec: dict, instance: Path) -> tuple[dict, dict]:
    instance.mkdir(parents=True)
    output = instance / "object.usd"
    authors = {
        "container": geometry._author_container,
        "spoon": geometry._author_spoon,
        "phone": geometry._author_phone,
        "apple": geometry._author_apple,
        "screwdriver": geometry._author_screwdriver,
        "cable": geometry._author_cable,
        "block": geometry._author_block,
    }
    metadata = authors[key](spec, output)
    geometry._write_json(instance / "metadata.json", metadata)
    geometry._write_json(
        instance / "description.json",
        {"caption": spec["description"], "description": [spec["description"]]},
    )
    return metadata, _validate_stage(output, spec)
