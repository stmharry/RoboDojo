"""Moonlake office fixture validation."""

from robodojo.workflows.asset_builders.moonlake_office import geometry


def _validate_static_fixture(stage: geometry.Usd.Stage, root_path: str, mount_frame: str) -> dict:
    required = (
        f"{root_path}/TableFrame/LegFL",
        f"{root_path}/ArmRail/Extrusion2060",
        f"{root_path}/CameraStand/D435Assembly/Body",
        f"{root_path}/{mount_frame}",
    )
    missing = [path for path in required if not stage.GetPrimAtPath(path).IsValid()]
    if missing:
        raise RuntimeError(f"generated Moonlake fixture is missing required prims: {missing}")

    collision_paths = []
    for prim in stage.Traverse():
        if prim.HasAPI(geometry.UsdPhysics.RigidBodyAPI) or prim.HasAPI(geometry.UsdPhysics.MassAPI):
            raise RuntimeError(f"generated Moonlake fixture contains dynamic physics at {prim.GetPath()}")
        if prim.IsA(geometry.UsdPhysics.Joint):
            raise RuntimeError(f"generated Moonlake fixture contains a joint at {prim.GetPath()}")
        if prim.IsA(geometry.UsdGeom.Camera):
            raise RuntimeError(f"generated Moonlake fixture contains a camera sensor at {prim.GetPath()}")
        drive_attributes = [
            attribute.GetName() for attribute in prim.GetAttributes() if attribute.GetName().startswith("drive:")
        ]
        if drive_attributes:
            raise RuntimeError(f"generated Moonlake fixture contains drive attributes at {prim.GetPath()}")
        if prim.HasAPI(geometry.UsdPhysics.CollisionAPI):
            collision_paths.append(str(prim.GetPath()))

    expected_collisions = [
        f"{root_path}/TableFrame/LegFL",
        f"{root_path}/TableFrame/LegFR",
        f"{root_path}/TableFrame/LegBL",
        f"{root_path}/TableFrame/LegBR",
        f"{root_path}/ArmRail/Extrusion2060",
    ]
    if collision_paths != expected_collisions:
        raise RuntimeError(f"generated Moonlake fixture collision contract changed: {collision_paths}")

    optical = stage.GetPrimAtPath(f"{root_path}/{mount_frame}")
    matrix = geometry.UsdGeom.XformCache().GetLocalToWorldTransform(optical)
    return {
        "static_only": True,
        "camera_sensor_count": 0,
        "joint_count": 0,
        "collision_prims": collision_paths,
        "mount_frame_transform": [[float(matrix[row][column]) for column in range(4)] for row in range(4)],
    }
