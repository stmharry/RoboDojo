from collections.abc import Mapping

from omegaconf import DictConfig, OmegaConf


def normalize_robot_mount_overrides(
    mount_overrides: Mapping | DictConfig | None,
    robot_count: int,
) -> dict:
    if isinstance(mount_overrides, DictConfig):
        mount_overrides = OmegaConf.to_container(mount_overrides, resolve=True)
    normalized = dict(mount_overrides or {})
    expected_mount_slots = {f"robot{idx}" for idx in range(robot_count)}
    unknown_mount_slots = sorted(set(normalized) - expected_mount_slots)
    if unknown_mount_slots:
        raise ValueError(f"scene robot mount overrides reference unknown slots: {unknown_mount_slots}")
    return normalized


def apply_robot_mount_override(robots, mount_override: Mapping | None) -> list[float]:
    """Apply a scene-owned root pose without changing the robot's embodiment contract."""
    if not isinstance(robots, tuple):
        robots = (robots,)
    if not mount_override:
        return list(robots[0].entity_origin_pose)
    position = list(mount_override["position"])
    orientation = list(mount_override["orientation"])
    base_pose = position + orientation
    for robot in robots:
        robot.default_root_pos = list(position)
        robot.default_root_rot = list(orientation)
        robot.entity_origin_pose = list(base_pose)
    return base_pose
