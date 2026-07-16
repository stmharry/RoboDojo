"""Validate task layouts at the environment/scene/task coordinate boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import yaml

from robodojo.core.layouts import ResolvedLayoutSet
from robodojo.core.models.environment import WorkspaceFrameContract

OBJECT_CONFIG_TYPES = ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid")
POSITION_TOLERANCE = 1e-6
SUPPORT_SURFACE_TOLERANCE = 0.01
PlacementBounds = tuple[float, float] | tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class TaskPlacementRule:
    label: str
    relative_plane: str
    xlim: PlacementBounds | None
    ylim: PlacementBounds | None
    expected_count: int | None


def _finite_vector(value: Any, *, length: int, field: str, context: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{context}: {field} must contain {length} finite values")
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: {field} must contain {length} finite values") from exc
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{context}: {field} must contain {length} finite values")
    return result


def _interval(value: Any, *, field: str) -> tuple[float, float]:
    result = _finite_vector(value, length=2, field=field, context="task configuration")
    if result[0] > result[1]:
        raise ValueError(f"task configuration: {field} minimum must not exceed its maximum")
    return result


def _bounds(value: Any, *, field: str) -> PlacementBounds | None:
    if value is None:
        return None
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and value
        and all(isinstance(item, Sequence) and not isinstance(item, (str, bytes)) for item in value)
    ):
        return tuple(_interval(item, field=f"{field} interval {index}") for index, item in enumerate(value))
    return _interval(value, field=field)


def task_placement_rules(task_config: Mapping[str, Any]) -> dict[str, TaskPlacementRule]:
    """Derive deterministic label/support contracts from upstream task YAML."""

    rules: dict[str, TaskPlacementRule] = {}
    for object_type in OBJECT_CONFIG_TYPES:
        groups = task_config.get(object_type, ())
        if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
            continue
        for group in groups:
            if not isinstance(group, Mapping):
                continue
            common = group.get("common", {})
            selection = group.get("select_mode", {})
            if not isinstance(common, Mapping) or not isinstance(selection, Mapping):
                continue
            labels = selection.get("label", ())
            if isinstance(labels, str):
                labels = (labels,)
            if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)) or not labels:
                continue
            relative_plane = common.get("relative_plane")
            if not isinstance(relative_plane, str) or not relative_plane:
                continue
            count = selection.get("nums")
            expected_count = 1 if isinstance(count, int) and count == len(labels) else None
            for raw_label in labels:
                if not isinstance(raw_label, str) or not raw_label:
                    raise ValueError("task configuration: explicit layout labels must be non-empty strings")
                rule = TaskPlacementRule(
                    label=raw_label,
                    relative_plane=relative_plane,
                    xlim=_bounds(common.get("xlim"), field=f"{raw_label} xlim"),
                    ylim=_bounds(common.get("ylim"), field=f"{raw_label} ylim"),
                    expected_count=expected_count,
                )
                previous = rules.get(raw_label)
                if previous is not None and previous != rule:
                    raise ValueError(f"task configuration: label {raw_label!r} has conflicting placement rules")
                rules[raw_label] = rule
    return rules


def _quaternion_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _position_in_frame(
    position: tuple[float, float, float],
    origin: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    norm = math.sqrt(sum(component * component for component in orientation))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=POSITION_TOLERANCE):
        raise ValueError("support-plane orientation must be a normalized scalar-first quaternion")
    delta = tuple(position[index] - origin[index] for index in range(3))
    inverse = (orientation[0], -orientation[1], -orientation[2], -orientation[3])
    rotated = _quaternion_multiply(
        _quaternion_multiply(inverse, (0.0, *delta)),
        orientation,
    )
    return rotated[1], rotated[2], rotated[3]


def _support_pose(
    layout: Mapping[str, Any],
    support: str,
    *,
    context: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float], Mapping[str, Any]] | None:
    config = layout.get(support)
    if not isinstance(config, Mapping):
        if support in {"Table", "Ground"}:
            raise ValueError(f"{context}: required support plane {support!r} is missing")
        return None
    origin = _finite_vector(config.get("default_pos"), length=3, field=f"{support} position", context=context)
    orientation = _finite_vector(
        config.get("default_ori", config.get("default_rot", (1.0, 0.0, 0.0, 0.0))),
        length=4,
        field=f"{support} orientation",
        context=context,
    )
    return origin, orientation, config


def _labelled_instances(layout: Mapping[str, Any], *, context: str) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    for object_type in OBJECT_CONFIG_TYPES:
        categories = layout.get(object_type, {})
        if not isinstance(categories, Mapping):
            raise ValueError(f"{context}: {object_type} must be a mapping")
        for category, instances in categories.items():
            if not isinstance(instances, Sequence) or isinstance(instances, (str, bytes)):
                raise ValueError(f"{context}: {object_type}/{category} must contain a list of instances")
            for instance in instances:
                if not isinstance(instance, Mapping):
                    raise ValueError(f"{context}: {object_type}/{category} contains a non-object instance")
                label = instance.get("label")
                if isinstance(label, str) and label:
                    result.setdefault(label, []).append(instance)
    return result


def _validate_task_placements(
    layout: Mapping[str, Any],
    task_config: Mapping[str, Any],
    *,
    context: str,
) -> None:
    instances_by_label = _labelled_instances(layout, context=context)
    for label, rule in task_placement_rules(task_config).items():
        instances = instances_by_label.get(label, ())
        if not instances:
            raise ValueError(f"{context}: task-required label {label!r} is missing")
        if rule.expected_count is not None and len(instances) != rule.expected_count:
            raise ValueError(
                f"{context}: task-required label {label!r} expected {rule.expected_count} instance(s), "
                f"found {len(instances)}"
            )
        support_name = rule.relative_plane.split("/", 1)[0]
        support = _support_pose(layout, support_name, context=context)
        for instance in instances:
            actual_plane = instance.get("relative_plane")
            if actual_plane != rule.relative_plane:
                raise ValueError(
                    f"{context}: label {label!r} must use relative_plane {rule.relative_plane!r}, "
                    f"found {actual_plane!r}"
                )
            position = _finite_vector(instance.get("default_pos"), length=3, field=f"{label} position", context=context)
            if support is None:
                continue
            origin, orientation, support_config = support
            local = _position_in_frame(position, origin, orientation)
            if support_name == "Table":
                scale = _finite_vector(
                    support_config.get("scale"),
                    length=3,
                    field="Table scale",
                    context=context,
                )
                if local[2] < scale[2] / 2 - SUPPORT_SURFACE_TOLERANCE:
                    raise ValueError(
                        f"{context}: label {label!r} is below the Table surface in its declared support frame"
                    )


def _validate_robot_roots(
    layout: Mapping[str, Any],
    workspace: WorkspaceFrameContract,
    robot_config: Mapping[str, Any],
    *,
    context: str,
) -> None:
    support = _support_pose(layout, workspace.anchor, context=context)
    assert support is not None
    origin, orientation, _ = support
    robots = robot_config.get("robots")
    if not isinstance(robots, Sequence) or isinstance(robots, (str, bytes)):
        raise ValueError(f"{context}: robot component must contain a robots list")
    actual_slots = {f"robot{index}" for index in range(len(robots))}
    expected_slots = set(workspace.robot_root_offsets)
    if actual_slots != expected_slots:
        raise ValueError(
            f"{context}: workspace robot slots mismatch: expected {sorted(expected_slots)}, "
            f"found {sorted(actual_slots)}"
        )
    for index, robot in enumerate(robots):
        slot = f"robot{index}"
        if not isinstance(robot, Mapping):
            raise ValueError(f"{context}: {slot} configuration must be a mapping")
        root = _finite_vector(robot.get("default_root_pos"), length=3, field=f"{slot} root position", context=context)
        local = _position_in_frame(root, origin, orientation)
        expected = workspace.robot_root_offsets[slot]
        if any(
            not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=POSITION_TOLERANCE)
            for actual, wanted in zip(local, expected)
        ):
            raise ValueError(
                f"{context}: {slot} root offset in {workspace.anchor} frame must be {list(expected)}, "
                f"found {list(local)}"
            )


def validate_layout_contract(
    layout: Mapping[str, Any],
    task_config: Mapping[str, Any],
    *,
    workspace: WorkspaceFrameContract | None = None,
    robot_config: Mapping[str, Any] | None = None,
    context: str = "layout",
) -> None:
    """Validate one saved layout without changing its upstream replay shape."""

    _validate_task_placements(layout, task_config, context=context)
    if workspace is not None:
        if robot_config is None:
            raise ValueError(f"{context}: workspace validation requires a robot component")
        _validate_robot_roots(layout, workspace, robot_config, context=context)


def validate_resolved_layout_set(
    resolved: ResolvedLayoutSet,
    *,
    task_config_path: Path,
    workspace: WorkspaceFrameContract | None,
    robot_config_path: Path,
) -> None:
    """Validate every layout selected by preflight or simulator startup."""

    task_config = yaml.safe_load(task_config_path.read_text(encoding="utf-8")) or {}
    robot_config = yaml.safe_load(robot_config_path.read_text(encoding="utf-8")) or {}
    for selected in resolved.layouts:
        layout = json.loads(selected.path.read_text(encoding="utf-8"))
        validate_layout_contract(
            layout,
            task_config,
            workspace=workspace,
            robot_config=robot_config,
            context=str(selected.path),
        )
