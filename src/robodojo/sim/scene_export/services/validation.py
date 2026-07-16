"""Export the composed, post-reset Isaac stage without editing the live stage."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pxr import Usd

logger = logging.getLogger(__name__)

USD_EXTENSIONS = frozenset({".usd", ".usda", ".usdc", ".usdz"})
MANIFEST_NAME = "scene_manifest.json"
REFERENCED_NAME = "scene_referenced.usda"
FLATTENED_NAME = "scene_flattened.usdc"


from robodojo.sim.scene_export.services.state_capture import _local_and_world


def _list_op_items(value) -> list[Any]:
    if value is None:
        return []
    items = []
    for name in ("explicitItems", "prependedItems", "appendedItems", "addedItems"):
        items.extend(getattr(value, name, []) or [])
    return items


def _validate_flattened(stage: Usd.Stage) -> tuple[list[str], list[str]]:
    external_arcs = []
    internal_arcs = []
    for prim in stage.TraverseAll():
        if prim.HasAuthoredReferences():
            references = _list_op_items(prim.GetMetadata("references"))
            external = [reference.assetPath for reference in references if reference.assetPath]
            target = external_arcs if external else internal_arcs
            target.append(f"reference:{prim.GetPath()}:{external}")
        if prim.HasAuthoredPayloads():
            payloads = _list_op_items(prim.GetMetadata("payload"))
            external = [payload.assetPath for payload in payloads if payload.assetPath]
            target = external_arcs if external else internal_arcs
            target.append(f"payload:{prim.GetPath()}:{external}")
    if stage.GetRootLayer().subLayerPaths:
        external_arcs.extend(f"sublayer:{path}" for path in stage.GetRootLayer().subLayerPaths)
    return external_arcs, internal_arcs


def _validate_reopened_stage(stage: Usd.Stage, state: dict[str, Any], label: str) -> None:
    missing = []
    for paths in state["inventory"].values():
        for path in paths:
            if not stage.GetPrimAtPath(path).IsValid():
                missing.append(path)
    if missing:
        raise RuntimeError(f"{label} export is missing expected prims: {missing[:20]}")
    for camera in state["cameras"]:
        actual = _local_and_world(stage, camera["sensor_path"])["world_matrix"]
        expected = camera["sensor_transform"]["world_matrix"]
        if actual is None or not np.allclose(actual, expected, rtol=0.0, atol=1e-7):
            raise RuntimeError(f"{label} camera transform mismatch: {camera['observation_key']}")
