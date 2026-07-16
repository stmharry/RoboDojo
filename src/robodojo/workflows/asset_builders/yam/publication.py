"""Build a provenance-preserving Isaac USD for the I2RT YAM arm."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

ARM_MESH_NAMES = ("base.stl", "link1.stl", "link2.stl", "link3.stl", "link4.stl", "link5.stl")
GRIPPER_MESH_NAMES = ("gripper.stl", "tip_left.stl", "tip_right.stl")
ARM_JOINT_NAMES = tuple(f"dof_joint{index}" for index in range(1, 7))
GRIPPER_JOINT_NAMES = ("dof_joint7", "dof_joint8")
FINGER_LOWER_LIMIT_M = -0.0475
PREVIEW_MATERIAL_KEYS = ("diffuse_color", "roughness", "metallic", "opacity")

logger = logging.getLogger(__name__)


from robodojo.workflows.asset_builders.yam.common import sha256
from robodojo.workflows.asset_builders.yam.conversion import _convert_to_usd
from robodojo.workflows.asset_builders.yam.geometry import derive_yam_urdf


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
