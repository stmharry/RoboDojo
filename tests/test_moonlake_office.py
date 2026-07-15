import json
import math
from pathlib import Path

import pytest
import yaml

from robodojo.core.models import SceneConfigDocument, SimulatorLaunchRequest
from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles import load_scene_profile
from robodojo.sim.launcher import resolve_scene_config

ROOT = Path(__file__).resolve().parents[1]
PATHS = RepositoryPaths.resolve(ROOT)
TASKS = ("general_pickup", "stack_blocks", "stack_bowls")


def _request(environment: str) -> SimulatorLaunchRequest:
    return SimulatorLaunchRequest(
        task="general_pickup",
        policy_name="PI05",
        port=9999,
        env_config=environment,
        scene_config="moonlake_office",
        additional_info="contract-test",
    )


def test_moonlake_office_profile_pins_exact_source_mounts():
    scene = load_scene_profile(PATHS, "moonlake_office")
    tooling = yaml.safe_load(PATHS.moonlake_office_manifest.read_text(encoding="utf-8"))

    assert scene.document.compatible_environments == ["bimanual_yam"]
    assert scene.document.layout_set == "moonlake_office"
    assert scene.document.mounts.robots == {
        name: type(scene.document.mounts.robots[name]).model_validate(mount)
        for name, mount in tooling["fixture"]["arm_mounts"].items()
    }
    head = scene.document.mounts.cameras["cam_head"]
    assert head.kind == "scene_fixture"
    assert head.target == "moonlake_office_fixture"
    assert head.frame == tooling["fixture"]["top_camera"]["mount_frame"]
    assert head.position == (0.0, 0.0, 0.0)
    assert head.orientation == (1.0, 0.0, 0.0, 0.0)

    component = scene.component
    assert component["Table"]["scale"] == [1.2, 0.7, 0.05]
    assert component["Table"]["default_pos"] == [0.0, 0.0, 0.725]
    assert component["Ground"]["thickness"] == 0.02
    assert "Room" not in component and "Background" not in component
    assert component["provenance"]["revision"] == tooling["sources"]["spatio_monorepo"]["revision"]


def test_moonlake_office_is_rejected_for_other_embodiments_before_isaac():
    assert resolve_scene_config(PATHS, _request("bimanual_yam")) == "moonlake_office"
    with pytest.raises(ValueError, match="compatible only with environment profiles"):
        resolve_scene_config(PATHS, _request("arx_x5"))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "compatible_environments": ["bimanual_yam"],
                "mounts": {"robots": {"left": {"position": [0, 0, 0], "orientation": [1, 0, 0, 0]}}},
            },
            "robot<N>",
        ),
        (
            {
                "mounts": {
                    "robots": {
                        "robot0": {"position": [math.inf, 0, 0], "orientation": [1, 0, 0, 0]}
                    }
                }
            },
            "finite",
        ),
        (
            {
                "mounts": {
                    "robots": {"robot0": {"position": [0, 0, 0], "orientation": [1, 1, 0, 0]}}
                }
            },
            "normalized",
        ),
        (
            {
                "mounts": {
                    "cameras": {
                        "cam_head": {
                            "kind": "scene_fixture",
                            "target": "fixture",
                            "frame": "../Camera",
                        }
                    }
                }
            },
            "safe relative USD prim path",
        ),
    ],
)
def test_scene_mount_schema_rejects_invalid_contracts(payload, message):
    document = {"config_name": "test", "component": "default", "layout_set": "default", **payload}
    with pytest.raises(ValueError, match=message):
        SceneConfigDocument.model_validate(document)


def test_molmo_yam_retains_its_existing_fallback_mounts():
    scene = load_scene_profile(PATHS, "molmo_yam")
    assert scene.document.compatible_environments == []
    assert scene.document.mounts.robots == {}
    assert scene.document.mounts.cameras == {}


def test_moonlake_layout_bundle_is_fixed_reachable_and_fixture_scoped():
    layout_root = ROOT / "configs" / "layout" / "moonlake_office" / "0"
    assert sorted(path.name for path in layout_root.glob("*.json")) == [f"{task}_0.json" for task in TASKS]

    expected = {
        "general_pickup": {"indices": [1], "labels": ["target"], "xy": [(0.22, -0.10)]},
        "stack_blocks": {
            "indices": [5, 8, 11],
            "labels": ["block_0", "block_1", "block_2"],
            "xy": [(-0.18, -0.06), (0.0, 0.06), (0.18, -0.06)],
        },
        "stack_bowls": {
            "indices": [2, 2, 2],
            "labels": ["bowl0", "bowl1", "bowl2"],
            "xy": [(-0.18, 0.05), (0.0, 0.05), (0.18, 0.05)],
        },
    }

    for task in TASKS:
        layout = json.loads((layout_root / f"{task}_0.json").read_text(encoding="utf-8"))
        assert layout["Table"]["scale"] == [1.2, 0.7, 0.05]
        assert layout["Table"]["default_pos"][2] + layout["Table"]["scale"][2] / 2 == 0.75
        fixture = layout["Geometry"]["moonlake_office_fixture"]
        assert len(fixture) == 1
        assert fixture[0]["label"] == "moonlake_office_fixture"
        assert fixture[0]["default_pos"] == [0.0, 0.0, 0.0]
        assert fixture[0]["physics"] == {"type": "geometry", "collision": False}
        objects = [instance for instances in layout["Rigid"].values() for instance in instances]
        assert [instance["category_idx"] for instance in objects] == expected[task]["indices"]
        assert [instance["label"] for instance in objects] == expected[task]["labels"]
        assert [tuple(instance["default_pos"][:2]) for instance in objects] == expected[task]["xy"]
        assert all(instance["default_ori"] == [1.0, 0.0, 0.0, 0.0] for instance in objects)
        assert all(-0.6 < instance["default_pos"][0] < 0.6 for instance in objects)
        assert all(-0.35 < instance["default_pos"][1] < 0.35 for instance in objects)
        assert layout["Light"]["dome"]["Dome"]["intensity_range"] == [1000.0, 1000.0]
        assert layout["Light"]["key"]["Distant"] == {
            "intensity_range": [3000.0, 3000.0],
            "angle_range": [1.0, 1.0],
        }
