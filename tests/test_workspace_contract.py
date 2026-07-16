from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from robodojo.core.models.environment import EnvironmentConfigDocument
from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles.environment import load_environment_profile
from robodojo.core.profiles.scene import load_scene_profile
from robodojo.core.workspace import task_placement_rules, validate_layout_contract

ROOT = Path(__file__).resolve().parents[1]
PATHS = RepositoryPaths.resolve(ROOT)


def _task_config():
    return yaml.safe_load((ROOT / "configs/task/general_pickup.yml").read_text(encoding="utf-8"))


def _profile_and_layout(environment, layout_set):
    profile = load_environment_profile(PATHS, environment)
    robot = yaml.safe_load(profile.component_paths["robot"].read_text(encoding="utf-8"))
    layout_path = ROOT / f"configs/layout/{layout_set}/0/general_pickup_0.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    return profile, robot, layout, layout_path


def test_yam_workspace_contract_is_inherited_and_validates_both_scenes():
    expected_offsets = {
        "robot0": (-0.24, -0.40, 0.025),
        "robot1": (0.24, -0.40, 0.025),
    }
    for environment, layout_set in (
        ("bimanual_yam_molmoact2", "molmo_yam"),
        ("bimanual_yam_moonlake_office", "moonlake_office"),
    ):
        profile, robot, layout, layout_path = _profile_and_layout(environment, layout_set)
        assert profile.document.workspace is not None
        assert profile.document.workspace.anchor == "Table"
        assert profile.document.workspace.robot_root_offsets == expected_offsets
        validate_layout_contract(
            layout,
            _task_config(),
            workspace=profile.document.workspace,
            robot_config=robot,
            context=str(layout_path),
        )


def test_general_pickup_rule_is_derived_from_upstream_task_yaml():
    rules = task_placement_rules(_task_config())
    assert set(rules) == {"target"}
    rule = rules["target"]
    assert rule.relative_plane == "Table"
    assert rule.xlim == (-0.4, 0.4)
    assert rule.ylim == (-0.2, 0.05)
    assert rule.expected_count == 1


def test_upstream_disjoint_placement_intervals_remain_explicit():
    task_config = yaml.safe_load((ROOT / "configs/task/deposit_coin.yml").read_text(encoding="utf-8"))
    rules = task_placement_rules(task_config)

    assert rules["piggy_bank"].xlim == ((-0.4, -0.3), (0.3, 0.4))


def test_general_pickup_scenes_have_no_hidden_container_contracts():
    forbidden_roles = {"box", "container", "item", "scene_bin"}
    for scene_name in ("molmo_yam", "moonlake_office"):
        scene = load_scene_profile(PATHS, scene_name)
        assert "general_pickup" not in scene.document.task_assets
        assert "general_pickup" not in scene.document.task_asset_builds

        layout_path = ROOT / f"configs/layout/{scene.document.layout_set}/0/general_pickup_0.json"
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        labels = {
            instance.get("label")
            for object_type in ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid")
            for instances in layout.get(object_type, {}).values()
            for instance in instances
        }
        assert labels.isdisjoint(forbidden_roles)
        assert "basket" not in layout.get("Geometry", {})


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda layout: layout["Rigid"]["ball"][0].update(relative_plane="Ground"), "relative_plane 'Table'"),
        (lambda layout: layout["Rigid"]["ball"][0].update(default_pos=[0.22, float("nan"), 0.753]), "finite"),
        (lambda layout: layout["Rigid"]["ball"][0].update(default_pos=[0.22, -0.05, 0.70]), "below the Table"),
        (lambda layout: layout["Rigid"]["ball"][0].pop("label"), "label 'target' is missing"),
    ],
)
def test_layout_contract_rejects_task_frame_drift(mutation, message):
    profile, robot, original, _ = _profile_and_layout("bimanual_yam_moonlake_office", "moonlake_office")
    layout = deepcopy(original)
    mutation(layout)
    with pytest.raises(ValueError, match=message):
        validate_layout_contract(
            layout,
            _task_config(),
            workspace=profile.document.workspace,
            robot_config=robot,
            context="moonlake regression",
        )


def test_layout_contract_rejects_environment_root_drift():
    profile, robot, layout, _ = _profile_and_layout("bimanual_yam_moonlake_office", "moonlake_office")
    robot["robots"][1]["default_root_pos"][1] -= 0.01
    with pytest.raises(ValueError, match="robot1 root offset"):
        validate_layout_contract(
            layout,
            _task_config(),
            workspace=profile.document.workspace,
            robot_config=robot,
            context="moonlake regression",
        )


@pytest.mark.parametrize(
    ("workspace", "message"),
    [
        ({"anchor": "../Table", "robot_root_offsets": {"robot0": [0, 0, 0]}}, "safe scene fixture"),
        ({"anchor": "Table", "robot_root_offsets": {"left": [0, 0, 0]}}, "robot<N>"),
        ({"anchor": "Table", "robot_root_offsets": {"robot0": [0, 0]}}, "3 finite"),
        ({"anchor": "Table", "robot_root_offsets": {}}, "must not be empty"),
    ],
)
def test_environment_workspace_schema_rejects_invalid_contracts(workspace, message):
    with pytest.raises(ValueError, match=message):
        EnvironmentConfigDocument.model_validate(
            {
                "config_name": "test",
                "workspace": workspace,
                "config": {"sim": "sim", "robot": "robot", "camera": "camera"},
            }
        )
