from pathlib import Path

from pydantic import ValidationError
import pytest
import yaml

from robodojo.core.contracts import (
    load_policy_catalog,
    load_protocol_catalog,
    load_recipe_catalog,
    resolve_contract,
    resolve_recipe,
    resolve_selection,
    validate_contract_catalogs,
)
from robodojo.core.layouts import resolve_layout_set
from robodojo.core.models import EvaluationRecipeDocument, SimulatorLaunchRequest, TaskProtocolDocument
from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles import load_scene_profile
from robodojo.core.storage import assets_root
from robodojo.sim.launcher import simulator_command

ROOT = Path(__file__).resolve().parents[1]
PATHS = RepositoryPaths.resolve(ROOT)
MOLMO_LONG = "molmoact2-bimanual_yam-moonlake_office-moonlake_office_general_pickup"
PI_LONG = "pi05-bimanual_yam-moonlake_office-moonlake_office_general_pickup"


def test_contract_documents_are_strict_and_typed():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TaskProtocolDocument(
            task="general_pickup",
            layout="general_pickup",
            episode_horizon=200,
            evaluation_episodes=50,
        )
    with pytest.raises(ValidationError, match="Field required"):
        EvaluationRecipeDocument(
            policy="pi05_bimanual_yam",
            environment="bimanual_yam_moonlake_office",
            scene="moonlake_office",
        )


def test_catalogs_cover_every_task_and_match_the_upstream_semantic_lock():
    resolved = validate_contract_catalogs(PATHS)
    assert len(resolved) == 24
    protocols = load_protocol_catalog(PATHS).protocols
    task_names = {path.stem for path in (ROOT / "src/robodojo/sim/tasks").glob("*.py") if path.name != "__init__.py"}
    assert {name for name, protocol in protocols.items() if name == protocol.task} == task_names


def test_long_pickup_protocol_reuses_the_upstream_task_keyed_layout():
    protocols = load_protocol_catalog(PATHS).protocols
    canonical = protocols["general_pickup"]
    long = protocols["moonlake_office_general_pickup"]
    assert canonical.task == long.task == "general_pickup"
    assert canonical.evaluation_episodes == long.evaluation_episodes == 50
    assert canonical.episode_horizon == 200
    assert long.episode_horizon == 400
    assert canonical.compatible_scenes == []
    assert long.compatible_scenes == ["moonlake_office"]

    scene = load_scene_profile(PATHS, "moonlake_office")
    layouts = resolve_layout_set(
        config_root=PATHS.environment_configs,
        assets_root=assets_root(),
        benchmark="RoboDojo",
        layout_set=scene.document.layout_set,
        layout_source=scene.document.layout_source,
        task=long.task,
        seed=0,
    )
    assert [layout.path.name for layout in layouts.layouts] == ["general_pickup_0.json"]


def test_general_pickup_task_contract_is_upstream_and_scene_independent():
    source = (ROOT / "src/robodojo/sim/tasks/general_pickup.py").read_text(encoding="utf-8")
    config = yaml.safe_load((ROOT / "configs/task/general_pickup.yml").read_text(encoding="utf-8"))
    assert 'templates = ["Pick up the <target> by 10 cm."]' in source
    assert 'is_lift(label="target", z_threshold=0.1)' in source
    assert "self.step_lim = 200" in source
    assert config["Rigid"][0]["select_mode"] == {"nums": 1, "mode": "unique", "label": ["target"]}
    assert config["Clutter"][0]["nums"] == 10
    for forbidden in ("scene_component", "scene_config", "camera_config", "robot_config", "env_cfg_type"):
        assert forbidden not in source


def test_task_metadata_contains_no_hidden_runtime_selectors():
    index = yaml.safe_load((ROOT / "configs/task/_task.yml").read_text(encoding="utf-8"))
    forbidden = {"scene_config", "camera_config", "robot_config", "env_cfg_type"}
    assert forbidden.isdisjoint(index["common"])
    assert all(forbidden.isdisjoint(values) for values in index["tasks"].values())


def test_recipe_resolution_validates_every_compatibility_edge():
    long = resolve_recipe(PATHS, PI_LONG)
    assert long.policy.embodiment == long.environment.policy_contract == "bimanual_yam"
    assert long.scene.name == "moonlake_office"
    assert long.protocol_name == "moonlake_office_general_pickup"

    with pytest.raises(ValueError, match="requires embodiment"):
        resolve_contract(
            PATHS,
            policy_name="pi05_bimanual_yam",
            environment_name="arx_x5",
            scene_name="default",
            protocol_name="general_pickup",
        )
    with pytest.raises(ValueError, match="compatible only with scenes"):
        resolve_contract(
            PATHS,
            policy_name="pi05_bimanual_yam",
            environment_name="bimanual_yam_molmoact2",
            scene_name="molmo_yam",
            protocol_name="moonlake_office_general_pickup",
        )


def test_selection_is_recipe_or_all_four_manual_components():
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_selection(
            PATHS,
            recipe=PI_LONG,
            policy=None,
            environment=None,
            scene="moonlake_office",
            protocol=None,
        )
    with pytest.raises(ValueError, match="missing environment, protocol, scene"):
        resolve_selection(
            PATHS,
            recipe=None,
            policy="pi05_bimanual_yam",
            environment=None,
            scene=None,
            protocol=None,
        )


@pytest.mark.parametrize("recipe_name", [MOLMO_LONG, PI_LONG])
def test_long_recipe_passes_base_task_and_distinct_protocol_to_runtime(recipe_name):
    contract = resolve_recipe(PATHS, recipe_name)
    values = contract.request_values(PATHS)
    request = SimulatorLaunchRequest(
        task=values["task"],
        protocol_name=values["protocol"],
        episode_horizon=values["episode_horizon"],
        native_eval_num=values["native_eval_num"],
        recipe=values["recipe"],
        contract_hash=values["contract_hash"],
        policy_name=Path(values["policy_dir"]).name,
        port=19000,
        env_config=values["env_config"],
        scene_config=values["scene_config"],
        additional_info="test",
        dry_run=True,
    )
    command, _ = simulator_command(PATHS, request)
    assert command[command.index("--task_name") + 1] == "general_pickup"
    assert command[command.index("--protocol_name") + 1] == "moonlake_office_general_pickup"
    assert "--layout_name" not in command
    assert "--layout-name" not in command
    assert command[command.index("--episode_horizon") + 1] == "400"


def test_pickup_protocols_have_no_task_specific_scene_assets():
    scene = load_scene_profile(PATHS, "moonlake_office").document
    assert "general_pickup" not in scene.task_asset_builds
    assert "general_pickup" not in scene.task_assets
    assert scene.task_asset_builds["pack_item_into_container"] == ["moonlake_packing"]


def test_policy_profiles_hold_adapter_runtime_checkpoint_and_action_contract():
    assert load_policy_catalog(PATHS).schema_version == 2
    assert load_protocol_catalog(PATHS).schema_version == 2
    assert load_recipe_catalog(PATHS).schema_version == 2
    policies = load_policy_catalog(PATHS).policies
    pi = policies["pi05_bimanual_yam"]
    assert pi.policy_dir == Path("XPolicyLab/policy/Pi_05")
    assert pi.runtime == "uv"
    assert pi.checkpoint == "pi05_yam_molmoact2"
    assert pi.embodiment == "bimanual_yam"
    assert pi.dataset == "RoboDojo"
    assert pi.action_type == "joint"


def test_protocol_identity_owns_result_paths_and_wrapper_horizon():
    source = (ROOT / "src/robodojo/sim/evaluation/eval_env.py").read_text(encoding="utf-8")
    assert 'self.step_lim = int(self.eval_cfg["episode_horizon"])' in source
    assert "self.protocol_name," in source
    assert '"task_name": self.task_name' in source
    assert '"protocol_name": self.protocol_name' in source
