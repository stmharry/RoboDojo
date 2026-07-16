from pathlib import Path

from pydantic import ValidationError
import pytest
import yaml

from robodojo.core.experiments.catalogs import (
    load_policy_catalog,
    load_protocol_catalog,
    load_recipe_catalog,
)
from robodojo.core.experiments.identity import experiment_hash, task_input_hash
from robodojo.core.experiments.selection import (
    compose_experiment,
    resolve_recipe,
    resolve_selection,
)
from robodojo.core.experiments.validation import validate_experiment_catalogs
from robodojo.core.layouts import resolve_layout_set
from robodojo.core.models.experiment import (
    EvaluationRecipeDocument,
    TaskProtocolDocument,
)
from robodojo.core.models.requests import SimulatorLaunchRequest
from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles.scene import load_scene_profile
from robodojo.core.storage import assets_root
from robodojo.sim.launcher import simulator_command
from robodojo.workflows.task_inventory import build_inventory

ROOT = Path(__file__).resolve().parents[1]
PATHS = RepositoryPaths.resolve(ROOT)
MOLMO_LONG = "molmoact2-bimanual_yam-moonlake_office-moonlake_office_general_pickup"
PI_LONG = "pi05-bimanual_yam-moonlake_office-moonlake_office_general_pickup"
PI_PICKUP_LONG = "pi05_pickup-bimanual_yam-moonlake_office-moonlake_office_general_pickup"


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


def test_catalogs_validate_without_requiring_protocol_coverage_for_every_task():
    resolved = validate_experiment_catalogs(PATHS)
    assert len(resolved) == 26


def test_runnable_task_discovery_does_not_require_a_protocol(tmp_path):
    root = tmp_path / "checkout"
    (root / "src/robodojo/sim/tasks").mkdir(parents=True)
    (root / "configs/task").mkdir(parents=True)
    (root / "src/robodojo/sim/tasks/new_upstream_task.py").write_text(
        "class new_upstream_task:\n    pass\n",
        encoding="utf-8",
    )
    (root / "configs/task/new_upstream_task.yml").write_text("Rigid: []\n", encoding="utf-8")

    inventory = build_inventory(RepositoryPaths(root=root))

    assert inventory["counts"]["runnable"] == 1
    assert inventory["tasks"][0]["name"] == "new_upstream_task"
    assert inventory["tasks"][0]["runnable"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("task", "missing_task", "non-runnable task"),
        ("compatible_scenes", ["missing_scene"], "unknown scenes"),
    ],
)
def test_protocols_reject_missing_task_or_scene(monkeypatch, field, value, message):
    import robodojo.core.experiments.validation as validation

    catalog = load_protocol_catalog(PATHS)
    protocols = dict(catalog.protocols)
    protocols["invalid_reference"] = protocols["general_pickup"].model_copy(update={field: value})
    monkeypatch.setattr(
        validation,
        "load_protocol_catalog",
        lambda _paths: catalog.model_copy(update={"protocols": protocols}),
    )

    with pytest.raises(ValueError, match=message):
        validate_experiment_catalogs(PATHS)


def test_long_pickup_protocol_reuses_the_upstream_task_keyed_layout():
    protocols = load_protocol_catalog(PATHS).protocols
    canonical = protocols["general_pickup"]
    long = protocols["moonlake_office_general_pickup"]
    assert canonical.task == long.task == "general_pickup"
    assert canonical.evaluation_episodes == 50
    assert long.evaluation_episodes == 20
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
    assert [layout.path.name for layout in layouts.layouts] == [f"general_pickup_{index}.json" for index in range(20)]

    layout_payloads = [yaml.safe_load(layout.path.read_text(encoding="utf-8")) for layout in layouts.layouts]
    original = layout_payloads[0]["Rigid"]["ball"][0]
    assert original["default_pos"] == pytest.approx([0.22, -0.05, 0.7533537298662597])
    assert original["visual"]["color"] == pytest.approx([0.35, 0.65, 0.08])

    expected = []
    for y_position, color in (
        (-0.10, [0.75, 0.12, 0.08]),
        (-0.04, [0.90, 0.72, 0.08]),
        (0.02, [0.35, 0.65, 0.08]),
        (0.08, [0.08, 0.30, 0.80]),
    ):
        for x_position in (-0.22, -0.11, 0.0, 0.11, 0.22):
            if (x_position, y_position) != (0.22, -0.04):
                expected.append((x_position, y_position, color))

    for payload, (x_position, y_position, color) in zip(layout_payloads[1:], expected, strict=True):
        target = payload["Rigid"]["ball"][0]
        assert target["default_pos"] == pytest.approx([x_position, y_position, original["default_pos"][2]])
        assert target["visual"]["color"] == pytest.approx(color)
        assert target["label"] == "target"


def test_general_pickup_task_config_remains_scene_independent():
    config = yaml.safe_load((ROOT / "configs/task/general_pickup.yml").read_text(encoding="utf-8"))
    assert config["Rigid"][0]["select_mode"] == {"nums": 1, "mode": "unique", "label": ["target"]}
    assert config["Clutter"][0]["nums"] == 10


def test_task_metadata_contains_no_hidden_runtime_selectors():
    index = yaml.safe_load((ROOT / "configs/task/_task.yml").read_text(encoding="utf-8"))
    forbidden = {"scene_config", "camera_config", "robot_config", "env_cfg_type"}
    assert forbidden.isdisjoint(index["common"])
    assert all(forbidden.isdisjoint(values or {}) for values in index["tasks"].values())


def test_recipe_resolution_validates_every_compatibility_edge():
    base = resolve_recipe(PATHS, PI_LONG)
    assert base.policy_name == "pi05_bimanual_yam"
    assert base.policy.checkpoint == "pi05_yam_molmoact2"
    assert base.policy_descriptor.execution.strategy == "fixed_prefix"
    assert base.policy_descriptor.adapter.image_transform == "pi05_yam_molmoact2_640x360_center_crop_v1"
    assert base.policy_reference_match == "domain_shift"

    pickup = resolve_recipe(PATHS, PI_PICKUP_LONG)
    assert pickup.policy_name == "pi05_bimanual_yam_pickup"
    assert pickup.policy.checkpoint == "pi05_yam_abc_pickplace"
    assert pickup.policy_descriptor.execution.strategy == "adaptive"
    assert pickup.policy_reference_match == "reference_match"

    for experiment in (base, pickup):
        assert experiment.policy_descriptor.interface.embodiment == experiment.environment.embodiment == "bimanual_yam"
        assert experiment.scene.name == "moonlake_office"
        assert experiment.task_protocol == "moonlake_office_general_pickup"
        assert experiment.protocol.evaluation_episodes == 20

    with pytest.raises(ValueError, match="requires embodiment"):
        compose_experiment(
            PATHS,
            policy_name="pi05_bimanual_yam",
            environment_name="arx_x5",
            scene_name="default",
            task_protocol="general_pickup",
        )
    with pytest.raises(ValueError, match="compatible only with scenes"):
        compose_experiment(
            PATHS,
            policy_name="pi05_bimanual_yam",
            environment_name="bimanual_yam_molmoact2",
            scene_name="molmo_yam",
            task_protocol="moonlake_office_general_pickup",
        )


def test_selection_is_recipe_or_all_four_manual_components():
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_selection(
            PATHS,
            recipe=PI_LONG,
            policy=None,
            environment=None,
            scene="moonlake_office",
            task_protocol=None,
        )
    with pytest.raises(ValueError, match="missing environment, scene, task_protocol"):
        resolve_selection(
            PATHS,
            recipe=None,
            policy="pi05_bimanual_yam",
            environment=None,
            scene=None,
            task_protocol=None,
        )


@pytest.mark.parametrize("recipe_name", [MOLMO_LONG, PI_LONG, PI_PICKUP_LONG])
def test_long_recipe_passes_base_task_and_distinct_protocol_to_runtime(recipe_name):
    experiment = resolve_recipe(PATHS, recipe_name)
    request = SimulatorLaunchRequest(
        experiment=experiment.spec(PATHS),
        policy_name=experiment.policy.policy_dir.name,
        port=19000,
        additional_info="test",
        dry_run=True,
    )
    command, _ = simulator_command(PATHS, request)
    assert command[command.index("--task") + 1] == "general_pickup"
    assert command[command.index("--task-protocol") + 1] == "moonlake_office_general_pickup"
    assert "--layout_name" not in command
    assert "--layout-name" not in command
    assert command[command.index("--episode-horizon") + 1] == "400"


def test_pickup_protocols_have_no_task_specific_scene_assets():
    scene = load_scene_profile(PATHS, "moonlake_office").document
    assert "general_pickup" not in scene.task_asset_builds
    assert "general_pickup" not in scene.task_assets
    assert scene.task_asset_builds["pack_item_into_container"] == ["moonlake_packing"]


def test_policy_profiles_hold_adapter_runtime_checkpoint_and_action_contract():
    assert load_policy_catalog(PATHS).schema_version == 3
    assert load_protocol_catalog(PATHS).schema_version == 2
    assert load_recipe_catalog(PATHS).schema_version == 3
    policies = load_policy_catalog(PATHS).policies
    pi = policies["pi05_bimanual_yam"]
    assert pi.policy_dir == Path("XPolicyLab/policy/Pi_05")
    assert pi.runtime == "uv"
    assert pi.checkpoint == "pi05_yam_molmoact2"
    resolved_pi = resolve_recipe(PATHS, "pi05-bimanual_yam-molmo_yam-general_pickup")
    assert resolved_pi.policy_descriptor.interface.embodiment == "bimanual_yam"
    assert resolved_pi.policy_descriptor.launch.dataset == "RoboDojo"
    assert resolved_pi.policy_descriptor.launch.action_type == "joint"
    assert resolved_pi.policy_descriptor.execution.model_dump() == {
        "strategy": "fixed_prefix",
        "prediction_horizon": 16,
        "nominal_execution_horizon": 8,
        "maximum_execution_horizon": 8,
    }

    pickup = policies["pi05_bimanual_yam_pickup"]
    assert pickup.policy_dir == Path("XPolicyLab/policy/Pi_05")
    assert pickup.runtime == "uv"
    assert pickup.checkpoint == "pi05_yam_abc_pickplace"
    resolved_pickup = resolve_recipe(PATHS, PI_PICKUP_LONG)
    assert resolved_pickup.policy_descriptor.interface.embodiment == "bimanual_yam"
    assert resolved_pickup.policy_descriptor.execution.strategy == "adaptive"
    assert resolved_pickup.policy_descriptor.execution.maximum_execution_horizon == 50
    assert resolved_pickup.policy_reference_match == "reference_match"


def test_local_task_inputs_contribute_to_experiment_identity(tmp_path):
    module = tmp_path / "example.py"
    config = tmp_path / "example.yml"
    module.write_text("class example:\n    pass\n", encoding="utf-8")
    config.write_text("Rigid: []\n", encoding="utf-8")
    original = experiment_hash({"task_inputs": task_input_hash(module, config)})

    config.write_text("Rigid:\n  - name: cube\n", encoding="utf-8")
    config_changed = experiment_hash({"task_inputs": task_input_hash(module, config)})
    module.write_text("class example:\n    version = 2\n", encoding="utf-8")
    module_changed = experiment_hash({"task_inputs": task_input_hash(module, config)})

    assert len({original, config_changed, module_changed}) == 3
