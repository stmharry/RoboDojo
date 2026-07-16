from pathlib import Path

import pytest
from typer.testing import CliRunner

from robodojo.cli import app
from robodojo.core.models.experiment import ExperimentSpec
from robodojo.core.models.requests import (
    PolicyServerLaunchRequest,
    SimulatorLaunchRequest,
    SweepRequest,
)
from robodojo.core.paths import RepositoryPaths
from robodojo.policy.adapter import policy_server_command
from robodojo.sim.launcher import resolve_scene_name, simulator_command
from robodojo.workflows import sweeps

ROOT = Path(__file__).resolve().parents[1]
RUNNER = CliRunner()
PI_RECIPE = "pi05-bimanual_yam-molmo_yam-general_pickup"
MOLMO_RECIPE = "molmoact2-bimanual_yam-molmo_yam-general_pickup"


def _write_minimal_config_graph(root: Path) -> RepositoryPaths:
    config_root = root / "configs"
    for directory in ("environment", "sim", "scene/profiles", "scene/components", "robot", "camera", "task"):
        (config_root / directory).mkdir(parents=True, exist_ok=True)
    (config_root / "environment/test.yml").write_text(
        "config_name: test\nembodiment: test\nconfig:\n  sim: sim\n  robot: robot\n  camera: camera\n",
        encoding="utf-8",
    )
    (config_root / "sim/sim.yml").write_text("scene:\n  num_envs: 1\n", encoding="utf-8")
    (config_root / "robot/robot.yml").write_text("robots: []\n", encoding="utf-8")
    (config_root / "camera/camera.yml").write_text("cameras: []\n", encoding="utf-8")
    for scene in ("task_scene", "explicit_scene"):
        (config_root / f"scene/profiles/{scene}.yml").write_text(
            f"config_name: {scene}\ncomponent: {scene}\nlayout_set: test\n",
            encoding="utf-8",
        )
        (config_root / f"scene/components/{scene}.yml").write_text("Room: {}\n", encoding="utf-8")
    (config_root / "task/_task.yml").write_text(
        "common:\n  scene_config: task_scene\ntasks:\n  special:\n    scene_config: task_scene\n",
        encoding="utf-8",
    )
    return RepositoryPaths(root=root)


def _request(scene: str = "explicit_scene") -> SimulatorLaunchRequest:
    return SimulatorLaunchRequest(
        experiment=ExperimentSpec(
            policy_dir=ROOT / "XPolicyLab/policy/TestPolicy",
            task="special",
            checkpoint="checkpoint",
            policy_profile="test",
            policy_runtime="policy-env",
            environment="test",
            embodiment="test",
            scene=scene,
            action_type="joint",
            task_protocol="special_protocol",
            episode_horizon=400,
            evaluation_episodes=50,
        ),
        policy_name="TestPolicy",
        port=19000,
        additional_info="test",
    )


def test_scene_resolution_uses_only_the_explicit_request(tmp_path):
    paths = _write_minimal_config_graph(tmp_path)
    assert resolve_scene_name(paths, _request()) == "explicit_scene"
    command, _ = simulator_command(paths, _request())
    assert command[command.index("--scene") + 1] == "explicit_scene"
    assert command[command.index("--task-protocol") + 1] == "special_protocol"
    assert command[command.index("--task") + 1] == "special"
    assert "--layout_name" not in command
    assert "--layout-name" not in command


def test_invalid_explicit_scene_fails_before_isaac(tmp_path):
    paths = _write_minimal_config_graph(tmp_path)
    with pytest.raises(ValueError, match="scene profile not found"):
        simulator_command(paths, _request("missing"))
    with pytest.raises(ValueError, match="letters, digits, and underscores"):
        simulator_command(paths, _request("../outside"))


def test_scene_is_not_passed_across_the_policy_adapter_boundary(tmp_path):
    policy = tmp_path / "Policy"
    policy.mkdir()
    adapter = policy / "setup_eval_policy_server.sh"
    adapter.touch()
    experiment = _request().experiment.model_copy(update={"policy_dir": policy})

    command = policy_server_command(PolicyServerLaunchRequest(experiment=experiment, port=19000), 19000)

    assert "explicit_scene" not in command


@pytest.mark.parametrize("command", ["eval", "client", "doctor", "preflight", "server"])
def test_manual_commands_expose_all_four_contract_components(command):
    result = RUNNER.invoke(app, [command, "--help"])
    assert result.exit_code == 0
    for option in ("--policy-profile", "--environment", "--scene", "--task-protocol"):
        assert option in result.stdout


@pytest.mark.parametrize("command", ["smoke", "benchmark"])
def test_sweeps_expose_recipes_without_component_cross_products(command):
    result = RUNNER.invoke(app, [command, "--help"])
    assert result.exit_code == 0
    assert "--recipe" in result.stdout
    assert "--scene" not in result.stdout
    assert "--task-protocol" not in result.stdout


def test_recipe_cannot_be_combined_with_component_override():
    result = RUNNER.invoke(
        app,
        ["eval", "--recipe", PI_RECIPE, "--scene", "moonlake_office", "--root", str(ROOT)],
    )
    assert result.exit_code == 2
    assert "--recipe cannot be combined" in result.output


def test_sweep_resume_identity_is_the_recipe(monkeypatch, tmp_path):
    monkeypatch.setattr(sweeps, "run_work_root", lambda: tmp_path)
    calls = []
    monkeypatch.setattr(sweeps, "run_evaluation", lambda paths, request, *, preflight: calls.append(request) or 0)
    base = SweepRequest(
        recipes=(PI_RECIPE,),
        policy_gpu=0,
        environment_gpu=1,
        run_id="recipe-resume",
    )

    assert sweeps.run_sweep(RepositoryPaths.resolve(ROOT), base) == 0
    assert calls[-1].experiment.recipe == PI_RECIPE
    assert sweeps.run_sweep(RepositoryPaths.resolve(ROOT), base.model_copy(update={"resume": True})) == 0
    assert len(calls) == 1
    changed = base.model_copy(update={"recipes": (MOLMO_RECIPE,), "resume": True})
    assert sweeps.run_sweep(RepositoryPaths.resolve(ROOT), changed) == 0
    assert len(calls) == 2
    assert calls[-1].experiment.recipe == MOLMO_RECIPE
