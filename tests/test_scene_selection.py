from pathlib import Path
import subprocess

from pydantic import ValidationError
import pytest
from typer.testing import CliRunner

from robodojo.cli import app
from robodojo.core.models import (
    EvaluationRequest,
    PolicyServerLaunchRequest,
    SimulatorLaunchRequest,
    SweepRequest,
)
from robodojo.core.paths import RepositoryPaths
from robodojo.sim.launcher import resolve_scene_config, simulator_command
from robodojo.workflows import docker, sweeps

ROOT = Path(__file__).resolve().parents[1]
RUNNER = CliRunner()


def _write_minimal_config_graph(root: Path) -> RepositoryPaths:
    config_root = root / "configs"
    for directory in (
        "environment",
        "sim",
        "scene/profiles",
        "scene/components",
        "robot",
        "camera",
        "task",
    ):
        (config_root / directory).mkdir(parents=True, exist_ok=True)
    (config_root / "environment/test.yml").write_text(
        """\
config_name: test
config:
  sim: sim_config
  robot: robot_config
  camera: camera_config
""",
        encoding="utf-8",
    )
    (config_root / "sim/sim_config.yml").write_text("scene:\n  num_envs: 2\n", encoding="utf-8")
    (config_root / "robot/robot_config.yml").write_text("robots: []\n", encoding="utf-8")
    (config_root / "camera/camera_config.yml").write_text("cameras: []\n", encoding="utf-8")
    for scene in ("default", "task_scene", "explicit_scene"):
        (config_root / f"scene/profiles/{scene}.yml").write_text(
            f"config_name: {scene}\ncomponent: {scene}\nlayout_set: arx_x5\n",
            encoding="utf-8",
        )
        (config_root / f"scene/components/{scene}.yml").write_text("Room: {}\n", encoding="utf-8")
    (config_root / "task/_task.yml").write_text(
        """\
common:
  scene_config: default
tasks:
  plain: {}
  special:
    scene_config: task_scene
  profile_default:
    scene_config: default
""",
        encoding="utf-8",
    )
    return RepositoryPaths(root=root)


def _simulator_request(task: str = "plain", scene_config: str | None = None) -> SimulatorLaunchRequest:
    return SimulatorLaunchRequest(
        task=task,
        policy_name="TestPolicy",
        port=19000,
        env_config="test",
        scene_config=scene_config,
        additional_info="test",
    )


def test_scene_resolution_precedence_and_pre_isaac_validation(tmp_path):
    paths = _write_minimal_config_graph(tmp_path)

    assert resolve_scene_config(paths, _simulator_request()) == "default"
    assert resolve_scene_config(paths, _simulator_request(task="special")) == "task_scene"
    assert resolve_scene_config(paths, _simulator_request(task="profile_default")) == "default"
    assert resolve_scene_config(paths, _simulator_request(task="special", scene_config="default")) == "default"
    assert resolve_scene_config(paths, _simulator_request(scene_config="explicit_scene")) == "explicit_scene"

    with pytest.raises(ValueError, match="scene profile not found"):
        simulator_command(paths, _simulator_request(scene_config="missing"))
    with pytest.raises(ValueError, match="letters, digits, and underscores"):
        simulator_command(paths, _simulator_request(scene_config="../outside"))


def test_simulator_argv_carries_only_the_resolved_scene(tmp_path):
    paths = _write_minimal_config_graph(tmp_path)
    command, _ = simulator_command(paths, _simulator_request(task="special"))
    assert command[command.index("--scene_config") + 1] == "task_scene"


def test_scene_is_not_part_of_the_policy_server_contract(tmp_path):
    with pytest.raises(ValidationError, match="scene_config"):
        PolicyServerLaunchRequest(
            policy_dir=tmp_path,
            task="plain",
            checkpoint="checkpoint",
            policy_env="policy-env",
            scene_config="explicit_scene",
        )


def test_eval_dry_run_keeps_scene_out_of_policy_argv(tmp_path):
    policy = tmp_path / "TestPolicy"
    policy.mkdir()
    (policy / "setup_eval_policy_server.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    result = RUNNER.invoke(
        app,
        [
            "eval",
            "--policy-dir",
            str(policy),
            "--task",
            "stack_bowls",
            "--ckpt",
            "checkpoint",
            "--policy-env",
            "policy-env",
            "--policy-gpu",
            "0",
            "--env-gpu",
            "1",
            "--scene",
            "molmo_yam",
            "--dry-run",
            "--root",
            str(ROOT),
        ],
    )
    assert result.exit_code == 0, result.output
    policy_line = next(line for line in result.stdout.splitlines() if "setup_eval_policy_server.sh" in line)
    simulator_line = next(line for line in result.stdout.splitlines() if "robodojo.sim.evaluation.main" in line)
    assert "scene_config" not in policy_line
    assert "--scene_config molmo_yam" in simulator_line


@pytest.mark.parametrize("command", ["eval", "client", "smoke", "benchmark", "doctor"])
def test_simulator_launching_commands_expose_scene_option(command):
    result = RUNNER.invoke(app, [command, "--help"])
    assert result.exit_code == 0
    assert "--scene" in result.stdout


def test_docker_smoke_propagates_scene(monkeypatch, tmp_path):
    seen: list[list[str]] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: seen.append(command) or subprocess.CompletedProcess(command, 0),
    )
    assert docker.smoke(RepositoryPaths(root=tmp_path), "image", "plain", "Policy", 19000, "test", "molmo_yam") == 0
    assert seen[0][seen[0].index("--scene") + 1] == "molmo_yam"


def test_sweep_resume_identity_includes_resolved_scene(monkeypatch, tmp_path):
    from robodojo.workflows import preflight as preflight_workflow

    monkeypatch.setattr(sweeps, "run_work_root", lambda: tmp_path)
    monkeypatch.setattr(sweeps, "_selected_tasks", lambda request: ["stack_bowls"])
    monkeypatch.setattr(
        preflight_workflow,
        "run_fast_preflight",
        lambda paths, request: preflight_workflow.build_report(
            [preflight_workflow.PreflightCheck(name="test", status="PASS", detail="ok")]
        ),
    )
    calls: list[EvaluationRequest] = []
    monkeypatch.setattr(
        sweeps,
        "run_evaluation",
        lambda paths, request, *, preflight: calls.append(request) or 0,
    )
    base = SweepRequest(
        policy_dir=tmp_path / "TestPolicy",
        checkpoint="checkpoint",
        policy_env="policy-env",
        scene_config="molmo_yam",
        policy_gpu=0,
        env_gpu=1,
        run_id="scene-resume",
    )

    assert sweeps.run_sweep(RepositoryPaths(root=ROOT), base) == 0
    assert calls[-1].scene_config == "molmo_yam"
    assert sweeps.run_sweep(RepositoryPaths(root=ROOT), base.model_copy(update={"resume": True})) == 0
    assert len(calls) == 1
    changed = base.model_copy(update={"resume": True, "scene_config": "default"})
    assert sweeps.run_sweep(RepositoryPaths(root=ROOT), changed) == 0
    assert len(calls) == 2
    assert calls[-1].scene_config == "default"
