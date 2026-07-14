import os
from pathlib import Path
import subprocess
import sys

from pydantic import ValidationError
import pytest
from typer.main import get_command
from typer.testing import CliRunner

from robodojo.cli import app
from robodojo.core.models import EvaluationRequest, PolicyServerLaunchRequest, SimulatorLaunchRequest
from robodojo.core.paths import RepositoryPaths, discover_repository_root
from robodojo.core.settings import RuntimeSettings
from robodojo.policy import adapter as policy_adapter
from robodojo.policy.adapter import policy_server_command
from robodojo.sim.launcher import load_simulator_config, simulator_command
from robodojo.workflows.task_inventory import build_inventory

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def test_cli_exposes_the_unified_command_surface():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("eval", "server", "client", "smoke", "storage", "assets", "data", "docker"):
        assert command in result.stdout
    assert "upstream" not in result.stdout


def test_every_public_command_and_parameter_has_human_readable_help():
    root = get_command(app)
    visited: list[tuple[str, ...]] = []

    def visit(command, path: tuple[str, ...]) -> None:
        assert command.help and command.help.strip(), f"missing command help: {' '.join(path) or 'robodojo'}"
        for parameter in command.params:
            if not parameter.hidden:
                assert getattr(parameter, "help", None), (
                    f"missing parameter help: {' '.join(path) or 'robodojo'} {parameter.name}"
                )
        visited.append(path)
        for name, child in getattr(command, "commands", {}).items():
            if not child.hidden:
                visit(child, (*path, name))

    visit(root, ())

    for path in visited:
        result = runner.invoke(app, [*path, "--help"])
        assert result.exit_code == 0, f"help failed for {' '.join(path) or 'robodojo'}: {result.output}"


def test_eval_help_explains_publication_and_evaluation_inputs():
    result = runner.invoke(app, ["eval", "--help"])

    assert result.exit_code == 0
    assert "--publish" in result.stdout
    assert "ROBODOJO_S3_URI" in result.stdout
    assert "Checkpoint name or path" in result.stdout
    assert "positive integer" in result.stdout


def test_publish_is_incompatible_with_scene_only_export(tmp_path):
    with pytest.raises(ValidationError, match="--publish cannot be combined with --export-scene-only"):
        EvaluationRequest(
            policy_dir=tmp_path,
            task="stack_bowls",
            checkpoint="test",
            policy_env="test",
            publish=True,
            export_scene_only=True,
        )


def test_make_eval_publishes_by_default_and_allows_local_override():
    common = [
        "make",
        "-n",
        "eval",
        "POLICY_DIR=XPolicyLab/policy/demo_policy",
        "POLICY_ENV=base",
        "CKPT=demo",
    ]
    default = subprocess.run(common, cwd=ROOT, check=True, capture_output=True, text=True)
    local_only = subprocess.run(
        [*common, "PUBLISH=false"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    invalid = subprocess.run(
        ["make", "-n", "help", "PUBLISH=maybe"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert "--publish" in default.stdout
    assert "--publish" not in local_only.stdout
    assert invalid.returncode != 0
    assert "PUBLISH must be true or false" in invalid.stderr


def test_task_inventory_reads_the_simulator_task_package():
    inventory = build_inventory()
    tasks = {item["name"]: item for item in inventory["tasks"]}
    assert inventory["config_dir"] == "configs/task"
    assert tasks["stack_bowls"]["runnable"] is True


def test_removed_openarm_cloth_profile_is_rejected():
    request = SimulatorLaunchRequest(
        task="fold_clothes",
        policy_name="LeRobot_Pi05_OpenArm",
        port=19000,
        env_config="openarm_cloth_folding",
        additional_info="RoboDojo",
    )
    with pytest.raises(ValueError, match="environment config not found"):
        load_simulator_config(RepositoryPaths.resolve(ROOT), request)


def test_removed_generic_openarm_profile_is_rejected():
    request = SimulatorLaunchRequest(
        task="fold_clothes",
        policy_name="LeRobot_Pi05_OpenArm",
        port=19000,
        env_config="openarm",
        additional_info="RoboDojo",
    )
    with pytest.raises(ValueError, match="environment config not found"):
        load_simulator_config(RepositoryPaths.resolve(ROOT), request)


@pytest.mark.parametrize("profile", ["openarm_wowrobo_v1_1", "openarm_anvil_v2"])
def test_unmeasured_openarm_profiles_are_release_blocked(profile):
    request = SimulatorLaunchRequest(
        task="fold_clothes",
        policy_name="LeRobot_Pi05_OpenArm",
        port=19000,
        env_config=profile,
        additional_info="RoboDojo",
    )
    with pytest.raises(ValueError, match="calibration is not release-ready"):
        load_simulator_config(RepositoryPaths.resolve(ROOT), request)


def test_server_dry_run_validates_and_builds_adapter_argv(tmp_path):
    policy = tmp_path / "Policy"
    policy.mkdir()
    adapter = policy / "setup_eval_policy_server.sh"
    adapter.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    request = PolicyServerLaunchRequest(
        policy_dir=policy,
        task="stack_bowls",
        checkpoint="run-1",
        policy_env="policy-env",
        port=19000,
    )
    command = policy_server_command(request, 19000)
    assert command[:2] == ["bash", str(adapter)]
    assert command[-2:] == ["19000", "0.0.0.0"]


@pytest.mark.parametrize("profile", ["openarm_lerobot", "openarm_wowrobo_v1_1", "openarm_anvil_v2"])
def test_openarm_policy_uses_current_environment_profile_name(tmp_path, profile):
    policy = tmp_path / "LeRobot_Pi05_OpenArm"
    policy.mkdir()
    (policy / "setup_eval_policy_server.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    request = PolicyServerLaunchRequest(
        policy_dir=policy,
        task="fold_clothes",
        checkpoint="folding_final",
        policy_env="lerobot-pi05",
        env_config=profile,
        action_type="joint",
        port=19000,
    )
    command = policy_server_command(request, 19000)
    assert command[5] == profile


def test_server_cli_rejects_invalid_port(tmp_path):
    policy = tmp_path / "Policy"
    policy.mkdir()
    (policy / "setup_eval_policy_server.sh").write_text("", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "server",
            "--policy-dir",
            str(policy),
            "--task",
            "stack_bowls",
            "--ckpt",
            "run-1",
            "--policy-env",
            "env",
            "--policy-port",
            "70000",
            "--dry-run",
        ],
    )
    assert result.exit_code == 2
    assert "less than or equal to 65535" in result.output


def test_cli_rejects_invalid_log_level():
    result = runner.invoke(app, ["--log-level", "verbose", "tasks"])
    assert result.exit_code == 2
    assert "Invalid value for --log-level" in result.output


def test_server_dry_run_separates_diagnostics_from_command_output(tmp_path):
    policy = tmp_path / "Policy"
    policy.mkdir()
    (policy / "setup_eval_policy_server.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "--log-level",
            "INFO",
            "server",
            "--policy-dir",
            str(policy),
            "--task",
            "stack_bowls",
            "--ckpt",
            "run-1",
            "--policy-env",
            "env",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "setup_eval_policy_server.sh" in result.stdout
    assert "policy server:" not in result.stdout
    assert "INFO robodojo.policy.adapter: policy server:" in result.stderr


def test_cli_log_level_is_propagated_for_child_processes(monkeypatch, tmp_path):
    policy = tmp_path / "Policy"
    policy.mkdir()
    (policy / "setup_eval_policy_server.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    seen: list[str | None] = []
    monkeypatch.setenv("ROBODOJO_LOG_LEVEL", "WARNING")
    monkeypatch.setattr(
        policy_adapter,
        "run_policy_server",
        lambda request: seen.append(os.environ.get("ROBODOJO_LOG_LEVEL")) or 0,
    )
    result = runner.invoke(
        app,
        [
            "--log-level",
            "debug",
            "server",
            "--policy-dir",
            str(policy),
            "--task",
            "stack_bowls",
            "--ckpt",
            "run-1",
            "--policy-env",
            "env",
        ],
    )
    assert result.exit_code == 0
    assert seen == ["DEBUG"]


def test_repository_root_precedence(monkeypatch, tmp_path):
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "pyproject.toml").write_text('[project]\nname = "robodojo"\n', encoding="utf-8")
    monkeypatch.setenv("ROBODOJO_ROOT", str(fake))
    assert discover_repository_root() == fake
    assert RepositoryPaths.resolve(ROOT).root == ROOT


def test_runtime_settings_rejects_removed_dotenv_variable(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "robodojo"\n', encoding="utf-8")
    (root / ".env").write_text("ROBODOJO_EVAL_ROOT=/from-file\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ROBODOJO_EVAL_ROOT"):
        RuntimeSettings.load(RepositoryPaths.resolve(root))


def test_policy_imports_work_without_simulator_extra():
    code = (
        "import sys; from robodojo.policy.adapter import policy_server_command; "
        "assert not any(name.startswith(('isaacsim', 'isaaclab', 'torch')) for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_simulator_command_uses_the_domain_module_path():
    request = SimulatorLaunchRequest(
        task="stack_bowls",
        policy_name="TestPolicy",
        port=19000,
        env_gpu=1,
        additional_info="ckpt_name=test,action_type=ee",
    )
    command, environment = simulator_command(RepositoryPaths.resolve(ROOT), request)
    assert command[command.index("-m") + 1] == "robodojo.sim.evaluation.main"
    assert command[command.index("--policy_server_url") + 1] == "ws://127.0.0.1:19000"
    assert command[command.index("--device") + 1] == "cuda:0"
    assert command[command.index("--device_id") + 1] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"


def test_simulator_entrypoint_propagates_app_device_before_environment_creation():
    source = (ROOT / "src/robodojo/sim/evaluation/main.py").read_text(encoding="utf-8")
    assert "argparse.ArgumentParser(allow_abbrev=False)" in source
    propagation = source.index('OmegaConf.update(env_cfg, "sim.device", args_cli.device, force_add=True)')
    creation = source.index("env = create_eval_env(", propagation)

    assert propagation < creation


def test_standard_and_openarm_profiles_keep_intended_parallelism():
    paths = RepositoryPaths.resolve(ROOT)
    arx = SimulatorLaunchRequest(
        task="stack_bowls",
        policy_name="TestPolicy",
        port=19000,
        env_config="arx_x5",
        additional_info="test",
    )
    openarm = arx.model_copy(update={"env_config": "openarm_lerobot"})

    assert load_simulator_config(paths, arx)[0] == 10
    assert load_simulator_config(paths, openarm)[0] == 1
