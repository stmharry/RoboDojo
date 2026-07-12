import os
from pathlib import Path
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from robodojo.cli import app
from robodojo.core.models import PolicyServerLaunchRequest, SimulatorLaunchRequest
from robodojo.core.paths import RepositoryPaths, discover_repository_root
from robodojo.core.settings import RuntimeSettings
from robodojo.policy import adapter as policy_adapter
from robodojo.policy.adapter import policy_server_command
from robodojo.sim.launcher import simulator_command
from robodojo.workflows.task_inventory import build_inventory

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def test_cli_exposes_the_unified_command_surface():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("eval", "server", "client", "smoke", "storage", "assets", "data", "docker"):
        assert command in result.stdout


def test_task_inventory_reads_the_simulator_task_package():
    tasks = {item["name"]: item for item in build_inventory()["tasks"]}
    assert tasks["stack_bowls"]["runnable"] is True


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
        additional_info="ckpt_name=test,action_type=ee",
    )
    command, environment = simulator_command(RepositoryPaths.resolve(ROOT), request)
    assert command[command.index("-m") + 1] == "robodojo.sim.evaluation.main"
    assert command[command.index("--policy_server_url") + 1] == "ws://127.0.0.1:19000"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
