import ast
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from robodojo.cli import app
from robodojo.core.logging import configure_logging, parse_log_level
from robodojo.sim.utils.load_file import load_object_metadata
from robodojo.workflows import results_stats, task_inventory
from robodojo.workflows.errors import ResultsError

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "robodojo"


@pytest.fixture()
def isolated_robodojo_logger(monkeypatch):
    logger = logging.getLogger("robodojo")
    prior_handlers = logger.handlers[:]
    prior_level = logger.level
    prior_propagate = logger.propagate
    logger.handlers.clear()
    monkeypatch.delenv("ROBODOJO_LOG_LEVEL", raising=False)
    yield logger
    logger.handlers[:] = prior_handlers
    logger.setLevel(prior_level)
    logger.propagate = prior_propagate


def test_log_level_precedence_and_validation(monkeypatch):
    monkeypatch.delenv("ROBODOJO_LOG_LEVEL", raising=False)
    assert parse_log_level(None) == logging.INFO
    monkeypatch.setenv("ROBODOJO_LOG_LEVEL", "warning")
    assert parse_log_level(None) == logging.WARNING
    assert parse_log_level("debug") == logging.DEBUG
    with pytest.raises(ValueError, match="expected one of"):
        parse_log_level("verbose")


def test_logging_is_idempotent_formatted_and_filtered(isolated_robodojo_logger, capsys):
    root_handlers = logging.getLogger().handlers[:]
    configure_logging("WARNING")
    configure_logging("WARNING")
    logger = logging.getLogger("robodojo.tests")
    logger.info("hidden")
    logger.warning("retry %s", 2)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "WARNING robodojo.tests: retry 2\n"
    assert (
        sum(getattr(handler, "_robodojo_console_handler", False) for handler in isolated_robodojo_logger.handlers) == 1
    )
    assert isolated_robodojo_logger.propagate is False
    assert logging.getLogger().handlers == root_handlers


def test_migrated_diagnostic_uses_error_level(isolated_robodojo_logger, capsys, tmp_path):
    metadata = tmp_path / "00001"
    metadata.mkdir()
    (metadata / "metadata.json").write_text("{invalid", encoding="utf-8")
    configure_logging("ERROR")

    assert load_object_metadata(tmp_path, 1) is None
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("ERROR robodojo.sim.utils.load_file: decoding JSON from file")


def test_debug_progress_is_opt_in(isolated_robodojo_logger, capsys):
    logger = logging.getLogger("robodojo.sim.evaluation.eval_env")
    configure_logging("INFO")
    logger.debug("env%s step: %s / %s", 0, 1, 10)
    assert capsys.readouterr().err == ""

    configure_logging("DEBUG")
    logger.debug("env%s step: %s / %s", 0, 1, 10)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "DEBUG robodojo.sim.evaluation.eval_env: env0 step: 1 / 10\n"


def test_results_stats_missing_root_is_a_domain_error(tmp_path):
    missing = tmp_path / "missing"

    with pytest.raises(ResultsError, match="Eval result directory not found"):
        results_stats.generate_score_report(results_root=missing)

    result = CliRunner().invoke(app, ["results", "stats", "--results-root", str(missing)])
    assert result.exit_code == 1
    assert result.stderr == f"Eval result directory not found: {missing}\n"


def test_task_inventory_check_keeps_json_clean(monkeypatch):
    inventory = {"tasks": [{"name": "broken_task", "runnable": False}]}
    monkeypatch.setattr(task_inventory, "build_inventory", lambda: inventory)

    result = CliRunner().invoke(app, ["catalog", "tasks", "--format", "json", "--check"])

    assert result.exit_code == 1
    assert json.loads(result.stdout) == inventory
    assert result.stderr == ""


def test_package_has_no_print_calls():
    unexpected: list[str] = []
    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(PACKAGE).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                unexpected.append(f"{relative}:{node.lineno}")
    assert unexpected == []


def test_simulator_step_progress_uses_debug_logging(caplog):
    from robodojo.sim.evaluation.services.actions import ActionsService

    class Environment(ActionsService):
        num_envs = 1
        step_lim = 3
        take_action_cnt = [0]
        end_flag = [False]
        physx_monitor_enabled = False
        interact = False
        robot_manager = SimpleNamespace(
            robot_list=[],
            control_manager=SimpleNamespace(push=lambda *_args: None),
        )
        reward_manager = SimpleNamespace(step=lambda **_kwargs: None)

        def validate_action_dict(self, _action):
            return None

        def get_action_type(self, _action):
            return "joint"

        def process_control_info(self, control_info, _env_idx):
            return [control_info]

        def have_empty(self, _env_idx_list):
            return True

        def is_episode_end(self):
            return None

    with caplog.at_level("DEBUG", logger="robodojo.sim.evaluation.services.actions"):
        Environment().take_action_batch([{}])

    assert "env0 step: 1 / 3" in caplog.messages
