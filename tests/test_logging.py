import ast
import logging
from pathlib import Path

import pytest

from robodojo.core.logging import configure_logging, parse_log_level
from robodojo.sim.utils.load_file import load_object_metadata

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


def _function_names(tree: ast.AST) -> dict[ast.AST, str | None]:
    names: dict[ast.AST, str | None] = {}

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.function: str | None = None

        def visit_FunctionDef(self, node):
            prior = self.function
            self.function = node.name
            self.generic_visit(node)
            self.function = prior

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            names[node] = self.function
            self.generic_visit(node)

    Visitor().visit(tree)
    return names


def test_print_calls_are_limited_to_approved_output_paths():
    renderer_files = {
        "sim/utils/update_embodiment_config_path.py",
        "workflows/assets_openarm.py",
        "workflows/doctor.py",
        "workflows/results_stats.py",
        "workflows/results_summary.py",
        "workflows/storage.py",
        "workflows/task_inventory.py",
    }
    dry_run_files = {
        "orchestration/evaluation.py",
        "policy/adapter.py",
        "sim/launcher.py",
    }
    unexpected: list[str] = []
    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(PACKAGE).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function_names = _function_names(tree)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"):
                continue
            approved = relative in renderer_files
            if relative == "workflows/downloads.py":
                approved = function_names[node] == "list_data"
            elif relative == "sim/evaluation/eval_env.py":
                approved = any(
                    keyword.arg == "end" and isinstance(keyword.value, ast.Constant) and keyword.value.value == "\r"
                    for keyword in node.keywords
                )
            elif relative in dry_run_files:
                approved = bool(
                    node.args
                    and isinstance(node.args[0], ast.Call)
                    and isinstance(node.args[0].func, ast.Name)
                    and node.args[0].func.id == "format_command"
                )
            if not approved:
                unexpected.append(f"{relative}:{node.lineno}")
    assert unexpected == []
