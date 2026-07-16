"""List runnable RoboDojo tasks without importing Isaac-dependent task modules."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
from typing import Any

from robodojo.core.paths import RepositoryPaths, discover_repository_root

ROOT_DIR = discover_repository_root()
BENCHMARK = "RoboDojo"
TASK_DIR = ROOT_DIR / "src" / "robodojo" / "sim" / "tasks"
CONFIG_DIR = RepositoryPaths.resolve(ROOT_DIR).task_configs


def _module_classes(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.name for node in tree.body if isinstance(node, ast.ClassDef)}


def _task_records(paths: RepositoryPaths | None = None) -> list[dict[str, Any]]:
    repository = paths or RepositoryPaths.resolve(ROOT_DIR)
    root = repository.root
    task_dir = root / "src" / "robodojo" / "sim" / "tasks"
    config_dir = repository.task_configs
    records = []
    for path in sorted(task_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        name = path.stem
        classes = _module_classes(path)
        config_path = config_dir / f"{name}.yml"
        record = {
            "name": name,
            "module": f"robodojo.sim.tasks.{name}",
            "class_name": name,
            "class_exists": name in classes,
            "config": str(config_path.relative_to(root)) if config_path.exists() else None,
            "config_exists": config_path.exists(),
        }
        record["runnable"] = bool(record["class_exists"] and record["config_exists"])
        records.append(record)
    return records


def build_inventory(paths: RepositoryPaths | None = None) -> dict[str, Any]:
    repository = paths or RepositoryPaths.resolve(ROOT_DIR)
    root = repository.root
    config_dir = repository.task_configs
    tasks = _task_records(repository)
    task_names = {task["name"] for task in tasks}
    config_names = {path.stem for path in config_dir.glob("*.yml") if not path.name.startswith("_")}
    inventory = {
        "benchmark": BENCHMARK,
        "root": str(root),
        "task_dir": "robodojo.sim.tasks",
        "config_dir": str(config_dir.relative_to(root)),
        "counts": {
            "tasks": len(tasks),
            "runnable": sum(1 for task in tasks if task["runnable"]),
            "missing_config": sum(1 for task in tasks if not task["config_exists"]),
            "missing_class": sum(1 for task in tasks if not task["class_exists"]),
            "config_only": len(config_names - task_names),
        },
        "tasks": tasks,
        "config_only": sorted(config_names - task_names),
    }
    return inventory


def print_plain(inventory: dict[str, Any], only_runnable: bool) -> None:
    for task in inventory["tasks"]:
        if only_runnable and not task["runnable"]:
            continue
        sys.stdout.write(f"{task['name']}\n")


def print_markdown(inventory: dict[str, Any]) -> None:
    sys.stdout.write("| Task | Runnable | Config | Issue |\n")
    sys.stdout.write("| --- | --- | --- | --- |\n")
    for task in inventory["tasks"]:
        issues = []
        if not task["class_exists"]:
            issues.append("missing exported class")
        if not task["config_exists"]:
            issues.append("missing config")
        sys.stdout.write(
            f"| `{task['name']}` | {'yes' if task['runnable'] else 'no'} | "
            f"`{task['config'] or '-'}` | {', '.join(issues) or '-'} |\n"
        )
    if inventory["config_only"]:
        sys.stdout.write("\nConfig-only entries:\n")
        for name in inventory["config_only"]:
            sys.stdout.write(f"- `{name}`\n")
