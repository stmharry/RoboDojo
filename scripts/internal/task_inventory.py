#!/usr/bin/env python3
"""List runnable RoboDojo tasks without importing Isaac-dependent task modules."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK = "RoboDojo"
TASK_DIR = ROOT_DIR / "task" / BENCHMARK / "tasks"
CONFIG_DIR = ROOT_DIR / "task" / BENCHMARK / "config"

sys.path.insert(0, str(ROOT_DIR))
from task.RoboDojo import task_registry  # noqa: E402


def _module_classes(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.name for node in tree.body if isinstance(node, ast.ClassDef)}


def _task_records() -> list[dict[str, Any]]:
    records = []
    for path in sorted(TASK_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        name = path.stem
        classes = _module_classes(path)
        config_path = task_registry.task_config_path(CONFIG_DIR, name)
        record = {
            "name": name,
            "module": str(path.relative_to(ROOT_DIR)),
            "class_name": name,
            "class_exists": name in classes,
            "config": str(config_path.relative_to(ROOT_DIR)) if config_path.exists() else None,
            "config_exists": config_path.exists(),
        }
        record["runnable"] = bool(record["class_exists"] and record["config_exists"])
        records.append(record)
    return records


def build_inventory() -> dict[str, Any]:
    tasks = _task_records()
    task_names = {task["name"] for task in tasks}
    config_names = {path.stem for path in CONFIG_DIR.glob("*.yml") if not path.name.startswith("_")}
    inventory = {
        "benchmark": BENCHMARK,
        "root": str(ROOT_DIR),
        "task_dir": str(TASK_DIR.relative_to(ROOT_DIR)),
        "config_dir": str(CONFIG_DIR.relative_to(ROOT_DIR)),
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


def _print_plain(inventory: dict[str, Any], only_runnable: bool) -> None:
    for task in inventory["tasks"]:
        if only_runnable and not task["runnable"]:
            continue
        print(task["name"])


def _print_markdown(inventory: dict[str, Any]) -> None:
    print("| Task | Runnable | Config | Issue |")
    print("| --- | --- | --- | --- |")
    for task in inventory["tasks"]:
        issues = []
        if not task["class_exists"]:
            issues.append("missing exported class")
        if not task["config_exists"]:
            issues.append("missing config")
        print(
            f"| `{task['name']}` | {'yes' if task['runnable'] else 'no'} | "
            f"`{task['config'] or '-'}` | {', '.join(issues) or '-'} |"
        )
    if inventory["config_only"]:
        print("\nConfig-only entries:")
        for name in inventory["config_only"]:
            print(f"- `{name}`")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("plain", "json", "markdown"),
        default="plain",
        help="Output format. Use json for agents and plain for shell loops.",
    )
    parser.add_argument(
        "--only-runnable",
        action="store_true",
        help="Only print runnable task names in plain output.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any task is missing its config or exported class.",
    )
    args = parser.parse_args()

    inventory = build_inventory()
    if args.format == "json":
        print(json.dumps(inventory, indent=2, sort_keys=True))
    elif args.format == "markdown":
        _print_markdown(inventory)
    else:
        _print_plain(inventory, only_runnable=args.only_runnable)

    if args.check:
        broken = [task for task in inventory["tasks"] if not task["runnable"]]
        if broken:
            for task in broken:
                print(f"[ERROR] Task is not runnable: {task['name']}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
