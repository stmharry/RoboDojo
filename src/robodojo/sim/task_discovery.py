"""Runtime discovery for task modules and their aligned exports."""

from __future__ import annotations

import importlib
import importlib.util


def load_task_class(task: str):
    module_name = f"robodojo.sim.tasks.{task}"
    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(f"Could not resolve task {task!r}")
    module = importlib.import_module(module_name)
    task_class = getattr(module, task, None)
    if task_class is None:
        raise ModuleNotFoundError(f"Could not resolve task class {task!r}")
    return task, task_class
