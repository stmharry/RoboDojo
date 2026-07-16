"""Behavioral validation across independently owned experiment catalogs."""

from __future__ import annotations

import ast

from robodojo.core.experiments.catalogs import (
    load_policy_catalog,
    load_policy_evaluation_contract,
    load_protocol_catalog,
    load_recipe_catalog,
)
from robodojo.core.experiments.selection import ResolvedExperiment, resolve_recipe
from robodojo.core.paths import RepositoryPaths


def _runnable_task(paths: RepositoryPaths, task: str) -> tuple[bool, bool, bool]:
    module = paths.root / "src" / "robodojo" / "sim" / "tasks" / f"{task}.py"
    config = paths.task_configs / f"{task}.yml"
    class_exists = False
    if module.is_file():
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        class_exists = any(isinstance(node, ast.ClassDef) and node.name == task for node in tree.body)
    return module.is_file(), class_exists, config.is_file()


def validate_experiment_catalogs(paths: RepositoryPaths) -> list[ResolvedExperiment]:
    for policy_name, policy in load_policy_catalog(paths).policies.items():
        descriptor, descriptor_path, _ = load_policy_evaluation_contract(paths, policy)
        for environment in descriptor.training.reference_environments:
            if not (paths.environment_profiles / f"{environment}.yml").is_file():
                raise ValueError(
                    f"policy profile {policy_name!r} references unknown environment {environment!r} "
                    f"in {descriptor_path.relative_to(paths.root)}"
                )
        for scene in descriptor.training.reference_scenes:
            if not (paths.scene_profiles / f"{scene}.yml").is_file():
                raise ValueError(
                    f"policy profile {policy_name!r} references unknown scene {scene!r} "
                    f"in {descriptor_path.relative_to(paths.root)}"
                )

    for name, protocol in load_protocol_catalog(paths).protocols.items():
        module_exists, class_exists, config_exists = _runnable_task(paths, protocol.task)
        if not (module_exists and class_exists and config_exists):
            raise ValueError(
                f"task protocol {name!r} references non-runnable task {protocol.task!r}: "
                f"module={module_exists} exported_class={class_exists} config={config_exists}"
            )
        missing_scenes = [
            scene for scene in protocol.compatible_scenes if not (paths.scene_profiles / f"{scene}.yml").is_file()
        ]
        if missing_scenes:
            raise ValueError(f"task protocol {name!r} references unknown scenes: {missing_scenes}")

    return [resolve_recipe(paths, name) for name in sorted(load_recipe_catalog(paths).recipes)]
