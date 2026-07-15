"""Typed policy, task-protocol, and evaluation-recipe contracts."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import yaml

from robodojo.core.models import (
    EvaluationRecipeCatalog,
    PolicyProfileCatalog,
    PolicyProfileDocument,
    TaskProtocolCatalog,
    TaskProtocolDocument,
)
from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles import (
    EnvironmentProfile,
    SceneProfile,
    load_environment_profile,
    load_scene_profile,
    validate_scene_environment_compatibility,
)


@dataclass(frozen=True)
class ResolvedExperimentContract:
    name: str | None
    policy_name: str
    policy: PolicyProfileDocument
    environment: EnvironmentProfile
    scene: SceneProfile
    protocol_name: str
    protocol: TaskProtocolDocument
    identity_hash: str

    def request_values(self, paths: RepositoryPaths) -> dict[str, Any]:
        return {
            "policy_dir": (paths.root / self.policy.policy_dir).resolve(),
            "task": self.protocol.task,
            "checkpoint": self.policy.checkpoint,
            "policy_env": self.policy.runtime,
            "dataset": self.policy.dataset,
            "env_config": self.environment.name,
            "policy_contract": self.policy.embodiment,
            "scene_config": self.scene.name,
            "action_type": self.policy.action_type,
            "recipe": self.name,
            "contract_hash": self.identity_hash,
            "protocol": self.protocol_name,
            "episode_horizon": self.protocol.episode_horizon,
            "native_eval_num": self.protocol.evaluation_episodes,
        }


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"contract catalog not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_policy_catalog(paths: RepositoryPaths) -> PolicyProfileCatalog:
    return PolicyProfileCatalog.model_validate(_load_yaml(paths.policy_profiles))


def load_protocol_catalog(paths: RepositoryPaths) -> TaskProtocolCatalog:
    return TaskProtocolCatalog.model_validate(_load_yaml(paths.task_protocols))


def load_recipe_catalog(paths: RepositoryPaths) -> EvaluationRecipeCatalog:
    return EvaluationRecipeCatalog.model_validate(_load_yaml(paths.evaluation_recipes))


def _safe_recipe_name(name: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
        raise ValueError("recipe names must contain only letters, digits, underscores, and hyphens")
    return name


def _entry_hash(paths: RepositoryPaths, *values: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"robodojo-experiment-contract-v2\0")
    for path in (
        paths.policy_profiles,
        paths.task_protocols,
        paths.evaluation_recipes,
        paths.upstream_task_contracts,
    ):
        digest.update(path.relative_to(paths.root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    for value in values:
        digest.update(value.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def resolve_contract(
    paths: RepositoryPaths,
    *,
    policy_name: str,
    environment_name: str,
    scene_name: str,
    protocol_name: str,
    recipe_name: str | None = None,
) -> ResolvedExperimentContract:
    policies = load_policy_catalog(paths).policies
    protocols = load_protocol_catalog(paths).protocols
    try:
        policy = policies[policy_name]
    except KeyError as exc:
        raise ValueError(f"unknown policy profile {policy_name!r}") from exc
    try:
        protocol = protocols[protocol_name]
    except KeyError as exc:
        raise ValueError(f"unknown task protocol {protocol_name!r}") from exc

    environment = load_environment_profile(paths, environment_name)
    scene = load_scene_profile(paths, scene_name)
    validate_scene_environment_compatibility(scene, environment)
    if policy.embodiment != environment.policy_contract:
        raise ValueError(
            f"policy profile {policy_name!r} requires embodiment {policy.embodiment!r}; "
            f"environment {environment_name!r} exposes {environment.policy_contract!r}"
        )
    if protocol.compatible_scenes and scene.name not in protocol.compatible_scenes:
        raise ValueError(
            f"task protocol {protocol_name!r} is compatible only with scenes "
            f"{protocol.compatible_scenes}; received {scene.name!r}"
        )

    task_module = paths.root / "src" / "robodojo" / "sim" / "tasks" / f"{protocol.task}.py"
    task_config = paths.task_configs / f"{protocol.task}.yml"
    if not task_module.is_file() or not task_config.is_file():
        raise ValueError(
            f"task protocol {protocol_name!r} requires runnable task {protocol.task!r}; "
            f"module={task_module.is_file()} config={task_config.is_file()}"
        )

    return ResolvedExperimentContract(
        name=recipe_name,
        policy_name=policy_name,
        policy=policy,
        environment=environment,
        scene=scene,
        protocol_name=protocol_name,
        protocol=protocol,
        identity_hash=_entry_hash(
            paths,
            recipe_name or "manual",
            policy_name,
            environment_name,
            scene_name,
            protocol_name,
            environment.identity_hash,
            scene.identity_hash,
            _upstream_semantic_hash(task_module, task_config),
        ),
    )


def resolve_recipe(paths: RepositoryPaths, name: str) -> ResolvedExperimentContract:
    name = _safe_recipe_name(name)
    recipes = load_recipe_catalog(paths).recipes
    try:
        recipe = recipes[name]
    except KeyError as exc:
        raise ValueError(f"unknown evaluation recipe {name!r}; run 'robodojo recipes' to list valid names") from exc
    return resolve_contract(
        paths,
        policy_name=recipe.policy,
        environment_name=recipe.environment,
        scene_name=recipe.scene,
        protocol_name=recipe.protocol,
        recipe_name=name,
    )


def resolve_selection(
    paths: RepositoryPaths,
    *,
    recipe: str | None,
    policy: str | None,
    environment: str | None,
    scene: str | None,
    protocol: str | None,
) -> ResolvedExperimentContract:
    manual = {"policy": policy, "environment": environment, "scene": scene, "protocol": protocol}
    supplied = sorted(key for key, value in manual.items() if value is not None)
    if recipe is not None:
        if supplied:
            raise ValueError(f"--recipe cannot be combined with manual contract fields: {', '.join(supplied)}")
        return resolve_recipe(paths, recipe)
    missing = sorted(key for key, value in manual.items() if value is None)
    if missing:
        raise ValueError(
            "select --recipe or provide the complete manual contract: "
            "--policy-profile, --environment, --scene, and --protocol; "
            f"missing {', '.join(missing)}"
        )
    return resolve_contract(
        paths,
        policy_name=str(policy),
        environment_name=str(environment),
        scene_name=str(scene),
        protocol_name=str(protocol),
    )


def _task_horizon(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "step_lim" for target in node.targets)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, int)
    ]
    if len(values) != 1:
        raise ValueError(f"task must declare exactly one literal step_lim: {path}")
    return values[0]


class _StripImports(ast.NodeTransformer):
    """Ignore fork-owned import paths while hashing upstream task semantics."""

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        return None


def _upstream_semantic_hash(task_path: Path, config_path: Path) -> str:
    tree = _StripImports().visit(ast.parse(task_path.read_text(encoding="utf-8"), filename=str(task_path)))
    ast.fix_missing_locations(tree)
    payload = {
        "task_ast": ast.dump(tree, annotate_fields=True, include_attributes=False),
        "task_config": _load_yaml(config_path),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_upstream_contract_lock(
    paths: RepositoryPaths,
    *,
    canonical: set[str],
    task_dir: Path,
) -> None:
    lock = _load_yaml(paths.upstream_task_contracts)
    if lock.get("schema_version") != 1:
        raise ValueError("unsupported upstream task contract lock schema")
    upstream = lock.get("tasks", {})
    local = set(lock.get("local_tasks", ()))
    if set(upstream).intersection(local):
        raise ValueError("upstream and local task contract inventories overlap")
    if set(upstream).union(local) != canonical:
        raise ValueError(
            "upstream task contract inventory mismatch: "
            f"unlocked={sorted(canonical - set(upstream) - local)} "
            f"unknown={sorted(set(upstream).union(local) - canonical)}"
        )
    for name, expected in upstream.items():
        actual = _upstream_semantic_hash(task_dir / f"{name}.py", paths.task_configs / f"{name}.yml")
        if actual != expected:
            raise ValueError(
                f"canonical task {name!r} diverges from the semantic upstream contract "
                f"at {lock.get('revision')}: expected {expected}, found {actual}"
            )


def validate_contract_catalogs(paths: RepositoryPaths) -> list[ResolvedExperimentContract]:
    protocols = load_protocol_catalog(paths).protocols
    task_dir = paths.root / "src" / "robodojo" / "sim" / "tasks"
    task_names = {path.stem for path in task_dir.glob("*.py") if path.name != "__init__.py"}
    config_names = {path.stem for path in paths.task_configs.glob("*.yml") if not path.name.startswith("_")}
    canonical = {name for name, protocol in protocols.items() if name == protocol.task}
    if canonical != task_names or canonical != config_names:
        raise ValueError(
            "canonical protocol/task inventory mismatch: "
            f"missing_protocols={sorted(task_names - canonical)} "
            f"extra_protocols={sorted(canonical - task_names)} "
            f"config_only={sorted(config_names - task_names)}"
        )

    _validate_upstream_contract_lock(paths, canonical=canonical, task_dir=task_dir)

    task_index = _load_yaml(paths.task_configs / "_task.yml")
    forbidden_index_fields = {"scene_config", "camera_config", "robot_config", "env_cfg_type"}
    index_scopes = (("common", task_index.get("common", {})), *task_index.get("tasks", {}).items())
    for scope, values in index_scopes:
        hidden = sorted(forbidden_index_fields.intersection(values))
        if hidden:
            raise ValueError(f"task metadata {scope!r} contains forbidden runtime selectors: {hidden}")
    common = task_index.get("common", {})
    overrides = task_index.get("tasks", {})
    for name in sorted(canonical):
        protocol = protocols[name]
        source_horizon = _task_horizon(task_dir / f"{name}.py")
        if protocol.episode_horizon != source_horizon:
            raise ValueError(
                f"canonical protocol {name!r} horizon {protocol.episode_horizon} "
                f"does not match task literal {source_horizon}"
            )
        native_count = overrides.get(name, {}).get("eval_nums", common.get("eval_nums", 50))
        if protocol.evaluation_episodes != native_count:
            raise ValueError(
                f"canonical protocol {name!r} evaluation count {protocol.evaluation_episodes} "
                f"does not match upstream task metadata {native_count}"
            )

    forbidden = ("scene_component", "scene_config", "camera_config", "robot_config", "env_cfg_type")
    for name in sorted(task_names):
        source = (task_dir / f"{name}.py").read_text(encoding="utf-8")
        used = [token for token in forbidden if token in source]
        if used:
            raise ValueError(f"task {name!r} contains forbidden combination-dependent fields: {used}")

    resolved = []
    for name in sorted(load_recipe_catalog(paths).recipes):
        resolved.append(resolve_recipe(paths, name))
    return resolved


def recipe_rows(paths: RepositoryPaths) -> list[dict[str, str]]:
    return [
        {
            "recipe": contract.name or "manual",
            "policy": contract.policy_name,
            "environment": contract.environment.name,
            "scene": contract.scene.name,
            "protocol": contract.protocol_name,
            "task": contract.protocol.task,
        }
        for contract in validate_contract_catalogs(paths)
    ]
