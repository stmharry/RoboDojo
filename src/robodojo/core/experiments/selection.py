"""Experiment selection, composition, and compatibility validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from robodojo.core.experiments.catalogs import (
    load_policy_catalog,
    load_policy_evaluation_contract,
    load_protocol_catalog,
    load_recipe_catalog,
)
from robodojo.core.experiments.identity import experiment_hash, task_input_hash
from robodojo.core.models.experiment import ExperimentSpec, TaskProtocolDocument
from robodojo.core.models.policy import PolicyEvaluationContract, PolicyProfileDocument
from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles.environment import EnvironmentProfile, load_environment_profile
from robodojo.core.profiles.scene import SceneProfile, load_scene_profile, validate_scene_environment_compatibility


@dataclass(frozen=True)
class ResolvedExperiment:
    name: str | None
    policy_name: str
    policy: PolicyProfileDocument
    policy_descriptor: PolicyEvaluationContract
    policy_descriptor_path: Path
    policy_descriptor_hash: str
    policy_reference_match: str
    environment: EnvironmentProfile
    scene: SceneProfile
    task_protocol: str
    protocol: TaskProtocolDocument
    identity_hash: str

    def spec(self, paths: RepositoryPaths) -> ExperimentSpec:
        return ExperimentSpec(
            policy_dir=(paths.root / self.policy.policy_dir).resolve(),
            task=self.protocol.task,
            checkpoint=self.policy.checkpoint,
            policy_profile=self.policy_name,
            policy_descriptor_hash=self.policy_descriptor_hash,
            policy_reference_match=self.policy_reference_match,
            policy_runtime=self.policy.runtime,
            dataset=self.policy_descriptor.launch.dataset,
            environment=self.environment.name,
            embodiment=self.policy_descriptor.interface.embodiment,
            scene=self.scene.name,
            action_type=self.policy_descriptor.launch.action_type,
            recipe=self.name,
            experiment_hash=self.identity_hash,
            task_protocol=self.task_protocol,
            episode_horizon=self.protocol.episode_horizon,
            evaluation_episodes=self.protocol.evaluation_episodes,
        )


def _validate_policy_environment_interface(
    policy_name: str,
    descriptor: PolicyEvaluationContract,
    environment: EnvironmentProfile,
) -> None:
    required = descriptor.interface
    actual = environment.policy_interface
    if required.embodiment != environment.embodiment:
        raise ValueError(
            f"policy profile {policy_name!r} requires embodiment {required.embodiment!r}; "
            f"environment {environment.name!r} exposes {environment.embodiment!r}"
        )
    if required.state.dimension != actual.state_dimension or required.action.dimension != actual.action_dimension:
        raise ValueError(
            f"policy profile {policy_name!r} requires state/action dimensions "
            f"{required.state.dimension}/{required.action.dimension}; environment {environment.name!r} exposes "
            f"{actual.state_dimension}/{actual.action_dimension}"
        )
    if required.action.rate_hz != actual.action_rate_hz:
        raise ValueError(
            f"policy profile {policy_name!r} requires {required.action.rate_hz} Hz actions; "
            f"environment {environment.name!r} exposes {actual.action_rate_hz} Hz"
        )
    for camera in required.cameras.required:
        source = actual.camera(camera.role)
        if source is None:
            raise ValueError(
                f"policy profile {policy_name!r} requires camera role {camera.role!r}; "
                f"environment {environment.name!r} does not expose it"
            )
        accepted = set(camera.accepted_resolutions)
        if accepted and (not source.resolutions or not set(source.resolutions).issubset(accepted)):
            raise ValueError(
                f"policy profile {policy_name!r} accepts {camera.role!r} resolutions "
                f"{sorted(accepted)}; environment {environment.name!r} exposes {list(source.resolutions)}"
            )
        if source.rate_hz is not None and source.rate_hz != required.action.rate_hz:
            raise ValueError(
                f"environment {environment.name!r} camera role {camera.role!r} runs at {source.rate_hz} Hz; "
                f"policy profile {policy_name!r} requires {required.action.rate_hz} Hz"
            )


def policy_reference_match(
    descriptor: PolicyEvaluationContract,
    environment_name: str,
    scene_name: str,
) -> str:
    training = descriptor.training
    references = bool(training.reference_environments or training.reference_scenes)
    if not references:
        return "unspecified"
    environment_matches = not training.reference_environments or environment_name in training.reference_environments
    scene_matches = not training.reference_scenes or scene_name in training.reference_scenes
    return "reference_match" if environment_matches and scene_matches else "domain_shift"


def compose_experiment(
    paths: RepositoryPaths,
    *,
    policy_name: str,
    environment_name: str,
    scene_name: str,
    task_protocol: str,
    recipe_name: str | None = None,
) -> ResolvedExperiment:
    policies = load_policy_catalog(paths).policies
    protocols = load_protocol_catalog(paths).protocols
    try:
        policy = policies[policy_name]
    except KeyError as exc:
        raise ValueError(f"unknown policy profile {policy_name!r}") from exc
    try:
        protocol = protocols[task_protocol]
    except KeyError as exc:
        raise ValueError(f"unknown task protocol {task_protocol!r}") from exc

    descriptor, descriptor_path, descriptor_hash = load_policy_evaluation_contract(paths, policy)
    environment = load_environment_profile(paths, environment_name)
    scene = load_scene_profile(paths, scene_name)
    validate_scene_environment_compatibility(scene, environment)
    _validate_policy_environment_interface(policy_name, descriptor, environment)
    if protocol.compatible_scenes and scene.name not in protocol.compatible_scenes:
        raise ValueError(
            f"task protocol {task_protocol!r} is compatible only with scenes "
            f"{protocol.compatible_scenes}; received {scene.name!r}"
        )

    task_module = paths.root / "src" / "robodojo" / "sim" / "tasks" / f"{protocol.task}.py"
    task_config = paths.task_configs / f"{protocol.task}.yml"
    if not task_module.is_file() or not task_config.is_file():
        raise ValueError(
            f"task protocol {task_protocol!r} requires runnable task {protocol.task!r}; "
            f"module={task_module.is_file()} config={task_config.is_file()}"
        )

    reference_match = policy_reference_match(descriptor, environment.name, scene.name)
    identity = experiment_hash(
        {"recipe": recipe_name or "manual"},
        {"policy": policy_name, "selection": policy.model_dump(mode="json")},
        {"policy_descriptor": descriptor.model_dump(mode="json")},
        {"environment": environment.name, "sha256": environment.identity_hash},
        {"scene": scene.name, "sha256": scene.identity_hash},
        {"task_protocol": task_protocol, "protocol": protocol.model_dump(mode="json")},
        {"task_inputs": task_input_hash(task_module, task_config)},
    )
    return ResolvedExperiment(
        name=recipe_name,
        policy_name=policy_name,
        policy=policy,
        policy_descriptor=descriptor,
        policy_descriptor_path=descriptor_path,
        policy_descriptor_hash=descriptor_hash,
        policy_reference_match=reference_match,
        environment=environment,
        scene=scene,
        task_protocol=task_protocol,
        protocol=protocol,
        identity_hash=identity,
    )


def resolve_recipe(paths: RepositoryPaths, name: str) -> ResolvedExperiment:
    if re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
        raise ValueError("recipe names must contain only letters, digits, underscores, and hyphens")
    try:
        recipe = load_recipe_catalog(paths).recipes[name]
    except KeyError as exc:
        raise ValueError(f"unknown evaluation recipe {name!r}; run 'robodojo recipes' to list valid names") from exc
    return compose_experiment(
        paths,
        policy_name=recipe.policy,
        environment_name=recipe.environment,
        scene_name=recipe.scene,
        task_protocol=recipe.task_protocol,
        recipe_name=name,
    )


def resolve_selection(
    paths: RepositoryPaths,
    *,
    recipe: str | None,
    policy: str | None,
    environment: str | None,
    scene: str | None,
    task_protocol: str | None,
) -> ResolvedExperiment:
    manual = {"policy": policy, "environment": environment, "scene": scene, "task_protocol": task_protocol}
    supplied = sorted(key for key, value in manual.items() if value is not None)
    if recipe is not None:
        if supplied:
            raise ValueError(f"--recipe cannot be combined with manual experiment fields: {', '.join(supplied)}")
        return resolve_recipe(paths, recipe)
    missing = sorted(key for key, value in manual.items() if value is None)
    if missing:
        raise ValueError(
            "select --recipe or provide the complete manual experiment: "
            "--policy-profile, --environment, --scene, and --task-protocol; "
            f"missing {', '.join(missing)}"
        )
    return compose_experiment(
        paths,
        policy_name=str(policy),
        environment_name=str(environment),
        scene_name=str(scene),
        task_protocol=str(task_protocol),
    )
