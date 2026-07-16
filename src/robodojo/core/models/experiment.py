"""Task protocol and evaluation recipe catalog models."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, field_validator

from robodojo.core.models.common import StrictModel
from robodojo.core.models.policy import PolicyProfileDocument


class ExperimentSpec(StrictModel):
    """Fully resolved, immutable experiment composition shared by workflows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_dir: Path
    task: str
    checkpoint: str
    policy_profile: str
    policy_descriptor_hash: str | None = None
    policy_reference_match: Literal["reference_match", "domain_shift", "unspecified"] = "unspecified"
    policy_runtime: str
    dataset: str = "RoboDojo"
    environment: str
    embodiment: str
    scene: str
    action_type: Literal["joint", "ee"]
    recipe: str | None = None
    experiment_hash: str | None = None
    task_protocol: str
    episode_horizon: Annotated[int, Field(ge=1)]
    evaluation_episodes: Annotated[int, Field(ge=1)]

    @field_validator(
        "task",
        "checkpoint",
        "policy_profile",
        "policy_runtime",
        "dataset",
        "environment",
        "embodiment",
        "scene",
        "task_protocol",
    )
    @classmethod
    def non_empty_value(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class TaskProtocolDocument(StrictModel):
    """Named benchmark settings layered over an unchanged task MDP."""

    task: str
    episode_horizon: Annotated[int, Field(ge=1)]
    evaluation_episodes: Annotated[int, Field(ge=1)]
    compatible_scenes: list[str] = Field(default_factory=list)

    @field_validator("task")
    @classmethod
    def safe_protocol_component(cls, value: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9_]+", value) is None:
            raise ValueError("must contain only letters, digits, and underscores")
        return value

    @field_validator("compatible_scenes")
    @classmethod
    def safe_protocol_scenes(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("compatible scenes must be unique")
        invalid = sorted(scene for scene in value if re.fullmatch(r"[A-Za-z0-9_]+", scene) is None)
        if invalid:
            raise ValueError(f"invalid compatible scene names: {invalid}")
        return value


class EvaluationRecipeDocument(StrictModel):
    """Explicit composition of policy, embodiment setup, scene, and protocol."""

    policy: str
    environment: str
    scene: str
    task_protocol: str

    @field_validator("policy", "environment", "scene", "task_protocol")
    @classmethod
    def safe_recipe_component(cls, value: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9_]+", value) is None:
            raise ValueError("must contain only letters, digits, and underscores")
        return value


class PolicyProfileCatalog(StrictModel):
    schema_version: Literal[3]
    policies: dict[str, PolicyProfileDocument]

    @field_validator("policies")
    @classmethod
    def safe_policy_names(cls, value: dict[str, PolicyProfileDocument]) -> dict[str, PolicyProfileDocument]:
        if not value:
            raise ValueError("policy catalog must not be empty")
        invalid = sorted(name for name in value if re.fullmatch(r"[A-Za-z0-9_]+", name) is None)
        if invalid:
            raise ValueError(f"invalid policy profile names: {invalid}")
        return value


class TaskProtocolCatalog(StrictModel):
    schema_version: Literal[2]
    protocols: dict[str, TaskProtocolDocument]

    @field_validator("protocols")
    @classmethod
    def safe_protocol_names(cls, value: dict[str, TaskProtocolDocument]) -> dict[str, TaskProtocolDocument]:
        if not value:
            raise ValueError("protocol catalog must not be empty")
        invalid = sorted(name for name in value if re.fullmatch(r"[A-Za-z0-9_]+", name) is None)
        if invalid:
            raise ValueError(f"invalid protocol names: {invalid}")
        return value


class EvaluationRecipeCatalog(StrictModel):
    schema_version: Literal[3]
    recipes: dict[str, EvaluationRecipeDocument]

    @field_validator("recipes")
    @classmethod
    def safe_recipe_names(cls, value: dict[str, EvaluationRecipeDocument]) -> dict[str, EvaluationRecipeDocument]:
        if not value:
            raise ValueError("recipe catalog must not be empty")
        invalid = sorted(name for name in value if re.fullmatch(r"[A-Za-z0-9_-]+", name) is None)
        if invalid:
            raise ValueError(f"invalid recipe names: {invalid}")
        return value
