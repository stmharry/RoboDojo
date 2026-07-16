"""Policy selection, adapter, interface, and training models."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from robodojo.core.models.common import StrictModel


class PolicyProfileDocument(StrictModel):
    """One root-owned selection of a policy-owned checkpoint contract."""

    policy_dir: Path
    runtime: str
    checkpoint: str

    @field_validator("runtime", "checkpoint")
    @classmethod
    def non_empty_policy_value(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("policy_dir")
    @classmethod
    def relative_policy_dir(cls, value: Path) -> Path:
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("policy_dir must be a repository-relative path")
        return value


class PolicyLaunchContract(StrictModel):
    dataset: str
    action_type: Literal["joint", "ee"]

    @field_validator("dataset")
    @classmethod
    def non_empty_dataset(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class PolicyTensorContract(StrictModel):
    representation: Literal["joint", "ee"]
    dimension: Annotated[int, Field(ge=1)]
    frame: str
    rate_hz: Annotated[int, Field(ge=1)] | None = None

    @field_validator("frame")
    @classmethod
    def non_empty_frame(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class PolicyCameraRequirement(StrictModel):
    role: str
    accepted_resolutions: list[tuple[Annotated[int, Field(ge=1)], Annotated[int, Field(ge=1)]]]

    @field_validator("role")
    @classmethod
    def safe_role(cls, value: str) -> str:
        if re.fullmatch(r"[a-z][a-z0-9_]*", value) is None:
            raise ValueError("camera role must be lowercase snake_case")
        return value

    @field_validator("accepted_resolutions")
    @classmethod
    def unique_resolutions(
        cls,
        value: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if len(value) != len(set(value)):
            raise ValueError("accepted camera resolutions must be unique")
        return value


class PolicyCameraContract(StrictModel):
    dtype: Literal["uint8"]
    required: list[PolicyCameraRequirement]

    @field_validator("required")
    @classmethod
    def unique_roles(cls, value: list[PolicyCameraRequirement]) -> list[PolicyCameraRequirement]:
        roles = [camera.role for camera in value]
        if len(roles) != len(set(roles)):
            raise ValueError("required camera roles must be unique")
        return value


class PolicyInterfaceContract(StrictModel):
    embodiment: str
    state: PolicyTensorContract
    action: PolicyTensorContract
    cameras: PolicyCameraContract

    @field_validator("embodiment")
    @classmethod
    def safe_embodiment(cls, value: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9_]+", value) is None:
            raise ValueError("embodiment must contain only letters, digits, and underscores")
        return value

    @model_validator(mode="after")
    def compatible_state_and_action(self) -> PolicyInterfaceContract:
        if self.state.representation != self.action.representation:
            raise ValueError("state and action representations must match")
        return self


class PolicyAdapterContract(StrictModel):
    state_transform: str
    action_transform: str
    image_transform: str

    @field_validator("state_transform", "action_transform", "image_transform")
    @classmethod
    def non_empty_transform(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class PolicyExecutionContract(StrictModel):
    strategy: Literal["full_chunk", "fixed_prefix", "adaptive"]
    prediction_horizon: Annotated[int, Field(ge=1)]
    nominal_execution_horizon: Annotated[int, Field(ge=1)]
    maximum_execution_horizon: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def ordered_horizons(self) -> PolicyExecutionContract:
        if self.nominal_execution_horizon > self.maximum_execution_horizon:
            raise ValueError("nominal execution horizon exceeds maximum execution horizon")
        if self.maximum_execution_horizon > self.prediction_horizon:
            raise ValueError("maximum execution horizon exceeds prediction horizon")
        return self


class PolicyTrainingContract(StrictModel):
    dataset_id: str
    setup_id: str
    scene_id: str | None = None
    reference_environments: list[str] = Field(default_factory=list)
    reference_scenes: list[str] = Field(default_factory=list)

    @field_validator("dataset_id", "setup_id")
    @classmethod
    def non_empty_training_value(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("reference_environments", "reference_scenes")
    @classmethod
    def unique_references(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("reference profile names must be unique")
        invalid = sorted(name for name in value if re.fullmatch(r"[A-Za-z0-9_]+", name) is None)
        if invalid:
            raise ValueError(f"invalid reference profile names: {invalid}")
        return value


class PolicyEvaluationContract(StrictModel):
    launch: PolicyLaunchContract
    interface: PolicyInterfaceContract
    adapter: PolicyAdapterContract
    execution: PolicyExecutionContract
    training: PolicyTrainingContract

    @model_validator(mode="after")
    def launch_matches_interface(self) -> PolicyEvaluationContract:
        if self.launch.action_type != self.interface.action.representation:
            raise ValueError("launch action type must match the interface representation")
        return self


class PolicyEvaluationContractCatalog(StrictModel):
    schema_version: Literal[1]
    profiles: dict[str, PolicyEvaluationContract]

    @field_validator("profiles")
    @classmethod
    def safe_profile_names(cls, value: dict[str, PolicyEvaluationContract]) -> dict[str, PolicyEvaluationContract]:
        if not value:
            raise ValueError("policy evaluation contract catalog must not be empty")
        invalid = sorted(name for name in value if re.fullmatch(r"[A-Za-z0-9_-]+", name) is None)
        if invalid:
            raise ValueError(f"invalid policy evaluation contract names: {invalid}")
        return value
