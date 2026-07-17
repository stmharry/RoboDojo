"""Typed command, workflow, and process launch requests."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from robodojo.core.gpu import GpuSelector
from robodojo.core.models.common import NonNegativeInt, Port, StrictModel
from robodojo.core.models.experiment import ExperimentSpec


class PolicyExperimentBase(StrictModel):
    """Runtime controls shared by requests that may carry an experiment."""

    experiment: ExperimentSpec | None = None
    seed: NonNegativeInt = 0
    policy_gpu: GpuSelector = "auto"

    def require_experiment(self) -> ExperimentSpec:
        if self.experiment is None:
            raise ValueError("this operation requires a resolved experiment")
        return self.experiment


class PolicyExperimentRequest(PolicyExperimentBase):
    experiment: ExperimentSpec

    def policy_request(
        self,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        dry_run: bool = False,
    ) -> PolicyServerLaunchRequest:
        return PolicyServerLaunchRequest(
            experiment=self.experiment,
            seed=self.seed,
            policy_gpu=self.policy_gpu,
            host=host,
            port=port,
            dry_run=dry_run,
        )


class ExperimentRequest(PolicyExperimentRequest):
    environment_gpu: GpuSelector = "auto"


class EvaluationRequest(ExperimentRequest):
    eval_num: int | Literal["native"] | None = None
    checkpoint_label: str | None = None
    export_scene: bool = False
    export_scene_only: bool = False
    layout_id: NonNegativeInt = 0
    publish: bool = False
    dry_run: bool = False

    @field_validator("eval_num")
    @classmethod
    def positive_eval_num(cls, value: int | str | None) -> int | str | None:
        if isinstance(value, int) and value < 1:
            raise ValueError("must be positive or 'native'")
        return value


class PolicyServerLaunchRequest(PolicyExperimentRequest):
    policy_gpu: NonNegativeInt = 0
    host: str = "0.0.0.0"
    port: Port | None = None
    dry_run: bool = False


class ServerRequest(ExperimentRequest):
    host: str = "0.0.0.0"
    port: Port | None = None
    dry_run: bool = False


class PreflightRequest(ExperimentRequest):
    publish: bool = False
    deep: bool = False
    timeout: Annotated[float, Field(gt=0)] = 600.0

    def policy_request(self, *, port: int | None = None) -> PolicyServerLaunchRequest:
        return super().policy_request(port=port)


class SetupStage(StrEnum):
    ROOT = "root"
    ASSETS = "assets"
    POLICY = "policy"


class SetupRequest(PolicyExperimentBase):
    stages: tuple[SetupStage, ...] = ()

    @model_validator(mode="after")
    def required_stage_values(self) -> SetupRequest:
        selected = set(self.stages) or set(SetupStage)
        if selected.intersection({SetupStage.ASSETS, SetupStage.POLICY}) and self.experiment is None:
            raise ValueError("asset and policy setup stages require a resolved experiment")
        return self

    def selected_stages(self) -> tuple[SetupStage, ...]:
        return self.stages or tuple(SetupStage)

    def policy_request(self) -> PolicyServerLaunchRequest:
        return PolicyServerLaunchRequest(
            experiment=self.require_experiment(),
            seed=self.seed,
            policy_gpu=self.policy_gpu,
        )


class SimulatorLaunchRequest(StrictModel):
    experiment: ExperimentSpec
    policy_name: str
    host: str = "127.0.0.1"
    port: Port
    environment_gpu: NonNegativeInt = 0
    seed: NonNegativeInt = 0
    eval_num: Annotated[int, Field(ge=1)] | Literal["native"] = 1
    additional_info: str
    transport: Literal["ws"] = "ws"
    policy_server_url: str = ""
    dry_run: bool = False

    @field_validator("policy_name", "additional_info")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class SweepRequest(StrictModel):
    recipes: tuple[str, ...]
    seed: NonNegativeInt = 0
    policy_gpu: GpuSelector = "auto"
    environment_gpu: GpuSelector = "auto"
    eval_num: Annotated[int, Field(ge=1)] | Literal["native"] = 1
    limit: Annotated[int, Field(ge=1)] | None = None
    resume: bool = False
    fail_fast: bool = False
    run_id: str | None = None
    dry_run: bool = False

    @field_validator("recipes")
    @classmethod
    def non_empty_recipe_list(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("at least one recipe is required")
        if len(value) != len(set(value)):
            raise ValueError("recipe selections must be unique")
        return value


class SnapshotBatchRequest(StrictModel):
    recipes: tuple[str, ...] = ()
    seed: NonNegativeInt = 0
    environment_gpu: GpuSelector = "auto"
    layout_id: NonNegativeInt = 0
    output_dir: Path | None = None
    export_scene: bool = False
    publish: bool = False
    resume: bool = False
    fail_fast: bool = False
    dry_run: bool = False

    @field_validator("recipes")
    @classmethod
    def unique_recipe_list(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("recipe selections must be unique")
        return value

    @model_validator(mode="after")
    def resume_requires_output_dir(self) -> SnapshotBatchRequest:
        if self.resume and self.output_dir is None:
            raise ValueError("--resume requires --output-dir")
        return self


class SnapshotCaptureRequest(ExperimentRequest):
    output_dir: Path
    layout_id: NonNegativeInt = 0
    export_scene: bool = False
    run_id: str
    dry_run: bool = False

    @field_validator("run_id")
    @classmethod
    def safe_snapshot_run_id(cls, value: str) -> str:
        if not value.strip() or any(character in value for character in "/\\"):
            raise ValueError("snapshot run id must be one non-empty path component")
        return value
