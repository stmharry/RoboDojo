"""Validated command and workflow contracts."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NonNegativeInt = Annotated[int, Field(ge=0)]
Port = Annotated[int, Field(ge=1, le=65535)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DataFormat(StrEnum):
    LEROBOT_V30 = "lerobot_v3.0"
    LEROBOT_V21 = "lerobot_v2.1"
    HDF5 = "hdf5"
    DEMO = "demo"
    REAL = "real"


class EvaluationRequest(StrictModel):
    policy_dir: Path
    task: str
    checkpoint: str
    policy_env: str
    dataset: str = "RoboDojo"
    env_config: str = "arx_x5"
    scene_config: str | None = None
    expert_num: Annotated[int, Field(ge=1)] = 100
    action_type: str = "ee"
    seed: NonNegativeInt = 0
    policy_gpu: NonNegativeInt = 0
    env_gpu: NonNegativeInt = 0
    eval_num: int | Literal["native"] | None = None
    checkpoint_label: str | None = None
    export_scene: bool = False
    export_scene_only: bool = False
    export_scene_dir: Path | None = None
    layout_id: NonNegativeInt = 0
    publish: bool = False
    dry_run: bool = False

    @field_validator("task", "checkpoint", "policy_env", "dataset", "env_config", "action_type")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("scene_config")
    @classmethod
    def optional_non_empty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("eval_num")
    @classmethod
    def positive_eval_num(cls, value: int | str | None) -> int | str | None:
        if isinstance(value, int) and value < 1:
            raise ValueError("must be positive or 'native'")
        return value

    @model_validator(mode="after")
    def publish_requires_evaluation_result(self) -> EvaluationRequest:
        if self.publish and self.export_scene_only:
            raise ValueError("--publish cannot be combined with --export-scene-only")
        return self


class PolicyServerLaunchRequest(StrictModel):
    policy_dir: Path
    task: str
    checkpoint: str
    policy_env: str
    dataset: str = "RoboDojo"
    env_config: str = "arx_x5"
    action_type: str = "ee"
    seed: NonNegativeInt = 0
    policy_gpu: NonNegativeInt = 0
    host: str = "0.0.0.0"
    port: Port | None = None
    dry_run: bool = False


class SimulatorLaunchRequest(StrictModel):
    task: str
    policy_name: str
    host: str = "127.0.0.1"
    port: Port
    env_config: str = "arx_x5"
    scene_config: str | None = None
    env_gpu: NonNegativeInt = 0
    seed: NonNegativeInt = 0
    eval_num: Annotated[int, Field(ge=1)] | Literal["native"] = 1
    additional_info: str
    protocol: Literal["ws"] = "ws"
    policy_server_url: str = ""
    dry_run: bool = False

    @field_validator("scene_config")
    @classmethod
    def optional_non_empty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be empty")
        return value


class SweepRequest(EvaluationRequest):
    task: str = "__sweep__"
    only: tuple[str, ...] = ()
    tasks_file: Path | None = None
    limit: Annotated[int, Field(ge=1)] | None = None
    resume: bool = False
    fail_fast: bool = False
    run_id: str | None = None


class EnvironmentConfigReferences(StrictModel):
    sim: str
    scene: str
    robot: str
    camera: str


class EnvironmentDiagnostics(StrictModel):
    matched_replay_manifest: str | None = None


class EnvironmentConfigDocument(BaseModel):
    """Typed profile metadata with forward-compatible upstream payload fields."""

    model_config = ConfigDict(extra="allow")

    config_name: str
    layout_config_name: str | None = None
    hardware_calibration: str | None = None
    diagnostics: EnvironmentDiagnostics | None = None
    config: EnvironmentConfigReferences
    observation: dict[str, Any] = Field(default_factory=dict)


class SmokeRecord(StrictModel):
    status: Literal["PASS", "FAIL", "SKIP", "DRY_RUN"]
    task: str
    scene_config: str | None = None
    exit_code: int
    elapsed_sec: float
    result_path: str = ""
    log_path: str = ""
    message: str = ""


class SmokeSummary(StrictModel):
    run_id: str
    eval_num: int | Literal["native"]
    results: list[SmokeRecord]
