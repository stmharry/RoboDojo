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


class PreflightRequest(StrictModel):
    policy_dir: Path
    task: str
    checkpoint: str
    policy_env: str
    dataset: str = "RoboDojo"
    env_config: str = "arx_x5"
    scene_config: str | None = None
    action_type: str = "ee"
    seed: NonNegativeInt = 0
    policy_gpu: NonNegativeInt = 0
    env_gpu: NonNegativeInt = 0
    publish: bool = False
    deep: bool = False
    timeout: Annotated[float, Field(gt=0)] = 600.0

    @field_validator("task", "checkpoint", "policy_env", "dataset", "env_config", "action_type")
    @classmethod
    def required_value(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("scene_config")
    @classmethod
    def optional_scene(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be empty")
        return value

    def policy_request(self, *, port: int | None = None) -> PolicyServerLaunchRequest:
        return PolicyServerLaunchRequest(
            policy_dir=self.policy_dir,
            task=self.task,
            checkpoint=self.checkpoint,
            policy_env=self.policy_env,
            dataset=self.dataset,
            env_config=self.env_config,
            action_type=self.action_type,
            seed=self.seed,
            policy_gpu=self.policy_gpu,
            host="127.0.0.1",
            port=port,
        )


class PreflightCheck(StrictModel):
    name: str
    status: Literal["PASS", "WARN", "FAIL"]
    detail: str
    remediation: str | None = None


class PreflightReport(StrictModel):
    status: Literal["PASS", "WARN", "FAIL"]
    checks: list[PreflightCheck]


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
    robot: str
    camera: str


class EnvironmentDiagnostics(StrictModel):
    matched_replay_manifest: str | None = None


class EnvironmentConfigDocument(BaseModel):
    """Typed profile metadata with forward-compatible upstream payload fields."""

    model_config = ConfigDict(extra="allow")

    config_name: str
    hardware_calibration: str | None = None
    diagnostics: EnvironmentDiagnostics | None = None
    config: EnvironmentConfigReferences
    observation: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def reject_scene_owned_fields(cls, value: Any) -> Any:
        if isinstance(value, dict):
            legacy = sorted({"layout_config_name", "task_instruction_overrides"}.intersection(value))
            if legacy:
                raise ValueError(f"scene/task-owned fields are not valid in environment profiles: {legacy}")
        return value


class SceneConfigDocument(StrictModel):
    """Scene selection owns world composition, saved layouts, and their assets."""

    config_name: str
    component: str
    layout_set: str
    task_asset_preparers: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("config_name", "component", "layout_set")
    @classmethod
    def safe_name(cls, value: str) -> str:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
        if not value or any(character not in allowed for character in value):
            raise ValueError("must contain only letters, digits, and underscores")
        return value

    @field_validator("task_asset_preparers")
    @classmethod
    def valid_preparers(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        for task, preparers in value.items():
            if not task.strip() or not preparers or any(not preparer.strip() for preparer in preparers):
                raise ValueError("task asset preparers require non-empty task and preparer names")
        return value


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
