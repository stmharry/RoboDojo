"""Validated command and workflow contracts."""

from __future__ import annotations

from enum import StrEnum
import math
from pathlib import Path
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from robodojo.core.gpu import GpuSelector

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


class PolicyExperimentBase(StrictModel):
    """Fields shared by setup and executable policy experiment requests."""

    policy_dir: Path | None = None
    task: str | None = None
    checkpoint: str | None = None
    policy_profile: str | None = None
    policy_descriptor_hash: str | None = None
    policy_reference_match: Literal["reference_match", "domain_shift", "unspecified"] = "unspecified"
    policy_env: str | None = None
    dataset: str = "RoboDojo"
    env_config: str | None = None
    policy_contract: str | None = None
    action_type: str | None = None
    recipe: str | None = None
    contract_hash: str | None = None
    protocol: str | None = None
    episode_horizon: Annotated[int, Field(ge=1)] | None = None
    native_eval_num: Annotated[int, Field(ge=1)] | None = None
    seed: NonNegativeInt = 0
    policy_gpu: GpuSelector = "auto"

    @field_validator(
        "task",
        "checkpoint",
        "policy_profile",
        "policy_descriptor_hash",
        "policy_env",
        "dataset",
        "env_config",
        "policy_contract",
        "action_type",
        "recipe",
        "contract_hash",
        "protocol",
    )
    @classmethod
    def experiment_value_non_empty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be empty")
        return value


class PolicyExperimentRequest(PolicyExperimentBase):
    policy_dir: Path
    task: str
    checkpoint: str
    policy_profile: str = "manual"
    policy_env: str
    env_config: str
    policy_contract: str
    action_type: str = "ee"

    def policy_request(
        self,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        dry_run: bool = False,
    ) -> PolicyServerLaunchRequest:
        return PolicyServerLaunchRequest(
            policy_dir=self.policy_dir,
            task=self.task,
            checkpoint=self.checkpoint,
            policy_profile=self.policy_profile or "manual",
            policy_descriptor_hash=self.policy_descriptor_hash,
            policy_reference_match=self.policy_reference_match,
            policy_env=self.policy_env,
            dataset=self.dataset,
            env_config=self.env_config,
            policy_contract=self.policy_contract,
            action_type=self.action_type,
            seed=self.seed,
            policy_gpu=self.policy_gpu,
            host=host,
            port=port,
            dry_run=dry_run,
        )


class ExperimentRequest(PolicyExperimentRequest):
    protocol: str
    episode_horizon: Annotated[int, Field(ge=1)]
    native_eval_num: Annotated[int, Field(ge=1)]
    scene_config: str
    env_gpu: GpuSelector = "auto"

    @field_validator("scene_config")
    @classmethod
    def scene_non_empty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be empty")
        return value


class EvaluationRequest(ExperimentRequest):
    eval_num: int | Literal["native"] | None = None
    checkpoint_label: str | None = None
    export_scene: bool = False
    export_scene_only: bool = False
    export_scene_dir: Path | None = None
    layout_id: NonNegativeInt = 0
    publish: bool = False
    dry_run: bool = False

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
    scene_config: str | None = None

    @field_validator("scene_config")
    @classmethod
    def setup_scene_non_empty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be empty")
        return value

    @model_validator(mode="after")
    def required_stage_values(self) -> SetupRequest:
        selected = set(self.stages) or set(SetupStage)
        required: set[str] = set()
        if SetupStage.ASSETS in selected:
            required.update(
                {
                    "task",
                    "env_config",
                    "policy_contract",
                    "scene_config",
                    "protocol",
                    "episode_horizon",
                    "native_eval_num",
                }
            )
        if SetupStage.POLICY in selected:
            required.update(
                {
                    "policy_dir",
                    "task",
                    "checkpoint",
                    "policy_env",
                    "env_config",
                    "policy_contract",
                    "action_type",
                    "protocol",
                    "episode_horizon",
                    "native_eval_num",
                }
            )
        missing = sorted(name for name in required if not getattr(self, name))
        if missing:
            raise ValueError(f"setup stage arguments are missing: {', '.join(missing)}")
        return self

    def selected_stages(self) -> tuple[SetupStage, ...]:
        return self.stages or tuple(SetupStage)

    def policy_request(self) -> PolicyServerLaunchRequest:
        return PolicyServerLaunchRequest(
            policy_dir=self.policy_dir,
            task=self.task,
            checkpoint=self.checkpoint,
            policy_profile=self.policy_profile or "manual",
            policy_descriptor_hash=self.policy_descriptor_hash,
            policy_reference_match=self.policy_reference_match,
            policy_env=self.policy_env,
            dataset=self.dataset,
            env_config=self.env_config,
            policy_contract=self.policy_contract,
            action_type=self.action_type,
            seed=self.seed,
            policy_gpu=self.policy_gpu,
        )


class SetupStageResult(StrictModel):
    name: str
    status: Literal["READY", "CHANGED", "SKIPPED", "WARN", "FAIL"]
    detail: str
    remediation: str | None = None


class SetupReport(StrictModel):
    status: Literal["PASS", "WARN", "FAIL"]
    stages: list[SetupStageResult]


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
    protocol_name: str
    episode_horizon: Annotated[int, Field(ge=1)]
    native_eval_num: Annotated[int, Field(ge=1)]
    recipe: str | None = None
    contract_hash: str | None = None
    policy_name: str
    policy_profile: str = "manual"
    policy_descriptor_hash: str | None = None
    policy_reference_match: Literal["reference_match", "domain_shift", "unspecified"] = "unspecified"
    host: str = "127.0.0.1"
    port: Port
    env_config: str = "arx_x5"
    scene_config: str
    env_gpu: NonNegativeInt = 0
    seed: NonNegativeInt = 0
    eval_num: Annotated[int, Field(ge=1)] | Literal["native"] = 1
    additional_info: str
    protocol: Literal["ws"] = "ws"
    policy_server_url: str = ""
    dry_run: bool = False

    @field_validator("protocol_name", "scene_config", "recipe", "contract_hash")
    @classmethod
    def optional_non_empty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be empty")
        return value


class SweepRequest(StrictModel):
    recipes: tuple[str, ...]
    seed: NonNegativeInt = 0
    policy_gpu: GpuSelector = "auto"
    env_gpu: GpuSelector = "auto"
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
    env_gpu: GpuSelector = "auto"
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


class EnvironmentConfigReferences(StrictModel):
    sim: str
    robot: str
    camera: str


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
    protocol: str

    @field_validator("policy", "environment", "scene", "protocol")
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
    schema_version: Literal[2]
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


class EnvironmentDiagnostics(StrictModel):
    matched_replay_manifest: str | None = None


class EnvironmentVariantTarget(StrictModel):
    policy: str | None = None
    scene: str | None = None
    protocol: str | None = None

    @field_validator("policy", "scene", "protocol")
    @classmethod
    def safe_target_name(cls, value: str | None) -> str | None:
        if value is not None and re.fullmatch(r"[A-Za-z0-9_]+", value) is None:
            raise ValueError("variant target names must contain only letters, digits, and underscores")
        return value

    @model_validator(mode="after")
    def at_least_one_target(self) -> EnvironmentVariantTarget:
        if self.policy is None and self.scene is None and self.protocol is None:
            raise ValueError("variant target must name at least one policy, scene, or protocol")
        return self


class EnvironmentVariant(StrictModel):
    kind: Literal["reference", "policy_tuned", "diagnostic"]
    derived_for: EnvironmentVariantTarget | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def tuned_variant_has_target(self) -> EnvironmentVariant:
        if self.kind == "policy_tuned" and self.derived_for is None:
            raise ValueError("policy_tuned variants must declare derived_for metadata")
        return self


class WorkspaceFrameContract(StrictModel):
    """Embodiment roots expressed in one scene-owned support frame."""

    anchor: str
    robot_root_offsets: dict[str, tuple[float, float, float]]

    @field_validator("anchor")
    @classmethod
    def safe_anchor(cls, value: str) -> str:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is None:
            raise ValueError("workspace anchor must be one safe scene fixture name")
        return value

    @field_validator("robot_root_offsets", mode="before")
    @classmethod
    def valid_robot_root_offsets(
        cls,
        value: Any,
    ) -> dict[str, tuple[float, float, float]]:
        if not isinstance(value, dict) or not value:
            raise ValueError("workspace robot root offsets must not be empty")
        invalid = sorted(slot for slot in value if re.fullmatch(r"robot\d+", slot) is None)
        if invalid:
            raise ValueError(f"workspace robot roots require robot<N> slot names: {invalid}")
        return {
            slot: _finite_vector(offset, length=3, field=f"workspace {slot} root offset")
            for slot, offset in value.items()
        }


class EnvironmentConfigDocument(BaseModel):
    """Typed profile metadata with forward-compatible upstream payload fields."""

    model_config = ConfigDict(extra="allow")

    config_name: str
    extends: str | None = None
    selectable: bool = True
    policy_contract: str | None = None
    hardware_calibration: str | None = None
    diagnostics: EnvironmentDiagnostics | None = None
    variant: EnvironmentVariant | None = None
    asset_builds: list[str] = Field(default_factory=list)
    workspace: WorkspaceFrameContract | None = None
    config: EnvironmentConfigReferences
    observation: dict[str, Any] = Field(default_factory=dict)

    @field_validator("config_name", "extends", "policy_contract")
    @classmethod
    def safe_profile_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
        if not value or any(character not in allowed for character in value):
            raise ValueError("must contain only letters, digits, and underscores")
        return value

    @field_validator("asset_builds")
    @classmethod
    def safe_asset_builds(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("environment asset builds must be unique")
        invalid = sorted(name for name in value if re.fullmatch(r"[a-z][a-z0-9_]*", name) is None)
        if invalid:
            raise ValueError(f"invalid environment asset build names: {invalid}")
        return value

    @model_validator(mode="before")
    @classmethod
    def reject_scene_owned_fields(cls, value: Any) -> Any:
        if isinstance(value, dict):
            legacy = sorted({"layout_config_name", "task_instruction_overrides"}.intersection(value))
            if legacy:
                raise ValueError(f"scene/task-owned fields are not valid in environment profiles: {legacy}")
        return value


class SceneCatalogAsset(StrictModel):
    """One object in RoboDojo's indexed runtime asset catalog."""

    object_type: Literal["Garment"]
    category: str
    index: NonNegativeInt

    @field_validator("category")
    @classmethod
    def safe_category(cls, value: str) -> str:
        if not value or "\\" in value or Path(value).name != value or value in {".", ".."}:
            raise ValueError("must be one safe catalog path segment")
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("must not contain control characters")
        return value


class SceneGarmentVariantRecipe(StrictModel):
    """Declarative, versioned derivation of a topology-compatible garment."""

    kind: Literal["garment_mesh_variant"]
    transform: Literal["yam_short_sleeve_v1"]
    source: SceneCatalogAsset
    destination: SceneCatalogAsset

    @model_validator(mode="after")
    def distinct_catalog_entries(self) -> SceneGarmentVariantRecipe:
        if self.source == self.destination:
            raise ValueError("scene asset recipe source and destination must differ")
        return self


def _finite_vector(value: tuple[float, ...], *, length: int, field: str) -> tuple[float, ...]:
    if len(value) != length or not all(math.isfinite(component) for component in value):
        raise ValueError(f"{field} must contain {length} finite values")
    return value


class SceneRobotMount(StrictModel):
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]

    @field_validator("position")
    @classmethod
    def finite_position(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        return _finite_vector(value, length=3, field="robot mount position")

    @field_validator("orientation")
    @classmethod
    def normalized_orientation(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        value = _finite_vector(value, length=4, field="robot mount orientation")
        norm = math.sqrt(sum(component * component for component in value))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("robot mount orientation must be a normalized scalar-first quaternion")
        return value


class SceneCameraMount(StrictModel):
    kind: Literal["world", "robot_link", "scene_fixture"]
    target: str | None = None
    frame: str | None = None
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    pose_convention: Literal["isaac_usd", "sapien_robotics"] = "isaac_usd"
    optical_roll_deg: Literal[-180.0, -90.0, 0.0, 90.0, 180.0] = 0.0
    near_clip_m: float | None = None
    basis: str | None = None

    @field_validator("position")
    @classmethod
    def finite_position(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        return _finite_vector(value, length=3, field="camera mount position")

    @field_validator("orientation")
    @classmethod
    def normalized_orientation(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        value = _finite_vector(value, length=4, field="camera mount orientation")
        norm = math.sqrt(sum(component * component for component in value))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("camera mount orientation must be a normalized scalar-first quaternion")
        return value

    @field_validator("near_clip_m")
    @classmethod
    def valid_near_clip(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError("camera near clip must be finite and positive")
        return value

    @model_validator(mode="after")
    def validate_target_and_frame(self) -> SceneCameraMount:
        if self.kind == "world":
            if self.target is not None or self.frame is not None:
                raise ValueError("world camera mounts may not declare target or frame")
            return self
        if not self.target or not self.target.strip():
            raise ValueError(f"{self.kind} camera mounts require a target")
        if self.frame is not None:
            if self.kind != "scene_fixture":
                raise ValueError("camera mount frame is valid only for scene_fixture mounts")
            parts = self.frame.split("/")
            if self.frame.startswith("/") or any(
                not part or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part) for part in parts
            ):
                raise ValueError("camera mount frame must be a safe relative USD prim path")
        return self


class SceneMounts(StrictModel):
    robots: dict[str, SceneRobotMount] = Field(default_factory=dict)
    cameras: dict[str, SceneCameraMount] = Field(default_factory=dict)

    @field_validator("robots")
    @classmethod
    def valid_robot_slots(cls, value: dict[str, SceneRobotMount]) -> dict[str, SceneRobotMount]:
        invalid = sorted(slot for slot in value if re.fullmatch(r"robot\d+", slot) is None)
        if invalid:
            raise ValueError(f"scene robot mounts require robot<N> slot names: {invalid}")
        return value

    @field_validator("cameras")
    @classmethod
    def valid_camera_keys(cls, value: dict[str, SceneCameraMount]) -> dict[str, SceneCameraMount]:
        if any(not key.strip() for key in value):
            raise ValueError("scene camera mount keys must not be empty")
        return value


SceneAssetBuildName = Literal["moonlake_office", "moonlake_packing"]


class SceneConfigDocument(StrictModel):
    """Scene selection owns world composition, saved layouts, and their assets."""

    config_name: str
    component: str
    layout_set: str
    layout_source: Literal["assets", "bundled"] = "assets"
    asset_builds: list[SceneAssetBuildName] = Field(default_factory=list)
    task_asset_builds: dict[str, list[SceneAssetBuildName]] = Field(default_factory=dict)
    task_assets: dict[str, list[SceneGarmentVariantRecipe]] = Field(default_factory=dict)
    compatible_environments: list[str] = Field(default_factory=list)
    mounts: SceneMounts = Field(default_factory=SceneMounts)

    @model_validator(mode="before")
    @classmethod
    def reject_opaque_asset_preparers(cls, value: Any) -> Any:
        if isinstance(value, dict) and "task_asset_preparers" in value:
            raise ValueError(
                "task_asset_preparers is no longer supported; use task_assets with typed scene asset recipes"
            )
        return value

    @field_validator("config_name", "component", "layout_set")
    @classmethod
    def safe_name(cls, value: str) -> str:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
        if not value or any(character not in allowed for character in value):
            raise ValueError("must contain only letters, digits, and underscores")
        return value

    @field_validator("compatible_environments")
    @classmethod
    def valid_compatible_environments(cls, value: list[str]) -> list[str]:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
        invalid = sorted(name for name in value if not name or any(character not in allowed for character in name))
        if invalid:
            raise ValueError(
                f"compatible environment names must contain only letters, digits, and underscores: {invalid}"
            )
        if len(value) != len(set(value)):
            raise ValueError("compatible environment names must be unique")
        return value

    @field_validator("asset_builds")
    @classmethod
    def unique_asset_builds(cls, value: list[SceneAssetBuildName]) -> list[SceneAssetBuildName]:
        if len(value) != len(set(value)):
            raise ValueError("scene asset builds must be unique")
        return value

    @field_validator("task_asset_builds")
    @classmethod
    def valid_task_asset_builds(
        cls,
        value: dict[str, list[SceneAssetBuildName]],
    ) -> dict[str, list[SceneAssetBuildName]]:
        for task, builds in value.items():
            if not task.strip() or not builds:
                raise ValueError("task asset builds require non-empty task names and build lists")
            if len(builds) != len(set(builds)):
                raise ValueError(f"task asset builds for {task} must be unique")
        return value

    @field_validator("task_assets")
    @classmethod
    def valid_task_assets(
        cls,
        value: dict[str, list[SceneGarmentVariantRecipe]],
    ) -> dict[str, list[SceneGarmentVariantRecipe]]:
        for task, recipes in value.items():
            if not task.strip() or not recipes:
                raise ValueError("task assets require non-empty task names and recipe lists")
            destinations = [recipe.destination for recipe in recipes]
            if len({destination.model_dump_json() for destination in destinations}) != len(destinations):
                raise ValueError(f"task assets for {task} contain duplicate destinations")
        return value


class SmokeRecord(StrictModel):
    status: Literal["PASS", "FAIL", "SKIP", "DRY_RUN"]
    recipe: str
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


class SnapshotRecord(StrictModel):
    status: Literal["PENDING", "PASS", "FAIL", "SKIP", "DRY_RUN"]
    recipe: str
    policy: str
    environment: str
    scene: str
    protocol: str
    task: str
    contract_hash: str
    exit_code: int | None = None
    elapsed_sec: float = 0.0
    output_dir: str
    message: str = ""


class SnapshotSummary(StrictModel):
    format_version: Literal[1] = 1
    run_id: str
    output_dir: str
    seed: NonNegativeInt
    layout_id: NonNegativeInt
    export_scene: bool
    recipes: tuple[str, ...]
    complete: bool = False
    results: list[SnapshotRecord]
