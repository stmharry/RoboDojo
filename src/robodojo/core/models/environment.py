"""Environment profile and workspace models."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from robodojo.core.models.common import StrictModel, finite_vector


class EnvironmentConfigReferences(StrictModel):
    sim: str
    robot: str
    camera: str


class EnvironmentDiagnostics(StrictModel):
    matched_replay_manifest: str | None = None


class EnvironmentVariantTarget(StrictModel):
    policy: str | None = None
    scene: str | None = None
    task_protocol: str | None = None

    @field_validator("policy", "scene", "task_protocol")
    @classmethod
    def safe_target_name(cls, value: str | None) -> str | None:
        if value is not None and re.fullmatch(r"[A-Za-z0-9_]+", value) is None:
            raise ValueError("variant target names must contain only letters, digits, and underscores")
        return value

    @model_validator(mode="after")
    def at_least_one_target(self) -> EnvironmentVariantTarget:
        if self.policy is None and self.scene is None and self.task_protocol is None:
            raise ValueError("variant target must name at least one policy, scene, or task protocol")
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
            slot: finite_vector(offset, length=3, field=f"workspace {slot} root offset")
            for slot, offset in value.items()
        }


class EnvironmentConfigDocument(BaseModel):
    """Typed profile metadata with forward-compatible upstream payload fields."""

    model_config = ConfigDict(extra="allow")

    config_name: str
    extends: str | None = None
    selectable: bool = True
    embodiment: str | None = None
    hardware_calibration: str | None = None
    diagnostics: EnvironmentDiagnostics | None = None
    variant: EnvironmentVariant | None = None
    asset_builds: list[str] = Field(default_factory=list)
    workspace: WorkspaceFrameContract | None = None
    config: EnvironmentConfigReferences
    observation: dict[str, Any] = Field(default_factory=dict)

    @field_validator("config_name", "extends", "embodiment")
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
