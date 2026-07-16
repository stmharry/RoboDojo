"""Scene composition, asset, and mount models."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from robodojo.core.models.common import NonNegativeInt, StrictModel, finite_vector

SceneAssetBuildName = Literal["moonlake_office", "moonlake_packing", "piper_pickplace"]


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


class SceneRobotMount(StrictModel):
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]

    @field_validator("position")
    @classmethod
    def finite_position(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        return finite_vector(value, length=3, field="robot mount position")

    @field_validator("orientation")
    @classmethod
    def normalized_orientation(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        value = finite_vector(value, length=4, field="robot mount orientation")
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
        return finite_vector(value, length=3, field="camera mount position")

    @field_validator("orientation")
    @classmethod
    def normalized_orientation(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        value = finite_vector(value, length=4, field="camera mount orientation")
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
