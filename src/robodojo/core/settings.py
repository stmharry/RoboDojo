"""Typed process-environment settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from robodojo.core.paths import RepositoryPaths


class RuntimeSettings(BaseSettings):
    """Runtime settings sourced exclusively from the process environment."""

    model_config = SettingsConfigDict(extra="ignore", env_ignore_empty=True)

    storage_root: Path | None = Field(None, validation_alias="ROBODOJO_STORAGE_ROOT")
    s3_uri: str | None = Field(None, validation_alias="ROBODOJO_S3_URI")
    aws_profile: str | None = Field(None, validation_alias="AWS_PROFILE")

    REMOVED_STORAGE_VARIABLES: ClassVar[set[str]] = {
        "ROBODOJO_LOCAL_SCRATCH_ROOT",
        "ROBODOJO_ASSETS_ROOT",
        "ROBODOJO_DATA_ROOT",
        "ROBO_DOJO_DATA_ROOT",
        "ROBODOJO_MODEL_ROOT",
        "ROBODOJO_CHECKPOINT_ROOT",
        "ROBODOJO_EVAL_ROOT",
        "ROBODOJO_EVAL_WORK_ROOT",
        "ROBODOJO_RUN_ROOT",
        "ROBODOJO_RUN_WORK_ROOT",
        "ROBODOJO_SUMMARY_PATH",
    }

    @classmethod
    def load(cls, _paths: RepositoryPaths) -> RuntimeSettings:
        removed = sorted(name for name in cls.REMOVED_STORAGE_VARIABLES if os.environ.get(name, "").strip())
        if removed:
            names = ", ".join(removed)
            raise RuntimeError(f"removed storage variable(s) configured: {names}; use ROBODOJO_STORAGE_ROOT")
        return cls()

    def export_missing(self) -> None:
        aliases = {
            "storage_root": "ROBODOJO_STORAGE_ROOT",
            "s3_uri": "ROBODOJO_S3_URI",
            "aws_profile": "AWS_PROFILE",
        }
        for field, alias in aliases.items():
            value = getattr(self, field)
            if value is not None:
                os.environ.setdefault(alias, str(value))
