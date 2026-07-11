"""Typed environment and .env settings."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from robodojo.core.paths import RepositoryPaths


class RuntimeSettings(BaseSettings):
    """Runtime settings with process environment over checkout .env precedence."""

    model_config = SettingsConfigDict(extra="ignore", env_ignore_empty=True)

    storage_root: Path | None = Field(None, validation_alias="ROBODOJO_STORAGE_ROOT")
    s3_uri: str | None = Field(None, validation_alias="ROBODOJO_S3_URI")
    local_scratch_root: Path | None = Field(None, validation_alias="ROBODOJO_LOCAL_SCRATCH_ROOT")
    assets_root: Path | None = Field(None, validation_alias="ROBODOJO_ASSETS_ROOT")
    data_root: Path | None = Field(
        None,
        validation_alias=AliasChoices("ROBODOJO_DATA_ROOT", "ROBO_DOJO_DATA_ROOT"),
    )
    model_root: Path | None = Field(None, validation_alias="ROBODOJO_MODEL_ROOT")
    checkpoint_root: Path | None = Field(None, validation_alias="ROBODOJO_CHECKPOINT_ROOT")
    eval_root: Path | None = Field(None, validation_alias="ROBODOJO_EVAL_ROOT")
    eval_work_root: Path | None = Field(None, validation_alias="ROBODOJO_EVAL_WORK_ROOT")
    run_root: Path | None = Field(None, validation_alias="ROBODOJO_RUN_ROOT")
    run_work_root: Path | None = Field(None, validation_alias="ROBODOJO_RUN_WORK_ROOT")
    summary_path: Path | None = Field(None, validation_alias="ROBODOJO_SUMMARY_PATH")
    aws_profile: str | None = Field(None, validation_alias="AWS_PROFILE")

    @classmethod
    def load(cls, paths: RepositoryPaths) -> RuntimeSettings:
        return cls(_env_file=paths.root / ".env", _env_file_encoding="utf-8")

    def export_missing(self) -> None:
        aliases = {
            "storage_root": "ROBODOJO_STORAGE_ROOT",
            "s3_uri": "ROBODOJO_S3_URI",
            "local_scratch_root": "ROBODOJO_LOCAL_SCRATCH_ROOT",
            "assets_root": "ROBODOJO_ASSETS_ROOT",
            "data_root": "ROBODOJO_DATA_ROOT",
            "model_root": "ROBODOJO_MODEL_ROOT",
            "checkpoint_root": "ROBODOJO_CHECKPOINT_ROOT",
            "eval_root": "ROBODOJO_EVAL_ROOT",
            "eval_work_root": "ROBODOJO_EVAL_WORK_ROOT",
            "run_root": "ROBODOJO_RUN_ROOT",
            "run_work_root": "ROBODOJO_RUN_WORK_ROOT",
            "summary_path": "ROBODOJO_SUMMARY_PATH",
            "aws_profile": "AWS_PROFILE",
        }
        for field, alias in aliases.items():
            value = getattr(self, field)
            if value is not None:
                os.environ.setdefault(alias, str(value))
