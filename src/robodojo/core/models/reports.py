"""Typed setup, validation, sweep, and snapshot reports."""

from __future__ import annotations

from typing import Literal

from robodojo.core.models.common import NonNegativeInt, StrictModel


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


class SmokeRecord(StrictModel):
    status: Literal["PASS", "FAIL", "SKIP", "DRY_RUN"]
    recipe: str
    scene: str | None = None
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
    task_protocol: str
    task: str
    experiment_hash: str
    exit_code: int | None = None
    elapsed_sec: float = 0.0
    output_dir: str
    message: str = ""


class SnapshotSummary(StrictModel):
    format_version: Literal[2] = 2
    run_id: str
    output_dir: str
    seed: NonNegativeInt
    layout_id: NonNegativeInt
    export_scene: bool
    recipes: tuple[str, ...]
    complete: bool = False
    results: list[SnapshotRecord]
