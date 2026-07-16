"""Reusable Typer option declarations and CLI-only value parsing."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

import typer


class ReportFormat(StrEnum):
    HUMAN = "human"
    JSON = "json"


class TaskFormat(StrEnum):
    PLAIN = "plain"
    JSON = "json"
    MARKDOWN = "markdown"


class RecipeFormat(StrEnum):
    TABLE = "table"
    PLAIN = "plain"
    JSON = "json"


RepositoryRootOption: TypeAlias = Annotated[
    Path | None,
    typer.Option(
        "--root",
        help="Repository checkout to use; auto-detected when omitted.",
    ),
]
RecipeOption: TypeAlias = Annotated[
    str | None,
    typer.Option("--recipe", help="Tracked evaluation recipe."),
]
PolicyProfileOption: TypeAlias = Annotated[
    str | None,
    typer.Option("--policy-profile", help="Manual policy profile."),
]
EnvironmentOption: TypeAlias = Annotated[
    str | None,
    typer.Option("--environment", help="Manual environment profile."),
]
SceneOption: TypeAlias = Annotated[
    str | None,
    typer.Option("--scene", help="Manual scene profile."),
]
TaskProtocolOption: TypeAlias = Annotated[
    str | None,
    typer.Option("--task-protocol", help="Manual task protocol."),
]
SeedOption: TypeAlias = Annotated[
    int,
    typer.Option("--seed", min=0, help="Nonnegative experiment seed."),
]
PolicyGpuOption: TypeAlias = Annotated[
    str,
    typer.Option(
        "--policy-gpu",
        envvar="POLICY_GPU",
        help="Policy GPU as a zero-based index or auto; POLICY_GPU is used when the flag is omitted.",
    ),
]
EnvironmentGpuOption: TypeAlias = Annotated[
    str,
    typer.Option(
        "--env-gpu",
        envvar="ENV_GPU",
        help="Simulator GPU as a zero-based index or auto; ENV_GPU is used when the flag is omitted.",
    ),
]
EvaluationCountOption: TypeAlias = Annotated[
    str,
    typer.Option(
        "--eval-num",
        help="Episode count as a positive integer, or native to keep the simulator config value.",
    ),
]
CheckpointLabelOption: TypeAlias = Annotated[
    str | None,
    typer.Option(
        "--checkpoint-label",
        help="Filesystem-safe result label; defaults to the checkpoint name or path basename.",
    ),
]
DryRunOption: TypeAlias = Annotated[
    bool,
    typer.Option("--dry-run", help="Resolve and print planned work without performing it."),
]
ReportFormatOption: TypeAlias = Annotated[
    ReportFormat,
    typer.Option("--format", help="Report format."),
]
ResultsRootOption: TypeAlias = Annotated[
    Path | None,
    typer.Option(
        "--results-root",
        help="Evaluation-result directory; defaults to canonical local storage.",
    ),
]
ResultEnvironmentOption: TypeAlias = Annotated[
    str | None,
    typer.Option("--environment", help="Include only this environment profile."),
]
ResultSceneOption: TypeAlias = Annotated[
    str | None,
    typer.Option("--scene", help="Include only this scene profile."),
]
ReplaceOption: TypeAlias = Annotated[
    bool,
    typer.Option("--replace", help="Replace an existing completed payload."),
]
RevisionOption: TypeAlias = Annotated[
    str,
    typer.Option("--revision", help="Git revision to download."),
]
DockerImageOption: TypeAlias = Annotated[
    str,
    typer.Option("--image", help="Repository and tag for the simulator image."),
]


def parse_evaluation_count(value: str | None) -> int | Literal["native"] | None:
    """Parse the shared episode-count syntax used by evaluation commands."""

    if value is None or value == "native":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise typer.BadParameter("expected a positive integer or native", param_hint="--eval-num") from exc
    if parsed < 1:
        raise typer.BadParameter("expected a positive integer or native", param_hint="--eval-num")
    return parsed
