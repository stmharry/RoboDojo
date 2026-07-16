"""Sequential smoke and benchmark command adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from robodojo.commands.common import model, paths
from robodojo.commands.options import (
    DryRunOption,
    EnvironmentGpuOption,
    EvaluationCountOption,
    PolicyGpuOption,
    RepositoryRootOption,
    SeedOption,
    parse_evaluation_count,
)
from robodojo.core.models import SweepRequest

RecipesOption = Annotated[
    list[str],
    typer.Option("--recipe", help="Recipe to sweep; repeat for multiple recipes."),
]
LimitOption = Annotated[
    int | None,
    typer.Option("--limit", min=1, help="Run at most this many selected tasks after filtering."),
]
ResumeOption = Annotated[
    bool,
    typer.Option(
        "--resume",
        help="Reuse the named sweep summary and skip task/scene pairs that already passed.",
    ),
]
FailFastOption = Annotated[
    bool,
    typer.Option("--fail-fast", help="Stop the sweep immediately after the first failed evaluation."),
]
RunIdOption = Annotated[
    str | None,
    typer.Option(
        "--run-id",
        help="Stable sweep identifier used for summary paths and required to resume a specific run.",
    ),
]


def _run_sweep(
    *,
    recipe: list[str],
    seed: int,
    policy_gpu: str,
    env_gpu: str,
    eval_num: str,
    limit: int | None,
    resume: bool,
    fail_fast: bool,
    run_id: str | None,
    dry_run: bool,
    root: Path | None,
) -> None:
    from robodojo.workflows.sweeps import run_sweep

    request = model(
        SweepRequest,
        recipes=tuple(recipe),
        seed=seed,
        policy_gpu=policy_gpu,
        env_gpu=env_gpu,
        eval_num=parse_evaluation_count(eval_num),
        limit=limit,
        resume=resume,
        fail_fast=fail_fast,
        run_id=run_id,
        dry_run=dry_run,
    )
    raise typer.Exit(run_sweep(paths(root), request))


def smoke(
    recipe: RecipesOption,
    seed: SeedOption = 0,
    policy_gpu: PolicyGpuOption = "auto",
    env_gpu: EnvironmentGpuOption = "auto",
    eval_num: EvaluationCountOption = "1",
    limit: LimitOption = None,
    resume: ResumeOption = False,
    fail_fast: FailFastOption = False,
    run_id: RunIdOption = None,
    dry_run: DryRunOption = False,
    root: RepositoryRootOption = None,
) -> None:
    """Run a one-episode sequential recipe sweep."""
    _run_sweep(**locals())


def benchmark(
    recipe: RecipesOption,
    eval_num: EvaluationCountOption,
    seed: SeedOption = 0,
    policy_gpu: PolicyGpuOption = "auto",
    env_gpu: EnvironmentGpuOption = "auto",
    limit: LimitOption = None,
    resume: ResumeOption = False,
    fail_fast: FailFastOption = False,
    run_id: RunIdOption = None,
    dry_run: DryRunOption = False,
    root: RepositoryRootOption = None,
) -> None:
    """Run a sequential benchmark sweep."""
    _run_sweep(**locals())
