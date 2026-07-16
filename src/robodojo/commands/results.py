"""Result-analysis CLI command group."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from robodojo.commands.common import workflow_error
from robodojo.commands.options import ResultEnvironmentOption, ResultSceneOption, ResultsRootOption
from robodojo.core.scene_identity import ArtifactSchemaError
from robodojo.workflows.errors import ResultsError

results_app = typer.Typer(no_args_is_help=True, help="Analyze evaluation results.")


def _run(operation: Callable[..., Any], **kwargs: Any) -> Any:
    try:
        return operation(**kwargs)
    except (ArtifactSchemaError, ResultsError) as exc:
        workflow_error(exc)


@results_app.command("summarize")
def summarize(
    results_root: ResultsRootOption = None,
    output: Path | None = typer.Option(None, "--output", help="Markdown destination; defaults to local storage."),
    environment: ResultEnvironmentOption = None,
    scene: ResultSceneOption = None,
) -> None:
    """Aggregate evaluation results into Markdown."""
    from robodojo.workflows.results_summary import summarize_results

    _run(
        summarize_results,
        results_root=results_root,
        output=output,
        environment=environment,
        scene=scene,
    )


@results_app.command("stats")
def results_stats(
    results_root: ResultsRootOption = None,
    policies: list[str] | None = typer.Option(None, "--policy", help="Policy to include; repeat to compare."),
    tasks: list[str] | None = typer.Option(None, "--task", help="Task to include; repeat to select multiple."),
    environment: ResultEnvironmentOption = None,
    scene: ResultSceneOption = None,
    per_seed: bool = typer.Option(False, "--per-seed", help="Break score counts down by evaluation seed."),
    json_out: Path | None = typer.Option(None, "--json-out", help="Write the complete distribution as JSON."),
) -> None:
    """Count evaluation scores by policy and task."""
    from robodojo.workflows.results_stats import generate_score_report

    _run(
        generate_score_report,
        results_root=results_root,
        policies=policies,
        tasks=tasks,
        environment=environment,
        scene=scene,
        per_seed=per_seed,
        json_out=json_out,
    )
