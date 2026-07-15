"""Result-analysis CLI command group."""

from __future__ import annotations

from pathlib import Path

import typer

results_app = typer.Typer(no_args_is_help=True, help="Analyze evaluation results.")


@results_app.command("summarize")
def summarize(
    output: Path | None = typer.Option(None, "--output", help="Markdown destination; defaults to local storage."),
    env_config: str | None = typer.Option(None, "--env-cfg", help="Include only this environment profile."),
    scene_config: str | None = typer.Option(None, "--scene", help="Include only this scene profile."),
) -> None:
    """Aggregate evaluation results into Markdown."""
    from robodojo.workflows.results_summary import main

    args: list[str] = []
    if output:
        args += ["--output", str(output)]
    if env_config:
        args += ["--env-cfg", env_config]
    if scene_config:
        args += ["--scene", scene_config]
    main(args)


@results_app.command("stats")
def results_stats(
    root: Path | None = typer.Option(None, "--root", help="Evaluation-result directory; defaults to local storage."),
    policies: list[str] | None = typer.Option(None, "--policy", help="Policy to include; repeat to compare."),
    tasks: list[str] | None = typer.Option(None, "--task", help="Task to include; repeat to select multiple."),
    env_config: str | None = typer.Option(None, "--env-cfg", help="Include only this environment profile."),
    scene_config: str | None = typer.Option(None, "--scene", help="Include only this scene profile."),
    per_seed: bool = typer.Option(False, "--per-seed", help="Break score counts down by evaluation seed."),
    json_out: Path | None = typer.Option(None, "--json-out", help="Write the complete distribution as JSON."),
) -> None:
    """Count evaluation scores by policy and task."""
    from robodojo.workflows.results_stats import main

    args: list[str] = []
    if root:
        args += ["--root", str(root)]
    if policies:
        args += ["--policies", *policies]
    for task in tasks or []:
        args += ["--task", task]
    if env_config:
        args += ["--env-cfg", env_config]
    if scene_config:
        args += ["--scene", scene_config]
    if per_seed:
        args.append("--per-seed")
    if json_out:
        args += ["--json-out", str(json_out)]
    raise typer.Exit(main(args))
