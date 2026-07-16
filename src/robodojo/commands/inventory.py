"""Task and recipe inventory command adapters."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from robodojo.commands.common import paths
from robodojo.commands.options import RecipeFormat, RepositoryRootOption, TaskFormat


def tasks(
    output_format: Annotated[TaskFormat, typer.Option("--format", help="Output format.")] = TaskFormat.PLAIN,
    only_runnable: bool = typer.Option(
        False,
        "--only-runnable",
        help="In plain output, omit tasks whose code or configuration is incomplete.",
    ),
    check: bool = typer.Option(
        False,
        "--check",
        help="Exit nonzero when any canonical task is not runnable.",
    ),
) -> None:
    """List and validate canonical task implementations and configurations."""
    from robodojo.workflows.task_inventory import build_inventory, print_markdown, print_plain

    inventory = build_inventory()
    if output_format == TaskFormat.JSON:
        typer.echo(json.dumps(inventory, indent=2, sort_keys=True))
    elif output_format == TaskFormat.MARKDOWN:
        print_markdown(inventory)
    else:
        print_plain(inventory, only_runnable)
    if check and any(not record["runnable"] for record in inventory["tasks"]):
        raise typer.Exit(1)


def recipes(
    output_format: Annotated[RecipeFormat, typer.Option("--format", help="Output format.")] = RecipeFormat.PLAIN,
    check: bool = typer.Option(False, "--check", help="Validate every policy/protocol/recipe reference."),
    root: RepositoryRootOption = None,
) -> None:
    """List and validate explicit evaluation recipes."""
    from robodojo.core.contracts import recipe_rows

    repository = paths(root)
    try:
        rows = recipe_rows(repository)
    except (OSError, RuntimeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1 if check else 2) from exc
    if output_format == RecipeFormat.JSON:
        typer.echo(json.dumps(rows, indent=2, sort_keys=True))
    elif output_format == RecipeFormat.PLAIN:
        for row in rows:
            typer.echo(
                f"{row['recipe']}\t{row['policy']}\t{row['environment']}\t"
                f"{row['scene']}\t{row['protocol']}\t{row['task']}"
            )
    else:
        from robodojo.workflows.recipe_inventory import print_recipe_table

        print_recipe_table(rows)
