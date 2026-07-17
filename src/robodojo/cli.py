"""Unified Typer command-line interface for RoboDojo."""

from __future__ import annotations

import os

import typer

from robodojo.commands.assets import assets_app, data_app
from robodojo.commands.docker import docker_app
from robodojo.commands.inventory import recipes, tasks
from robodojo.commands.results import results_app
from robodojo.commands.runtime import adapter_client, client, doctor, evaluate, preflight, server
from robodojo.commands.setup import setup
from robodojo.commands.snapshots import snapshots
from robodojo.commands.storage import storage_app
from robodojo.commands.sweeps import benchmark, smoke
from robodojo.core.logging import LOG_LEVEL_ENV, configure_logging, parse_log_level

app = typer.Typer(no_args_is_help=True, help="RoboDojo evaluation and operations CLI.")
eval_app = typer.Typer(no_args_is_help=True, help="Run and validate RoboDojo evaluations.")
catalog_app = typer.Typer(no_args_is_help=True, help="Inspect RoboDojo tasks and evaluation recipes.")
workspace_app = typer.Typer(no_args_is_help=True, help="Prepare and operate the RoboDojo workspace.")

eval_app.command(name="run")(evaluate)
eval_app.command()(preflight)
eval_app.command()(server)
eval_app.command()(client)
eval_app.command()(smoke)
eval_app.command()(benchmark)
eval_app.command()(snapshots)

catalog_app.command()(tasks)
catalog_app.command()(recipes)

workspace_app.command()(setup)
workspace_app.command()(doctor)
workspace_app.add_typer(assets_app, name="assets")
workspace_app.add_typer(data_app, name="data")
workspace_app.add_typer(storage_app, name="storage")
workspace_app.add_typer(docker_app, name="docker")

app.add_typer(eval_app, name="eval")
app.add_typer(catalog_app, name="catalog")
app.add_typer(workspace_app, name="workspace")
app.add_typer(results_app, name="results")
app.command(
    "_adapter-client",
    hidden=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)(adapter_client)


@app.callback()
def configure_cli_logging(
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        help="RoboDojo diagnostic level: DEBUG, INFO, WARNING, ERROR, or CRITICAL.",
    ),
) -> None:
    """Configure RoboDojo diagnostics before dispatching a command."""
    try:
        parse_log_level(log_level)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--log-level") from exc
    if log_level is not None:
        os.environ[LOG_LEVEL_ENV] = log_level.strip().upper()
    configure_logging(log_level)


if __name__ == "__main__":
    app()
