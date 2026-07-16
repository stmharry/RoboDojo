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

app.command()(setup)
app.command()(doctor)
app.command()(tasks)
app.command()(recipes)
app.command()(snapshots)
app.command()(preflight)
app.command(name="eval")(evaluate)
app.command()(server)
app.command()(client)
app.command()(smoke)
app.command()(benchmark)
app.command(
    "_adapter-client",
    hidden=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)(adapter_client)

app.add_typer(assets_app, name="assets")
app.add_typer(data_app, name="data")
app.add_typer(storage_app, name="storage")
app.add_typer(results_app, name="results")
app.add_typer(docker_app, name="docker")


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
