"""Durable storage command group."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from robodojo.commands.common import workflow_error
from robodojo.commands.options import DryRunOption, ReplaceOption
from robodojo.workflows.errors import StorageError

storage_app = typer.Typer(no_args_is_help=True, help="Manage durable RoboDojo storage.")


def _run(operation: Callable[..., None], *args: Any, **kwargs: Any) -> None:
    try:
        operation(*args, **kwargs)
    except StorageError as exc:
        workflow_error(exc)


@storage_app.command("doctor")
def storage_doctor() -> None:
    """Check local storage writes and optional S3 access."""
    from robodojo.workflows.storage import doctor

    _run(doctor)


@storage_app.command()
def publish(
    source: Path = typer.Argument(..., help="Local directory whose files should be published."),
    relative: str = typer.Argument(
        ...,
        help="Destination below ROBODOJO_S3_URI, beginning with assets, datasets, model_weights, or runs.",
    ),
    replace: ReplaceOption = False,
    dry_run: DryRunOption = False,
) -> None:
    """Publish a local directory to an explicit canonical S3 destination."""
    from robodojo.workflows.storage import publish as publish_payload

    _run(publish_payload, source, relative, replace=replace, dry_run=dry_run)


@storage_app.command()
def pull(
    relative: str = typer.Argument(
        ...,
        help="Completed payload below ROBODOJO_S3_URI to restore into canonical local storage.",
    ),
    replace: ReplaceOption = False,
    dry_run: DryRunOption = False,
) -> None:
    """Download and verify one completed S3 payload."""
    from robodojo.workflows.storage import pull as pull_payload

    _run(pull_payload, relative, replace=replace, dry_run=dry_run)


@storage_app.command("publish-eval")
def publish_eval(
    source: Path | None = typer.Option(
        None,
        "--source",
        help="Completed evaluation directory; the current directory is used when no selector is supplied.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Timestamped run name to find uniquely below canonical local evaluation storage.",
    ),
    replace: ReplaceOption = False,
    dry_run: DryRunOption = False,
) -> None:
    """Publish one completed evaluation result directory."""
    from robodojo.workflows.storage import publish_evaluation

    _run(publish_evaluation, source, run_id=run_id, replace=replace, dry_run=dry_run)
