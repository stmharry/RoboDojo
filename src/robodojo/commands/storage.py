"""Durable storage command group."""

from __future__ import annotations

from pathlib import Path

import typer

storage_app = typer.Typer(no_args_is_help=True, help="Manage durable RoboDojo storage.")


@storage_app.command("status")
@storage_app.command("doctor")
def storage_doctor() -> None:
    """Check local storage writes and optional S3 access."""
    from robodojo.workflows.storage import doctor

    doctor()


@storage_app.command()
def publish(
    source: Path = typer.Argument(..., help="Local directory whose files should be published."),
    relative: str = typer.Argument(
        ...,
        help="Destination below ROBODOJO_S3_URI, beginning with assets, datasets, model_weights, or runs.",
    ),
    replace: bool = typer.Option(
        False,
        "--replace",
        help="Replace an already completed remote payload instead of preserving its immutability.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the resolved source and S3 destination without uploading files.",
    ),
) -> None:
    """Publish a local directory to an explicit canonical S3 destination."""
    from robodojo.workflows.storage import publish as publish_payload

    publish_payload(source, relative, replace=replace, dry_run=dry_run)


def _storage_passthrough(arguments: list[str]) -> None:
    from robodojo.workflows.storage import main

    raise typer.Exit(main(arguments))


def _publish_arguments(command: str, values: list[str], replace: bool, dry_run: bool) -> list[str]:
    arguments = [command, *values]
    if replace:
        arguments.append("--replace")
    if dry_run:
        arguments.append("--dry-run")
    return arguments


@storage_app.command()
def pull(
    relative: str = typer.Argument(
        ...,
        help="Completed payload below ROBODOJO_S3_URI to restore into canonical local storage.",
    ),
    replace: bool = typer.Option(
        False,
        "--replace",
        help="Replace an existing local payload after the remote copy passes verification.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the resolved S3 source and local destination without downloading files.",
    ),
) -> None:
    """Download and verify one completed S3 payload."""
    arguments = ["pull", relative]
    if replace:
        arguments.append("--replace")
    if dry_run:
        arguments.append("--dry-run")
    _storage_passthrough(arguments)


@storage_app.command("publish-assets")
def publish_assets(
    source: Path = typer.Argument(..., help="Local benchmark asset directory to publish."),
    replace: bool = typer.Option(False, "--replace", help="Replace the completed remote assets payload."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the destination without uploading files."),
) -> None:
    """Publish the canonical benchmark asset payload."""
    _storage_passthrough(_publish_arguments("publish-assets", [str(source)], replace, dry_run))


@storage_app.command("publish-data")
def publish_data(
    dataset: str = typer.Argument(..., help="Dataset name below the remote datasets prefix."),
    source: Path = typer.Argument(..., help="Local dataset directory to publish."),
    replace: bool = typer.Option(False, "--replace", help="Replace the completed remote dataset payload."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the destination without uploading files."),
) -> None:
    """Publish one named dataset payload."""
    _storage_passthrough(_publish_arguments("publish-data", [dataset, str(source)], replace, dry_run))


@storage_app.command("publish-checkpoint")
def publish_checkpoint(
    policy: str = typer.Argument(..., help="Policy name below the remote model_weights prefix."),
    checkpoint: str = typer.Argument(..., help="Checkpoint name used as the payload directory."),
    source: Path = typer.Argument(..., help="Local checkpoint directory to publish."),
    replace: bool = typer.Option(False, "--replace", help="Replace the completed remote checkpoint payload."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the destination without uploading files."),
) -> None:
    """Publish one policy checkpoint payload."""
    _storage_passthrough(_publish_arguments("publish-checkpoint", [policy, checkpoint, str(source)], replace, dry_run))


@storage_app.command("publish-model")
def publish_model(
    policy: str = typer.Argument(..., help="Policy name below the remote model_weights prefix."),
    model: str = typer.Argument(..., help="Model name used as the payload directory."),
    source: Path = typer.Argument(..., help="Local model directory to publish."),
    replace: bool = typer.Option(False, "--replace", help="Replace the completed remote model payload."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the destination without uploading files."),
) -> None:
    """Publish one named policy model payload."""
    _storage_passthrough(_publish_arguments("publish-model", [policy, model, str(source)], replace, dry_run))


@storage_app.command("publish-reference-cache")
def publish_reference_cache(
    name: str = typer.Argument(..., help="Reference-cache name below the remote datasets prefix."),
    revision: str = typer.Argument(..., help="Source revision used to version the cache payload."),
    source: Path = typer.Argument(..., help="Local reference-cache directory to publish."),
    replace: bool = typer.Option(False, "--replace", help="Replace the completed remote cache payload."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the destination without uploading files."),
) -> None:
    """Publish a versioned reference-data cache."""
    _storage_passthrough(_publish_arguments("publish-reference-cache", [name, revision, str(source)], replace, dry_run))


@storage_app.command("publish-eval")
def publish_eval(
    source: Path = typer.Option(
        Path("."),
        "--source",
        help="Completed evaluation directory, used directly when --run-id is omitted.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Timestamped run name to find uniquely below canonical local evaluation storage.",
    ),
    replace: bool = typer.Option(False, "--replace", help="Replace the completed remote evaluation payload."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the destination without uploading files."),
) -> None:
    """Publish one completed evaluation result directory."""
    values = [str(source)]
    if run_id:
        values += ["--run-id", run_id]
    _storage_passthrough(_publish_arguments("publish-eval", values, replace, dry_run))


@storage_app.command("publish-run")
def publish_run(
    kind: str = typer.Argument(..., help="Run category below the remote runs prefix."),
    run_id: str = typer.Argument(..., help="Stable run identifier used as the payload directory."),
    source: Path = typer.Argument(..., help="Local run directory to publish."),
    replace: bool = typer.Option(False, "--replace", help="Replace the completed remote run payload."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the destination without uploading files."),
) -> None:
    """Publish a named run payload under a caller-selected category."""
    _storage_passthrough(_publish_arguments("publish-run", [kind, run_id, str(source)], replace, dry_run))
