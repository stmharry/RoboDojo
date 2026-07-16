"""Shared CLI-only validation and repository resolution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn

from pydantic import ValidationError
import typer

from robodojo.core.paths import RepositoryPaths


def paths(root: Path | None = None) -> RepositoryPaths:
    try:
        resolved = RepositoryPaths.resolve(root)
        from robodojo.core.settings import RuntimeSettings

        RuntimeSettings.load(resolved).export_missing()
        return resolved
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


def model(model_type: type[Any], **values: Any) -> Any:
    try:
        return model_type.model_validate(values)
    except ValidationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


def experiment_spec(
    repository: RepositoryPaths,
    *,
    recipe: str | None,
    policy_profile: str | None,
    environment: str | None,
    scene: str | None,
    task_protocol: str | None,
):
    """Resolve one recipe or complete manual selection into an immutable aggregate."""

    from robodojo.core.experiments.selection import resolve_selection

    try:
        experiment = resolve_selection(
            repository,
            recipe=recipe,
            policy=policy_profile,
            environment=environment,
            scene=scene,
            task_protocol=task_protocol,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    return experiment.spec(repository)


def workflow_error(exc: Exception, *, code: int = 1) -> NoReturn:
    """Render an expected workflow failure at the CLI boundary."""

    typer.echo(str(exc), err=True)
    raise typer.Exit(code) from exc
