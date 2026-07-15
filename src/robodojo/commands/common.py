"""Shared CLI-only validation and repository resolution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def contract_values(
    repository: RepositoryPaths,
    *,
    recipe: str | None,
    policy_profile: str | None,
    environment: str | None,
    scene: str | None,
    protocol: str | None,
) -> dict[str, Any]:
    """Resolve one strict recipe or complete manual component selection."""

    from robodojo.core.contracts import resolve_selection

    try:
        contract = resolve_selection(
            repository,
            recipe=recipe,
            policy=policy_profile,
            environment=environment,
            scene=scene,
            protocol=protocol,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    return contract.request_values(repository)


def report_format(value: str) -> str:
    if value not in {"human", "json"}:
        raise typer.BadParameter("expected human or json", param_hint="--format")
    return value
