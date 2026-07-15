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


def report_format(value: str) -> str:
    if value not in {"human", "json"}:
        raise typer.BadParameter("expected human or json", param_hint="--format")
    return value
