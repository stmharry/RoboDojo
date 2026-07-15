"""Asset and dataset CLI command groups."""

from __future__ import annotations

from pathlib import Path

import typer

from robodojo.commands.common import paths
from robodojo.core.models import DataFormat

assets_app = typer.Typer(no_args_is_help=True, help="Download and build benchmark assets.")
data_app = typer.Typer(no_args_is_help=True, help="Download benchmark datasets.")


@assets_app.command("download")
def assets_download(
    root: Path | None = typer.Option(None, "--root", help="Repository checkout used to resolve settings."),
    revision: str = typer.Option("main", "--revision", help="Git revision of the asset dataset to download."),
) -> None:
    """Download the benchmark asset bundle into canonical local storage."""
    from robodojo.workflows.downloads import download_assets

    download_assets(paths(root), revision)


@assets_app.command("build-openarm")
def assets_build_openarm(
    root: Path | None = typer.Option(None, "--root", help="Repository containing the pinned OpenArm manifest."),
) -> None:
    """Build the pinned OpenArm robot assets into canonical local storage."""
    from robodojo.workflows.assets import build_openarm

    raise typer.Exit(build_openarm(paths(root)))


@assets_app.command("build-yam")
def assets_build_yam(
    root: Path | None = typer.Option(None, "--root", help="Repository containing the pinned YAM manifest."),
) -> None:
    """Build the pinned I2RT YAM robot assets into canonical local storage."""
    from robodojo.workflows.assets import build_yam

    raise typer.Exit(build_yam(paths(root)))


@assets_app.command("build-moonlake-office")
def assets_build_moonlake_office(
    root: Path | None = typer.Option(None, "--root", help="Repository containing the pinned Moonlake manifest."),
) -> None:
    """Build the pinned internal Moonlake office fixture into canonical local storage."""
    from robodojo.workflows.assets import build_moonlake_office

    raise typer.Exit(build_moonlake_office(paths(root)))


@assets_app.command("build-moonlake-packing")
def assets_build_moonlake_packing(
    root: Path | None = typer.Option(None, "--root", help="Repository containing the Moonlake packing manifest."),
) -> None:
    """Build internal Moonlake packing assets into canonical local storage."""
    from robodojo.workflows.assets import build_moonlake_packing

    raise typer.Exit(build_moonlake_packing(paths(root)))


@data_app.command("list")
def data_list() -> None:
    """List available dataset formats, sizes, and destination names."""
    from robodojo.workflows.downloads import list_data

    list_data()


@data_app.command("download")
def data_download(
    data_format: DataFormat = typer.Argument(
        ...,
        help="Dataset format to download; run `robodojo data list` to compare choices.",
    ),
    root: Path | None = typer.Option(None, "--root", help="Repository checkout used to resolve settings."),
    revision: str = typer.Option("main", "--revision", help="Git revision of the dataset repository."),
) -> None:
    """Download one benchmark dataset into canonical local storage."""
    from robodojo.workflows.downloads import download_data

    download_data(paths(root), data_format, revision)
