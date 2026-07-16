"""First-frame snapshot command adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from robodojo.commands.common import model, paths
from robodojo.commands.options import DryRunOption, EnvironmentGpuOption, RepositoryRootOption, SeedOption
from robodojo.core.models import SnapshotBatchRequest


def snapshots(
    recipe: Annotated[
        list[str] | None,
        typer.Option(
            "--recipe",
            help="Recipe to capture; repeat for multiple recipes. All tracked recipes are used when omitted.",
        ),
    ] = None,
    seed: SeedOption = 0,
    layout_id: Annotated[
        int,
        typer.Option("--layout-id", min=0, help="Nonnegative layout index captured for every recipe."),
    ] = 0,
    env_gpu: EnvironmentGpuOption = "auto",
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Batch output directory; defaults to ROBODOJO_STORAGE_ROOT/runs/snapshots/<timestamp>.",
    ),
    export_scene: bool = typer.Option(
        False,
        "--export-scene",
        help="Also create the referenced USDA, flattened USDC, and preview USDZ scene bundle.",
    ),
    publish: bool = typer.Option(
        False,
        "--publish",
        help="Publish the completed batch after every recipe succeeds.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume an exact existing --output-dir and reuse completed recipe bundles.",
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast",
        help="Stop after the first failed recipe instead of recording all failures.",
    ),
    dry_run: DryRunOption = False,
    root: RepositoryRootOption = None,
) -> None:
    """Capture the first rollout RGB observation for selected evaluation recipes."""
    from robodojo.workflows.snapshots import run_snapshot_batch

    request = model(
        SnapshotBatchRequest,
        recipes=tuple(recipe or ()),
        seed=seed,
        layout_id=layout_id,
        env_gpu=env_gpu,
        output_dir=output_dir,
        export_scene=export_scene,
        publish=publish,
        resume=resume,
        fail_fast=fail_fast,
        dry_run=dry_run,
    )
    raise typer.Exit(run_snapshot_batch(paths(root), request))
