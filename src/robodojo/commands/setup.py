"""Repository-local setup command adapter."""

from __future__ import annotations

from pathlib import Path

import typer

from robodojo.commands.common import contract_values, model, paths, report_format
from robodojo.core.models import SetupRequest, SetupStage


def setup(
    only: list[SetupStage] | None = typer.Option(
        None,
        "--only",
        help="Prepare only this stage; repeat for root, assets, or policy.",
    ),
    recipe: str | None = typer.Option(None, "--recipe", help="Tracked evaluation recipe."),
    policy_profile: str | None = typer.Option(None, "--policy-profile", help="Manual policy profile."),
    environment: str | None = typer.Option(None, "--environment", help="Manual environment profile."),
    scene: str | None = typer.Option(None, "--scene", help="Manual scene profile."),
    protocol: str | None = typer.Option(None, "--protocol", help="Manual task protocol."),
    seed: int = typer.Option(0, "--seed", help="Nonnegative experiment seed."),
    policy_gpu: str = typer.Option(
        "auto",
        "--policy-gpu",
        envvar="POLICY_GPU",
        help="Policy GPU as a zero-based index or auto; POLICY_GPU is used when the flag is omitted.",
    ),
    output_format: str = typer.Option("human", "--format", help="Report format: human or json."),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Repository checkout to prepare; auto-detected when omitted.",
    ),
) -> None:
    """Prepare repository-local dependencies, assets, policy runtime, and checkpoint."""
    from robodojo.workflows.setup import setup as setup_workflow

    repository = paths(root)
    selected = set(only or tuple(SetupStage))
    contract = {}
    if selected != {SetupStage.ROOT}:
        contract = contract_values(
            repository,
            recipe=recipe,
            policy_profile=policy_profile,
            environment=environment,
            scene=scene,
            protocol=protocol,
        )
    request = model(
        SetupRequest,
        **contract,
        stages=tuple(only or ()),
        seed=seed,
        policy_gpu=policy_gpu,
    )
    raise typer.Exit(setup_workflow(repository, request, output_format=report_format(output_format)))
