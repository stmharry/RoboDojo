"""Repository-local setup command adapter."""

from __future__ import annotations

import typer

from robodojo.commands.common import experiment_spec, model, paths
from robodojo.commands.options import (
    EnvironmentOption,
    PolicyGpuOption,
    PolicyProfileOption,
    RecipeOption,
    ReportFormat,
    ReportFormatOption,
    RepositoryRootOption,
    SceneOption,
    SeedOption,
    TaskProtocolOption,
)
from robodojo.core.models.requests import (
    SetupRequest,
    SetupStage,
)


def setup(
    only: list[SetupStage] | None = typer.Option(
        None,
        "--only",
        help="Prepare only this stage; repeat for root, assets, or policy.",
    ),
    recipe: RecipeOption = None,
    policy_profile: PolicyProfileOption = None,
    environment: EnvironmentOption = None,
    scene: SceneOption = None,
    task_protocol: TaskProtocolOption = None,
    seed: SeedOption = 0,
    policy_gpu: PolicyGpuOption = "auto",
    output_format: ReportFormatOption = ReportFormat.HUMAN,
    root: RepositoryRootOption = None,
) -> None:
    """Prepare repository-local dependencies, assets, policy runtime, and checkpoint."""
    from robodojo.workflows.setup import setup as setup_workflow

    repository = paths(root)
    selected = set(only or tuple(SetupStage))
    experiment = None
    if selected != {SetupStage.ROOT}:
        experiment = experiment_spec(
            repository,
            recipe=recipe,
            policy_profile=policy_profile,
            environment=environment,
            scene=scene,
            task_protocol=task_protocol,
        )
    request = model(
        SetupRequest,
        experiment=experiment,
        stages=tuple(only or ()),
        seed=seed,
        policy_gpu=policy_gpu,
    )
    raise typer.Exit(setup_workflow(repository, request, output_format=output_format.value))
