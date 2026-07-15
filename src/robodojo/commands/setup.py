"""Repository-local setup command adapter."""

from __future__ import annotations

from pathlib import Path

import typer

from robodojo.commands.common import model, paths, report_format
from robodojo.core.models import SetupRequest, SetupStage


def setup(
    only: list[SetupStage] | None = typer.Option(
        None,
        "--only",
        help="Prepare only this stage; repeat for root, assets, or policy.",
    ),
    policy_dir: Path | None = typer.Option(None, "--policy-dir", help="XPolicyLab policy adapter directory."),
    task: str | None = typer.Option(None, "--task", help="Canonical task used to resolve assets and policy setup."),
    checkpoint: str | None = typer.Option(None, "--ckpt", help="Checkpoint alias or path to prepare."),
    policy_env: str | None = typer.Option(None, "--policy-env", help="Policy runtime environment name or path."),
    dataset: str = typer.Option("RoboDojo", "--dataset", help="Benchmark or dataset family."),
    env_config: str | None = typer.Option(None, "--env-cfg", help="Environment profile used to infer assets."),
    scene_config: str | None = typer.Option(None, "--scene", help="Optional scene profile override."),
    action_type: str | None = typer.Option(None, "--action-type", help="Policy action representation."),
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

    request = model(
        SetupRequest,
        stages=tuple(only or ()),
        policy_dir=policy_dir,
        task=task,
        checkpoint=checkpoint,
        policy_env=policy_env,
        dataset=dataset,
        env_config=env_config,
        scene_config=scene_config,
        action_type=action_type,
        seed=seed,
        policy_gpu=policy_gpu,
    )
    raise typer.Exit(setup_workflow(paths(root), request, output_format=report_format(output_format)))
