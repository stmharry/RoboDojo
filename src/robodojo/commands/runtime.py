"""Experiment validation, evaluation, and split-runtime command adapters."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from robodojo.commands.common import contract_values, model, paths
from robodojo.commands.options import (
    CheckpointLabelOption,
    DryRunOption,
    EnvironmentGpuOption,
    EnvironmentOption,
    EvaluationCountOption,
    PolicyGpuOption,
    PolicyProfileOption,
    ProtocolOption,
    RecipeOption,
    ReportFormat,
    ReportFormatOption,
    RepositoryRootOption,
    SceneOption,
    SeedOption,
    parse_evaluation_count,
)
from robodojo.core.models import (
    EvaluationRequest,
    PreflightRequest,
    ServerRequest,
    SimulatorLaunchRequest,
)


def doctor(
    recipe: RecipeOption = None,
    policy_profile: PolicyProfileOption = None,
    environment: EnvironmentOption = None,
    scene: SceneOption = None,
    protocol: ProtocolOption = None,
    skip_policy: bool = typer.Option(
        False,
        "--skip-policy",
        help="Check only RoboDojo and do not require a policy adapter.",
    ),
    root: RepositoryRootOption = None,
) -> None:
    """Validate the checkout, configuration, assets, and optional policy adapter."""
    from robodojo.workflows.doctor import run_doctor

    repository = paths(root)
    contract = contract_values(
        repository,
        recipe=recipe,
        policy_profile=policy_profile,
        environment=environment,
        scene=scene,
        protocol=protocol,
    )
    code = run_doctor(
        repository,
        contract["task"],
        contract["protocol"],
        contract["episode_horizon"],
        contract["native_eval_num"],
        contract["env_config"],
        None if skip_policy else contract["policy_dir"],
        contract["scene_config"],
    )
    raise typer.Exit(code)


def preflight(
    recipe: RecipeOption = None,
    policy_profile: PolicyProfileOption = None,
    environment: EnvironmentOption = None,
    scene: SceneOption = None,
    protocol: ProtocolOption = None,
    seed: SeedOption = 0,
    policy_gpu: PolicyGpuOption = "auto",
    env_gpu: EnvironmentGpuOption = "auto",
    publish: bool = typer.Option(False, "--publish", help="Validate publication prerequisites."),
    deep: bool = typer.Option(False, "--deep", help="Start the normal policy server on a temporary port."),
    timeout: Annotated[
        float,
        typer.Option("--timeout", min=0.001, help="Deep policy-server readiness timeout in seconds."),
    ] = 600,
    output_format: ReportFormatOption = ReportFormat.HUMAN,
    root: RepositoryRootOption = None,
) -> None:
    """Validate an experiment without installing, downloading, simulating, or publishing."""
    from robodojo.workflows.preflight import run_preflight

    repository = paths(root)
    contract = contract_values(
        repository,
        recipe=recipe,
        policy_profile=policy_profile,
        environment=environment,
        scene=scene,
        protocol=protocol,
    )
    request = model(
        PreflightRequest,
        **contract,
        seed=seed,
        policy_gpu=policy_gpu,
        env_gpu=env_gpu,
        publish=publish,
        deep=deep,
        timeout=timeout,
    )
    raise typer.Exit(run_preflight(repository, request, output_format=output_format.value))


def evaluate(
    recipe: RecipeOption = None,
    policy_profile: PolicyProfileOption = None,
    environment: EnvironmentOption = None,
    scene: SceneOption = None,
    protocol: ProtocolOption = None,
    seed: SeedOption = 0,
    policy_gpu: PolicyGpuOption = "auto",
    env_gpu: EnvironmentGpuOption = "auto",
    eval_num: EvaluationCountOption = "native",
    checkpoint_label: CheckpointLabelOption = None,
    export_scene: bool = typer.Option(
        False,
        "--export-scene",
        help="Export the selected scene layout before continuing with evaluation.",
    ),
    export_scene_only: bool = typer.Option(
        False,
        "--export-scene-only",
        help="Export the selected scene without starting a policy server or producing evaluation results.",
    ),
    export_scene_dir: Path | None = typer.Option(
        None,
        "--export-scene-dir",
        help="Directory for scene exports; the simulator default is used when omitted.",
    ),
    layout_id: Annotated[
        int,
        typer.Option("--layout-id", min=0, help="Nonnegative layout index used by scene export."),
    ] = 0,
    publish: bool = typer.Option(
        False,
        "--publish",
        help="After a successful evaluation, publish its completed run to ROBODOJO_S3_URI.",
    ),
    dry_run: DryRunOption = False,
    root: RepositoryRootOption = None,
) -> None:
    """Run a local policy server and simulator evaluation."""
    from robodojo.orchestration.evaluation import run_evaluation

    repository = paths(root)
    contract = contract_values(
        repository,
        recipe=recipe,
        policy_profile=policy_profile,
        environment=environment,
        scene=scene,
        protocol=protocol,
    )
    request = model(
        EvaluationRequest,
        **contract,
        seed=seed,
        policy_gpu=policy_gpu,
        env_gpu=env_gpu,
        eval_num=parse_evaluation_count(eval_num),
        checkpoint_label=checkpoint_label,
        export_scene=export_scene,
        export_scene_only=export_scene_only,
        export_scene_dir=export_scene_dir,
        layout_id=layout_id,
        publish=publish,
        dry_run=dry_run,
    )
    raise typer.Exit(run_evaluation(repository, request))


def server(
    recipe: RecipeOption = None,
    policy_profile: PolicyProfileOption = None,
    environment: EnvironmentOption = None,
    scene: SceneOption = None,
    protocol: ProtocolOption = None,
    seed: SeedOption = 0,
    policy_gpu: PolicyGpuOption = "auto",
    env_gpu: EnvironmentGpuOption = "auto",
    policy_port: Annotated[
        int | None,
        typer.Option(
            "--policy-port",
            min=1,
            max=65535,
            help="TCP port for the WebSocket server; an available port is chosen when omitted.",
        ),
    ] = None,
    bind_host: str = typer.Option(
        "0.0.0.0",
        "--bind-host",
        help="Interface or address on which the policy server listens.",
    ),
    dry_run: DryRunOption = False,
    root: RepositoryRootOption = None,
) -> None:
    """Start an XPolicyLab policy server adapter without simulator dependencies."""
    from robodojo.orchestration.split import run_server

    repository = paths(root)
    contract = contract_values(
        repository,
        recipe=recipe,
        policy_profile=policy_profile,
        environment=environment,
        scene=scene,
        protocol=protocol,
    )
    request = model(
        ServerRequest,
        **contract,
        seed=seed,
        policy_gpu=policy_gpu,
        env_gpu=env_gpu,
        port=policy_port,
        host=bind_host,
        dry_run=dry_run,
    )
    raise typer.Exit(run_server(repository, request))


def _run_client(
    *,
    root: Path | None,
    recipe: str | None,
    policy_profile: str | None,
    environment: str | None,
    scene: str | None,
    protocol: str | None,
    policy_host: str,
    policy_port: int,
    env_gpu: str,
    seed: int,
    eval_num: str,
    checkpoint_label: str | None,
    dry_run: bool,
    connect_timeout: float,
) -> int:
    from robodojo.core.gpu import GpuSelectionError, parse_gpu_selector, resolve_gpus
    from robodojo.core.storage import checkpoint_label as safe_checkpoint_label
    from robodojo.orchestration.split import run_client

    repository = paths(root)
    contract = contract_values(
        repository,
        recipe=recipe,
        policy_profile=policy_profile,
        environment=environment,
        scene=scene,
        protocol=protocol,
    )
    resolved_name = Path(contract["policy_dir"]).name
    label = safe_checkpoint_label(contract["checkpoint"], checkpoint_label)
    try:
        assignment = resolve_gpus(env_gpu=parse_gpu_selector(env_gpu))
    except GpuSelectionError as exc:
        typer.echo(f"GPU selection failed: {exc}", err=True)
        return 2
    request = model(
        SimulatorLaunchRequest,
        task=contract["task"],
        protocol_name=contract["protocol"],
        episode_horizon=contract["episode_horizon"],
        native_eval_num=contract["native_eval_num"],
        recipe=contract["recipe"],
        contract_hash=contract["contract_hash"],
        policy_name=resolved_name,
        policy_profile=contract["policy_profile"],
        policy_descriptor_hash=contract["policy_descriptor_hash"],
        policy_reference_match=contract["policy_reference_match"],
        host=policy_host,
        port=policy_port,
        env_config=contract["env_config"],
        scene_config=contract["scene_config"],
        env_gpu=assignment.env_gpu,
        seed=seed,
        eval_num=parse_evaluation_count(eval_num),
        additional_info=f"ckpt_name={label},action_type={contract['action_type']}",
        dry_run=dry_run,
    )
    return run_client(repository, request, connect_timeout=connect_timeout)


def client(
    policy_port: Annotated[
        int,
        typer.Option("--policy-port", min=1, max=65535, help="TCP port of the external policy WebSocket server."),
    ],
    recipe: RecipeOption = None,
    policy_profile: PolicyProfileOption = None,
    environment: EnvironmentOption = None,
    scene: SceneOption = None,
    protocol: ProtocolOption = None,
    policy_host: str = typer.Option(
        "127.0.0.1",
        "--policy-host",
        help="Hostname or IP address of the external policy server.",
    ),
    env_gpu: EnvironmentGpuOption = "auto",
    seed: SeedOption = 0,
    eval_num: EvaluationCountOption = "native",
    checkpoint_label: CheckpointLabelOption = None,
    connect_timeout: Annotated[
        float,
        typer.Option(
            "--connect-timeout",
            min=0.001,
            help="Seconds to wait for an initial server reachability check before warning and continuing.",
        ),
    ] = 5,
    dry_run: DryRunOption = False,
    root: RepositoryRootOption = None,
) -> None:
    """Run the simulator client against an external policy server."""
    raise typer.Exit(_run_client(**locals()))


def adapter_client(
    root_dir: Path = typer.Option(..., "--root-dir", "--root_dir"),
    task_name: str = typer.Option(..., "--task-name", "--task_name"),
    task_protocol: str = typer.Option(..., "--task-protocol", "--task_protocol"),
    episode_horizon: int = typer.Option(..., "--episode-horizon", "--episode_horizon"),
    native_eval_num: int = typer.Option(..., "--native-eval-num", "--native_eval_num"),
    scene_config: str = typer.Option(..., "--scene-config", "--scene_config"),
    env_config: str = typer.Option(..., "--env-cfg-type", "--env_cfg_type"),
    device_id: int = typer.Option(..., "--device-id", "--device_id"),
    policy_name: str = typer.Option(..., "--policy-name", "--policy_name"),
    port: int = typer.Option(..., "--port"),
    additional_info: str = typer.Option(..., "--additional-info", "--additional_info"),
    seed: int = typer.Option(0, "--seed"),
    host: str = typer.Option("localhost", "--host"),
    protocol: str = typer.Option("ws", "--protocol"),
    policy_server_url: str = typer.Option("", "--policy-server-url", "--policy_server_url"),
) -> None:
    """Private adapter used by unchanged XPolicyLab shell launchers."""
    from robodojo.orchestration.evaluation import run_simulator_session

    request = model(
        SimulatorLaunchRequest,
        task=task_name,
        protocol_name=task_protocol,
        episode_horizon=episode_horizon,
        native_eval_num=native_eval_num,
        policy_name=policy_name,
        host=host,
        port=port,
        env_config=env_config,
        scene_config=scene_config,
        env_gpu=device_id,
        seed=seed,
        eval_num=os.environ.get("EVAL_NUM", "native"),
        additional_info=additional_info,
        protocol=protocol,
        policy_server_url=policy_server_url,
    )
    raise typer.Exit(run_simulator_session(paths(root_dir), request))
