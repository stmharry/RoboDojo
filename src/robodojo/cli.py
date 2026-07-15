"""Unified Typer command-line interface for RoboDojo."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import typer

from robodojo.commands.assets import assets_app, data_app
from robodojo.commands.common import model as _model, paths as _paths, report_format as _report_format
from robodojo.commands.docker import docker_app
from robodojo.commands.results import results_app
from robodojo.commands.setup import setup as setup_command
from robodojo.commands.storage import storage_app
from robodojo.core.logging import LOG_LEVEL_ENV, configure_logging, parse_log_level
from robodojo.core.models import (
    EvaluationRequest,
    PreflightRequest,
    ServerRequest,
    SimulatorLaunchRequest,
    SweepRequest,
)

app = typer.Typer(no_args_is_help=True, help="RoboDojo evaluation and operations CLI.")
app.add_typer(assets_app, name="assets")
app.add_typer(data_app, name="data")
app.add_typer(storage_app, name="storage")
app.add_typer(results_app, name="results")
app.add_typer(docker_app, name="docker")
app.command(name="setup")(setup_command)


@app.callback()
def configure_cli_logging(
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        help="RoboDojo diagnostic level: DEBUG, INFO, WARNING, ERROR, or CRITICAL.",
    ),
) -> None:
    """Configure RoboDojo diagnostics before dispatching a command."""
    try:
        parse_log_level(log_level)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--log-level") from exc
    if log_level is not None:
        os.environ[LOG_LEVEL_ENV] = log_level.strip().upper()
    configure_logging(log_level)


@app.command()
def doctor(
    task: str = typer.Option("stack_bowls", "--task", help="Task whose configuration and assets should be checked."),
    env_config: str = typer.Option(
        "arx_x5",
        "--env-cfg",
        help="Environment profile that selects the robot, camera, simulator, and default scene.",
    ),
    scene_config: str | None = typer.Option(
        None,
        "--scene",
        help="Scene configuration override; otherwise task and environment defaults apply.",
    ),
    policy_dir: Path | None = typer.Option(
        None,
        "--policy-dir",
        help="XPolicyLab policy directory to validate in addition to the simulator checkout.",
    ),
    skip_policy: bool = typer.Option(
        False,
        "--skip-policy",
        help="Check only RoboDojo and do not require a policy adapter.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Repository checkout to inspect; auto-detected when omitted.",
    ),
) -> None:
    """Validate the checkout, configuration, assets, and optional policy adapter."""
    from robodojo.workflows.doctor import run_doctor

    code = run_doctor(_paths(root), task, env_config, None if skip_policy else policy_dir, scene_config)
    raise typer.Exit(code)


@app.command()
def tasks(
    format: str = typer.Option(
        "plain",
        "--format",
        help="Output format: plain, json, or markdown.",
    ),
    only_runnable: bool = typer.Option(
        False,
        "--only-runnable",
        help="In plain output, omit tasks whose code or configuration is incomplete.",
    ),
    check: bool = typer.Option(
        False,
        "--check",
        help="Exit nonzero when any canonical task is not runnable.",
    ),
) -> None:
    """List and validate canonical task implementations and configurations."""
    import json

    from robodojo.workflows.task_inventory import _print_markdown, _print_plain, build_inventory

    inventory = build_inventory()
    if format == "json":
        typer.echo(json.dumps(inventory, indent=2, sort_keys=True))
    elif format == "markdown":
        _print_markdown(inventory)
    elif format == "plain":
        _print_plain(inventory, only_runnable)
    else:
        raise typer.BadParameter("expected plain, json, or markdown", param_hint="--format")
    if check and any(not record["runnable"] for record in inventory["tasks"]):
        raise typer.Exit(1)


@app.command()
def preflight(
    policy_dir: Path = typer.Option(..., "--policy-dir", help="XPolicyLab policy adapter directory."),
    task: str = typer.Option(..., "--task", help="Canonical evaluation task."),
    checkpoint: str = typer.Option(..., "--ckpt", help="Checkpoint alias or path."),
    policy_env: str = typer.Option(..., "--policy-env", help="Policy runtime environment name or project path."),
    dataset: str = typer.Option("RoboDojo", "--dataset", help="Benchmark or dataset family."),
    env_config: str = typer.Option("arx_x5", "--env-cfg", help="Environment profile."),
    scene_config: str | None = typer.Option(None, "--scene", help="Optional scene profile override."),
    action_type: str = typer.Option("ee", "--action-type", help="Policy action representation."),
    seed: int = typer.Option(0, "--seed", help="Nonnegative experiment seed."),
    policy_gpu: str = typer.Option(
        "auto",
        "--policy-gpu",
        envvar="POLICY_GPU",
        help="Policy GPU as a zero-based index or auto; POLICY_GPU is used when the flag is omitted.",
    ),
    env_gpu: str = typer.Option(
        "auto",
        "--env-gpu",
        envvar="ENV_GPU",
        help="Simulator GPU as a zero-based index or auto; ENV_GPU is used when the flag is omitted.",
    ),
    publish: bool = typer.Option(False, "--publish", help="Validate publication prerequisites."),
    deep: bool = typer.Option(False, "--deep", help="Start the normal policy server on a temporary port."),
    timeout: float = typer.Option(600, "--timeout", help="Deep policy-server readiness timeout in seconds."),
    output_format: str = typer.Option("human", "--format", help="Report format: human or json."),
    root: Path | None = typer.Option(None, "--root", help="Repository checkout to inspect."),
) -> None:
    """Validate an experiment without installing, downloading, simulating, or publishing."""
    from robodojo.workflows.preflight import run_preflight

    request = _model(
        PreflightRequest,
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
        env_gpu=env_gpu,
        publish=publish,
        deep=deep,
        timeout=timeout,
    )
    raise typer.Exit(run_preflight(_paths(root), request, output_format=_report_format(output_format)))


def _evaluation_request(
    *,
    policy_dir: Path,
    task: str,
    checkpoint: str,
    policy_env: str,
    dataset: str,
    env_config: str,
    scene_config: str | None,
    action_type: str,
    seed: int,
    policy_gpu: str,
    env_gpu: str,
    eval_num: str | None,
    checkpoint_label: str | None,
    export_scene: bool,
    export_scene_only: bool,
    export_scene_dir: Path | None,
    layout_id: int,
    publish: bool,
    dry_run: bool,
) -> EvaluationRequest:
    parsed_eval_num: int | str | None = eval_num
    if eval_num is not None and eval_num != "native":
        try:
            parsed_eval_num = int(eval_num)
        except ValueError as exc:
            raise typer.BadParameter("expected a positive integer or native", param_hint="--eval-num") from exc
    return _model(
        EvaluationRequest,
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
        env_gpu=env_gpu,
        eval_num=parsed_eval_num,
        checkpoint_label=checkpoint_label,
        export_scene=export_scene,
        export_scene_only=export_scene_only,
        export_scene_dir=export_scene_dir,
        layout_id=layout_id,
        publish=publish,
        dry_run=dry_run,
    )


@app.command("eval")
def evaluate(
    policy_dir: Path = typer.Option(
        ...,
        "--policy-dir",
        help="XPolicyLab policy directory containing setup_eval_policy_server.sh.",
    ),
    task: str = typer.Option(..., "--task", help="Canonical task module and configuration name to evaluate."),
    checkpoint: str = typer.Option(
        ...,
        "--ckpt",
        help="Checkpoint name or path passed unchanged to the policy adapter.",
    ),
    policy_env: str = typer.Option(
        ...,
        "--policy-env",
        help="Policy runtime environment name, environment path, or uv project understood by the adapter.",
    ),
    dataset: str = typer.Option(
        "RoboDojo",
        "--dataset",
        help="Benchmark or dataset family passed to XPolicyLab as bench_name.",
    ),
    env_config: str = typer.Option(
        "arx_x5",
        "--env-cfg",
        help="Environment profile selecting the robot, cameras, simulator, and default scene.",
    ),
    scene_config: str | None = typer.Option(
        None,
        "--scene",
        help="Scene configuration override; otherwise task and environment defaults apply.",
    ),
    action_type: str = typer.Option(
        "ee",
        "--action-type",
        help="Policy action space passed to XPolicyLab, typically ee or joint.",
    ),
    seed: int = typer.Option(0, "--seed", help="Nonnegative evaluation seed used for task layout and policy setup."),
    policy_gpu: str = typer.Option(
        "auto",
        "--policy-gpu",
        envvar="POLICY_GPU",
        help="Policy GPU as a zero-based index or auto; POLICY_GPU is used when the flag is omitted.",
    ),
    env_gpu: str = typer.Option(
        "auto",
        "--env-gpu",
        envvar="ENV_GPU",
        help="Simulator GPU as a zero-based index or auto; ENV_GPU is used when the flag is omitted.",
    ),
    eval_num: str | None = typer.Option(
        None,
        "--eval-num",
        help="Episode count as a positive integer, or native to keep the simulator config value.",
    ),
    checkpoint_label: str | None = typer.Option(
        None,
        "--ckpt-label",
        help="Filesystem-safe result label; defaults to the checkpoint name or path basename.",
    ),
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
    layout_id: int = typer.Option(
        0,
        "--layout-id",
        help="Nonnegative layout index used by scene export.",
    ),
    publish: bool = typer.Option(
        False,
        "--publish",
        help="After a successful evaluation, publish its completed run to ROBODOJO_S3_URI.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print resolved policy and simulator commands without starting processes or accessing S3.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Repository checkout to use; auto-detected when omitted.",
    ),
) -> None:
    """Run a local policy server and simulator evaluation."""
    request = _evaluation_request(**{key: value for key, value in locals().items() if key != "root"})
    from robodojo.orchestration.evaluation import run_evaluation

    raise typer.Exit(run_evaluation(_paths(root), request))


@app.command()
def server(
    policy_dir: Path = typer.Option(
        ...,
        "--policy-dir",
        help="XPolicyLab policy directory containing setup_eval_policy_server.sh.",
    ),
    task: str = typer.Option(..., "--task", help="Task name passed to the policy adapter."),
    checkpoint: str = typer.Option(
        ...,
        "--ckpt",
        help="Checkpoint name or path passed unchanged to the policy adapter.",
    ),
    policy_env: str = typer.Option(
        ...,
        "--policy-env",
        help="Policy runtime environment name, environment path, or uv project understood by the adapter.",
    ),
    dataset: str = typer.Option(
        "RoboDojo",
        "--dataset",
        help="Benchmark or dataset family passed to XPolicyLab as bench_name.",
    ),
    env_config: str = typer.Option(
        "arx_x5",
        "--env-cfg",
        help="Robot and observation profile passed to the policy adapter.",
    ),
    action_type: str = typer.Option(
        "ee",
        "--action-type",
        help="Policy action space passed to XPolicyLab, typically ee or joint.",
    ),
    seed: int = typer.Option(0, "--seed", help="Nonnegative seed passed to the policy adapter."),
    policy_gpu: str = typer.Option(
        "auto",
        "--policy-gpu",
        envvar="POLICY_GPU",
        help="Policy GPU as a zero-based index or auto; POLICY_GPU is used when the flag is omitted.",
    ),
    env_gpu: str = typer.Option(
        "auto",
        "--env-gpu",
        envvar="ENV_GPU",
        help="Simulator GPU as a zero-based index or auto; ENV_GPU is used when the flag is omitted.",
    ),
    scene_config: str | None = typer.Option(
        None,
        "--scene",
        help="Scene profile validated before the split policy server starts.",
    ),
    policy_port: int | None = typer.Option(
        None,
        "--policy-port",
        help="TCP port for the WebSocket server; an available port is chosen when omitted.",
    ),
    bind_host: str = typer.Option(
        "0.0.0.0",
        "--bind-host",
        help="Interface or address on which the policy server listens.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the resolved policy-server command without starting it.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Repository checkout whose simulator-side experiment contract should be validated.",
    ),
) -> None:
    """Start an XPolicyLab policy server adapter without simulator dependencies."""
    from robodojo.orchestration.split import run_server

    request = _model(
        ServerRequest,
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
        env_gpu=env_gpu,
        port=policy_port,
        host=bind_host,
        dry_run=dry_run,
    )
    raise typer.Exit(run_server(_paths(root), request))


def _client(
    *,
    root: Path | None,
    task: str,
    policy_name: str | None,
    policy_dir: Path | None,
    policy_host: str,
    policy_port: int,
    env_config: str,
    scene_config: str | None,
    env_gpu: str,
    seed: int,
    eval_num: str,
    checkpoint: str,
    checkpoint_label: str | None,
    action_type: str,
    dry_run: bool,
    connect_timeout: float | None = None,
) -> int:
    from robodojo.core.storage import checkpoint_label as safe_checkpoint_label
    from robodojo.orchestration.split import run_client

    resolved_name = policy_name or (policy_dir.resolve().name if policy_dir else None)
    if not resolved_name:
        raise typer.BadParameter("provide --policy-name or --policy-dir")
    parsed_eval_num: int | str = eval_num if eval_num == "native" else int(eval_num)
    label = safe_checkpoint_label(checkpoint, checkpoint_label)
    from robodojo.core.gpu import GpuSelectionError, parse_gpu_selector, resolve_gpus

    try:
        assignment = resolve_gpus(env_gpu=parse_gpu_selector(env_gpu))
    except GpuSelectionError as exc:
        typer.echo(f"GPU selection failed: {exc}", err=True)
        return 2
    request = _model(
        SimulatorLaunchRequest,
        task=task,
        policy_name=resolved_name,
        host=policy_host,
        port=policy_port,
        env_config=env_config,
        scene_config=scene_config,
        env_gpu=assignment.env_gpu,
        seed=seed,
        eval_num=parsed_eval_num,
        additional_info=f"ckpt_name={label},action_type={action_type}",
        dry_run=dry_run,
    )
    return run_client(_paths(root), request, connect_timeout=connect_timeout if connect_timeout is not None else 5)


@app.command()
def client(
    task: str = typer.Option(..., "--task", help="Canonical task module and configuration name to evaluate."),
    policy_name: str | None = typer.Option(
        None,
        "--policy-name",
        help="XPolicyLab policy directory name; required unless --policy-dir supplies it.",
    ),
    policy_dir: Path | None = typer.Option(
        None,
        "--policy-dir",
        help="Policy directory used only to derive the policy name; alternatively pass --policy-name.",
    ),
    policy_host: str = typer.Option(
        "127.0.0.1",
        "--policy-host",
        help="Hostname or IP address of the external policy server.",
    ),
    policy_port: int = typer.Option(..., "--policy-port", help="TCP port of the external policy WebSocket server."),
    env_config: str = typer.Option(
        "arx_x5",
        "--env-cfg",
        help="Environment profile selecting the robot, cameras, simulator, and default scene.",
    ),
    scene_config: str | None = typer.Option(
        None,
        "--scene",
        help="Scene configuration override; otherwise task and environment defaults apply.",
    ),
    env_gpu: str = typer.Option(
        "auto",
        "--env-gpu",
        envvar="ENV_GPU",
        help="Simulator GPU as a zero-based index or auto; ENV_GPU is used when the flag is omitted.",
    ),
    seed: int = typer.Option(0, "--seed", help="Nonnegative evaluation seed used for task layout."),
    eval_num: str = typer.Option(
        "1",
        "--eval-num",
        help="Episode count as a positive integer, or native to keep the simulator config value.",
    ),
    checkpoint: str = typer.Option(
        "external",
        "--ckpt",
        help="Checkpoint identifier recorded in local result metadata.",
    ),
    checkpoint_label: str | None = typer.Option(
        None,
        "--ckpt-label",
        help="Filesystem-safe result label; defaults to the checkpoint name or path basename.",
    ),
    action_type: str = typer.Option(
        "ee",
        "--action-type",
        help="Policy action space recorded for the evaluation, typically ee or joint.",
    ),
    connect_timeout: float = typer.Option(
        5,
        "--connect-timeout",
        help="Seconds to wait for an initial server reachability check before warning and continuing.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the resolved simulator command without checking the server or starting Isaac Sim.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Repository checkout to use; auto-detected when omitted.",
    ),
) -> None:
    """Run the simulator client against an external policy server."""
    raise typer.Exit(_client(**locals()))


def _sweep_command(
    *,
    mode: str,
    policy_dir: Path,
    checkpoint: str,
    policy_env: str,
    env_config: str,
    scene_config: str | None,
    action_type: str,
    seed: int,
    policy_gpu: str,
    env_gpu: str,
    eval_num: str,
    only: str | None,
    tasks_file: Path | None,
    limit: int | None,
    resume: bool,
    fail_fast: bool,
    run_id: str | None,
    dry_run: bool,
    root: Path | None,
) -> int:
    from robodojo.workflows.sweeps import run_sweep

    parsed_eval_num: int | str = eval_num if eval_num == "native" else int(eval_num)
    request = _model(
        SweepRequest,
        policy_dir=policy_dir,
        checkpoint=checkpoint,
        policy_env=policy_env,
        env_config=env_config,
        scene_config=scene_config,
        action_type=action_type,
        seed=seed,
        policy_gpu=policy_gpu,
        env_gpu=env_gpu,
        eval_num=parsed_eval_num,
        only=tuple(item.strip() for item in (only or "").split(",") if item.strip()),
        tasks_file=tasks_file,
        limit=limit,
        resume=resume,
        fail_fast=fail_fast,
        run_id=run_id,
        dry_run=dry_run,
    )
    return run_sweep(_paths(root), request)


def _sweep_options(mode: str, **values: Any) -> None:
    raise typer.Exit(_sweep_command(mode=mode, **values))


@app.command()
def smoke(
    policy_dir: Path = typer.Option(
        ...,
        "--policy-dir",
        help="XPolicyLab policy directory containing setup_eval_policy_server.sh.",
    ),
    checkpoint: str = typer.Option(..., "--ckpt", help="Checkpoint name or path passed to the policy adapter."),
    policy_env: str = typer.Option(
        ...,
        "--policy-env",
        help="Policy runtime environment name, environment path, or uv project understood by the adapter.",
    ),
    env_config: str = typer.Option(
        "arx_x5",
        "--env-cfg",
        help="Environment profile used for every selected task.",
    ),
    scene_config: str | None = typer.Option(
        None,
        "--scene",
        help="Scene override used for every selected task; normal task defaults apply when omitted.",
    ),
    action_type: str = typer.Option(
        "ee",
        "--action-type",
        help="Policy action space passed to XPolicyLab, typically ee or joint.",
    ),
    seed: int = typer.Option(0, "--seed", help="Nonnegative seed used for each evaluation in the sweep."),
    policy_gpu: str = typer.Option(
        "auto",
        "--policy-gpu",
        envvar="POLICY_GPU",
        help="Policy GPU as a zero-based index or auto; POLICY_GPU is used when the flag is omitted.",
    ),
    env_gpu: str = typer.Option(
        "auto",
        "--env-gpu",
        envvar="ENV_GPU",
        help="Simulator GPU as a zero-based index or auto; ENV_GPU is used when the flag is omitted.",
    ),
    eval_num: str = typer.Option(
        "1",
        "--eval-num",
        help="Episodes per task as a positive integer, or native for each simulator config.",
    ),
    only: str | None = typer.Option(
        None,
        "--only",
        help="Comma-separated task names to run instead of the full runnable inventory.",
    ),
    tasks_file: Path | None = typer.Option(
        None,
        "--tasks-file",
        help="Text file of task names to add to --only selections, one name per line.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Run at most this many selected tasks after filtering.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Reuse the named sweep summary and skip task/scene pairs that already passed.",
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast",
        help="Stop the sweep immediately after the first failed evaluation.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Stable sweep identifier used for summary paths and required to resume a specific run.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Resolve selections and print child commands without starting policy or simulator processes.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Repository checkout to use; auto-detected when omitted.",
    ),
) -> None:
    """Run a one-episode sequential task sweep."""
    _sweep_options("smoke", **locals())


@app.command()
def benchmark(
    policy_dir: Path = typer.Option(
        ...,
        "--policy-dir",
        help="XPolicyLab policy directory containing setup_eval_policy_server.sh.",
    ),
    checkpoint: str = typer.Option(..., "--ckpt", help="Checkpoint name or path passed to the policy adapter."),
    policy_env: str = typer.Option(
        ...,
        "--policy-env",
        help="Policy runtime environment name, environment path, or uv project understood by the adapter.",
    ),
    eval_num: str = typer.Option(
        ...,
        "--eval-num",
        help="Episodes per task as a positive integer, or native for each simulator config.",
    ),
    env_config: str = typer.Option(
        "arx_x5",
        "--env-cfg",
        help="Environment profile used for every selected task.",
    ),
    scene_config: str | None = typer.Option(
        None,
        "--scene",
        help="Scene override used for every selected task; normal task defaults apply when omitted.",
    ),
    action_type: str = typer.Option(
        "ee",
        "--action-type",
        help="Policy action space passed to XPolicyLab, typically ee or joint.",
    ),
    seed: int = typer.Option(0, "--seed", help="Nonnegative seed used for each evaluation in the sweep."),
    policy_gpu: str = typer.Option(
        "auto",
        "--policy-gpu",
        envvar="POLICY_GPU",
        help="Policy GPU as a zero-based index or auto; POLICY_GPU is used when the flag is omitted.",
    ),
    env_gpu: str = typer.Option(
        "auto",
        "--env-gpu",
        envvar="ENV_GPU",
        help="Simulator GPU as a zero-based index or auto; ENV_GPU is used when the flag is omitted.",
    ),
    only: str | None = typer.Option(
        None,
        "--only",
        help="Comma-separated task names to run instead of the full runnable inventory.",
    ),
    tasks_file: Path | None = typer.Option(
        None,
        "--tasks-file",
        help="Text file of task names to add to --only selections, one name per line.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Run at most this many selected tasks after filtering.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Reuse the named sweep summary and skip task/scene pairs that already passed.",
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast",
        help="Stop the sweep immediately after the first failed evaluation.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Stable sweep identifier used for summary paths and required to resume a specific run.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Resolve selections and print child commands without starting policy or simulator processes.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Repository checkout to use; auto-detected when omitted.",
    ),
) -> None:
    """Run a sequential benchmark sweep."""
    _sweep_options("benchmark", **locals())


@app.command(
    "_adapter-client",
    hidden=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def adapter_client(
    root_dir: Path = typer.Option(..., "--root-dir", "--root_dir"),
    task_name: str = typer.Option(..., "--task-name", "--task_name"),
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

    request = _model(
        SimulatorLaunchRequest,
        task=task_name,
        policy_name=policy_name,
        host=host,
        port=port,
        env_config=env_config,
        env_gpu=device_id,
        seed=seed,
        eval_num=os.environ.get("EVAL_NUM", "native"),
        additional_info=additional_info,
        protocol=protocol,
        policy_server_url=policy_server_url,
    )
    raise typer.Exit(run_simulator_session(_paths(root_dir), request))


if __name__ == "__main__":
    app()
