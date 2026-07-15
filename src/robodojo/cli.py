"""Unified Typer command-line interface for RoboDojo."""

from __future__ import annotations

import os
from pathlib import Path
import socket
from typing import Any

from pydantic import ValidationError
import typer

from robodojo.core.logging import LOG_LEVEL_ENV, configure_logging, parse_log_level
from robodojo.core.models import (
    DataFormat,
    EvaluationRequest,
    PolicyServerLaunchRequest,
    SimulatorLaunchRequest,
    SweepRequest,
    UpstreamOutputFormat,
    UpstreamProject,
)
from robodojo.core.paths import RepositoryPaths

app = typer.Typer(no_args_is_help=True, help="RoboDojo evaluation and operations CLI.")
assets_app = typer.Typer(no_args_is_help=True, help="Download and build benchmark assets.")
data_app = typer.Typer(no_args_is_help=True, help="Download benchmark datasets.")
storage_app = typer.Typer(no_args_is_help=True, help="Manage durable RoboDojo storage.")
results_app = typer.Typer(no_args_is_help=True, help="Analyze evaluation results.")
docker_app = typer.Typer(no_args_is_help=True, help="Build and validate RoboDojo containers.")
upstream_app = typer.Typer(no_args_is_help=True, help="Review official upstream changes and local parity.")
app.add_typer(assets_app, name="assets")
app.add_typer(data_app, name="data")
app.add_typer(storage_app, name="storage")
app.add_typer(results_app, name="results")
app.add_typer(docker_app, name="docker")
app.add_typer(upstream_app, name="upstream")


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


def _paths(root: Path | None = None) -> RepositoryPaths:
    try:
        paths = RepositoryPaths.resolve(root)
        from robodojo.core.settings import RuntimeSettings

        RuntimeSettings.load(paths).export_missing()
        return paths
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


def _model(model: type[Any], **values: Any) -> Any:
    try:
        return model.model_validate(values)
    except ValidationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@app.command()
def install(
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Repository checkout to install; auto-detected when omitted.",
    ),
    from_step: str = typer.Option(
        "system",
        "--from",
        help="Resume at this installation stage: system, submodules, or sync.",
    ),
) -> None:
    """Install system dependencies, submodules, and the locked simulator environment."""
    from robodojo.workflows.install import InstallStep, install as install_workflow

    try:
        step = InstallStep(from_step)
    except ValueError as exc:
        raise typer.BadParameter("expected system, submodules, or sync", param_hint="--from") from exc
    install_workflow(_paths(root), step)


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


@upstream_app.command("check")
def upstream_check(
    project: UpstreamProject = typer.Option(
        UpstreamProject.ALL,
        "--project",
        help="Official project to inspect: all, robodojo, or xpolicylab.",
    ),
    format: UpstreamOutputFormat = typer.Option(
        UpstreamOutputFormat.PLAIN,
        "--format",
        help="Report format: plain or json.",
    ),
    ref: str | None = typer.Option(
        None,
        "--ref",
        help="Override the manifest branch or ref for the selected project(s).",
    ),
    source: Path | None = typer.Option(
        None,
        "--source",
        help="Inspect one local upstream checkout instead of fetching its official repository.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="RoboDojo checkout whose manifest and mapped files should be validated.",
    ),
) -> None:
    """Detect official changes and verify mapped local compatibility contracts."""
    from robodojo.workflows.upstream import check_upstreams, format_upstream_report, json_upstream_report

    report, code = check_upstreams(_paths(root), project=project, ref=ref, source=source)
    rendered = json_upstream_report(report) if format == UpstreamOutputFormat.JSON else format_upstream_report(report)
    typer.echo(rendered)
    raise typer.Exit(code)


def _evaluation_request(
    *,
    policy_dir: Path,
    task: str,
    checkpoint: str,
    policy_env: str,
    dataset: str,
    env_config: str,
    scene_config: str | None,
    expert_num: int,
    action_type: str,
    seed: int,
    policy_gpu: int,
    env_gpu: int,
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
        expert_num=expert_num,
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
    expert_num: int = typer.Option(
        100,
        "--expert-num",
        help="Reserved expert-count compatibility setting; it does not alter the current simulator loop.",
    ),
    action_type: str = typer.Option(
        "ee",
        "--action-type",
        help="Policy action space passed to XPolicyLab, typically ee or joint.",
    ),
    seed: int = typer.Option(0, "--seed", help="Nonnegative evaluation seed used for task layout and policy setup."),
    policy_gpu: int = typer.Option(
        0,
        "--policy-gpu",
        help="Zero-based GPU index exposed to the policy server.",
    ),
    env_gpu: int = typer.Option(
        0,
        "--env-gpu",
        help="Zero-based GPU index exposed to the simulator.",
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
    policy_gpu: int = typer.Option(
        0,
        "--policy-gpu",
        help="Zero-based GPU index exposed to the policy server.",
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
) -> None:
    """Start an XPolicyLab policy server adapter without simulator dependencies."""
    from robodojo.policy.adapter import run_policy_server

    request = _model(
        PolicyServerLaunchRequest,
        policy_dir=policy_dir,
        task=task,
        checkpoint=checkpoint,
        policy_env=policy_env,
        dataset=dataset,
        env_config=env_config,
        action_type=action_type,
        seed=seed,
        policy_gpu=policy_gpu,
        port=policy_port,
        host=bind_host,
        dry_run=dry_run,
    )
    raise typer.Exit(run_policy_server(request))


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
    env_gpu: int,
    seed: int,
    eval_num: str,
    checkpoint: str,
    checkpoint_label: str | None,
    action_type: str,
    dry_run: bool,
    connect_timeout: float | None = None,
) -> int:
    from robodojo.core.storage import checkpoint_label as safe_checkpoint_label
    from robodojo.orchestration.evaluation import run_simulator_session

    resolved_name = policy_name or (policy_dir.resolve().name if policy_dir else None)
    _ = connect_timeout
    if not resolved_name:
        raise typer.BadParameter("provide --policy-name or --policy-dir")
    parsed_eval_num: int | str = eval_num if eval_num == "native" else int(eval_num)
    label = safe_checkpoint_label(checkpoint, checkpoint_label)
    request = _model(
        SimulatorLaunchRequest,
        task=task,
        policy_name=resolved_name,
        host=policy_host,
        port=policy_port,
        env_config=env_config,
        scene_config=scene_config,
        env_gpu=env_gpu,
        seed=seed,
        eval_num=parsed_eval_num,
        additional_info=f"ckpt_name={label},action_type={action_type}",
        dry_run=dry_run,
    )
    return run_simulator_session(_paths(root), request)


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
    env_gpu: int = typer.Option(0, "--env-gpu", help="Zero-based GPU index exposed to the simulator."),
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
    if not dry_run:
        try:
            with socket.create_connection((policy_host, policy_port), timeout=connect_timeout):
                pass
        except OSError:
            typer.echo(f"warning: {policy_host}:{policy_port} is not reachable yet; the client will retry", err=True)
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
    policy_gpu: int,
    env_gpu: int,
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
    policy_gpu: int = typer.Option(0, "--policy-gpu", help="Zero-based GPU index exposed to each policy server."),
    env_gpu: int = typer.Option(0, "--env-gpu", help="Zero-based GPU index exposed to each simulator."),
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
    policy_gpu: int = typer.Option(0, "--policy-gpu", help="Zero-based GPU index exposed to each policy server."),
    env_gpu: int = typer.Option(0, "--env-gpu", help="Zero-based GPU index exposed to each simulator."),
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


@app.command()
def summarize(
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Markdown destination; defaults to runs/reports/_summary.md in local storage.",
    ),
    env_config: str | None = typer.Option(
        None,
        "--env-cfg",
        help="Include only results recorded with this environment profile.",
    ),
    scene_config: str | None = typer.Option(
        None,
        "--scene",
        help="Include only results recorded with this scene configuration.",
    ),
) -> None:
    """Aggregate evaluation results into Markdown."""
    from robodojo.workflows.results_summary import main

    args: list[str] = []
    if output:
        args += ["--output", str(output)]
    if env_config:
        args += ["--env-cfg", env_config]
    if scene_config:
        args += ["--scene", scene_config]
    main(args)


@assets_app.command("download")
def assets_download(
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Repository checkout used to resolve settings; auto-detected when omitted.",
    ),
    revision: str = typer.Option(
        "main",
        "--revision",
        help="Git revision of the RoboDojo dataset repository to download.",
    ),
) -> None:
    """Download the benchmark asset bundle into canonical local storage."""
    from robodojo.workflows.downloads import download_assets

    download_assets(_paths(root), revision)


@assets_app.command("build-openarm")
def assets_build_openarm(
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Repository checkout containing the pinned OpenArm source manifest.",
    ),
) -> None:
    """Build the pinned OpenArm robot assets into canonical local storage."""
    from robodojo.workflows.assets import build_openarm

    raise typer.Exit(build_openarm(_paths(root)))


@assets_app.command("build-yam")
def assets_build_yam(
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Repository checkout containing the pinned YAM source manifest.",
    ),
) -> None:
    """Build the pinned I2RT YAM robot assets into canonical local storage."""
    from robodojo.workflows.assets import build_yam

    raise typer.Exit(build_yam(_paths(root)))


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
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Repository checkout used to resolve settings; auto-detected when omitted.",
    ),
    revision: str = typer.Option(
        "main",
        "--revision",
        help="Git revision of the RoboDojo dataset repository to download.",
    ),
) -> None:
    """Download one benchmark dataset into canonical local storage."""
    from robodojo.workflows.downloads import download_data

    download_data(_paths(root), data_format, revision)


@storage_app.command("status")
@storage_app.command("doctor")
def storage_doctor() -> None:
    """Check local storage writes and optional S3 access."""
    from robodojo.workflows.storage import doctor

    doctor()


@storage_app.command()
def publish(
    source: Path = typer.Argument(..., help="Local directory whose files should be published."),
    relative: str = typer.Argument(
        ...,
        help="Destination below ROBODOJO_S3_URI, beginning with assets, datasets, model_weights, or runs.",
    ),
    replace: bool = typer.Option(
        False,
        "--replace",
        help="Replace an already completed remote payload instead of preserving its immutability.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the resolved source and S3 destination without uploading files.",
    ),
) -> None:
    """Publish a local directory to an explicit canonical S3 destination."""
    from robodojo.workflows.storage import publish as publish_payload

    publish_payload(source, relative, replace=replace, dry_run=dry_run)


def _storage_passthrough(arguments: list[str]) -> None:
    from robodojo.workflows.storage import main

    raise typer.Exit(main(arguments))


def _publish_arguments(command: str, values: list[str], replace: bool, dry_run: bool) -> list[str]:
    arguments = [command, *values]
    if replace:
        arguments.append("--replace")
    if dry_run:
        arguments.append("--dry-run")
    return arguments


@storage_app.command()
def pull(
    relative: str = typer.Argument(
        ...,
        help="Completed payload below ROBODOJO_S3_URI to restore into canonical local storage.",
    ),
    replace: bool = typer.Option(
        False,
        "--replace",
        help="Replace an existing local payload after the remote copy passes verification.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the resolved S3 source and local destination without downloading files.",
    ),
) -> None:
    """Download and verify one completed S3 payload."""
    args = ["pull", relative]
    if replace:
        args.append("--replace")
    if dry_run:
        args.append("--dry-run")
    _storage_passthrough(args)


@storage_app.command("publish-assets")
def storage_publish_assets(
    source: Path = typer.Argument(..., help="Local benchmark asset directory to publish."),
    replace: bool = typer.Option(
        False,
        "--replace",
        help="Replace the completed remote assets payload if it already exists.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the canonical assets destination without uploading files.",
    ),
) -> None:
    """Publish the canonical benchmark asset payload."""
    _storage_passthrough(_publish_arguments("publish-assets", [str(source)], replace, dry_run))


@storage_app.command("publish-data")
def storage_publish_data(
    dataset: str = typer.Argument(..., help="Dataset directory name used below the remote datasets prefix."),
    source: Path = typer.Argument(..., help="Local dataset directory to publish."),
    replace: bool = typer.Option(
        False,
        "--replace",
        help="Replace the completed remote dataset payload if it already exists.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the canonical dataset destination without uploading files.",
    ),
) -> None:
    """Publish one named dataset payload."""
    _storage_passthrough(_publish_arguments("publish-data", [dataset, str(source)], replace, dry_run))


@storage_app.command("publish-checkpoint")
def storage_publish_checkpoint(
    policy: str = typer.Argument(..., help="Policy name used below the remote model_weights prefix."),
    checkpoint: str = typer.Argument(..., help="Checkpoint name used as the remote payload directory."),
    source: Path = typer.Argument(..., help="Local checkpoint directory to publish."),
    replace: bool = typer.Option(
        False,
        "--replace",
        help="Replace the completed remote checkpoint payload if it already exists.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the canonical checkpoint destination without uploading files.",
    ),
) -> None:
    """Publish one policy checkpoint payload."""
    _storage_passthrough(_publish_arguments("publish-checkpoint", [policy, checkpoint, str(source)], replace, dry_run))


@storage_app.command("publish-model")
def storage_publish_model(
    policy: str = typer.Argument(..., help="Policy name used below the remote model_weights prefix."),
    model: str = typer.Argument(..., help="Model name used as the remote payload directory."),
    source: Path = typer.Argument(..., help="Local model directory to publish."),
    replace: bool = typer.Option(
        False,
        "--replace",
        help="Replace the completed remote model payload if it already exists.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the canonical model destination without uploading files.",
    ),
) -> None:
    """Publish one named policy model payload."""
    _storage_passthrough(_publish_arguments("publish-model", [policy, model, str(source)], replace, dry_run))


@storage_app.command("publish-reference-cache")
def storage_publish_reference_cache(
    name: str = typer.Argument(..., help="Reference-cache name used below the remote datasets prefix."),
    revision: str = typer.Argument(..., help="Source revision used to version the remote cache payload."),
    source: Path = typer.Argument(..., help="Local reference-cache directory to publish."),
    replace: bool = typer.Option(
        False,
        "--replace",
        help="Replace the completed remote cache payload if it already exists.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the canonical cache destination without uploading files.",
    ),
) -> None:
    """Publish a versioned reference-data cache."""
    _storage_passthrough(_publish_arguments("publish-reference-cache", [name, revision, str(source)], replace, dry_run))


@storage_app.command("publish-eval")
def storage_publish_eval(
    source: Path = typer.Option(
        Path("."),
        "--source",
        help="Completed evaluation directory, used directly when --run-id is omitted.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Timestamped run name to find uniquely below canonical local evaluation storage.",
    ),
    replace: bool = typer.Option(
        False,
        "--replace",
        help="Replace the completed remote evaluation payload if it already exists.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the canonical evaluation destination without uploading files.",
    ),
) -> None:
    """Publish one completed evaluation result directory."""
    values = [str(source)]
    if run_id:
        values += ["--run-id", run_id]
    _storage_passthrough(_publish_arguments("publish-eval", values, replace, dry_run))


@storage_app.command("publish-run")
def storage_publish_run(
    kind: str = typer.Argument(..., help="Run category used directly below the remote runs prefix."),
    run_id: str = typer.Argument(..., help="Stable run identifier used as the remote payload directory."),
    source: Path = typer.Argument(..., help="Local run directory to publish."),
    replace: bool = typer.Option(
        False,
        "--replace",
        help="Replace the completed remote run payload if it already exists.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the canonical run destination without uploading files.",
    ),
) -> None:
    """Publish a named run payload under a caller-selected category."""
    _storage_passthrough(_publish_arguments("publish-run", [kind, run_id, str(source)], replace, dry_run))


@results_app.command("stats")
def results_stats(
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Evaluation-result directory; defaults to canonical local evaluation storage.",
    ),
    policies: list[str] | None = typer.Option(
        None,
        "--policy",
        help="Policy to include; repeat the option to compare multiple policies.",
    ),
    tasks: list[str] | None = typer.Option(
        None,
        "--task",
        help="Task to include; repeat the option to select multiple tasks.",
    ),
    env_config: str | None = typer.Option(
        None,
        "--env-cfg",
        help="Include only results recorded with this environment profile.",
    ),
    scene_config: str | None = typer.Option(
        None,
        "--scene",
        help="Include only results recorded with this scene configuration.",
    ),
    per_seed: bool = typer.Option(
        False,
        "--per-seed",
        help="Break score counts down by evaluation seed in addition to policy and task.",
    ),
    json_out: Path | None = typer.Option(
        None,
        "--json-out",
        help="Write the complete aggregated distribution to this JSON file.",
    ),
) -> None:
    """Count evaluation scores by policy and task."""
    from robodojo.workflows.results_stats import main

    args: list[str] = []
    if root:
        args += ["--root", str(root)]
    if policies:
        args += ["--policies", *policies]
    for task in tasks or []:
        args += ["--task", task]
    if env_config:
        args += ["--env-cfg", env_config]
    if scene_config:
        args += ["--scene", scene_config]
    if per_seed:
        args.append("--per-seed")
    if json_out:
        args += ["--json-out", str(json_out)]
    raise typer.Exit(main(args))


@docker_app.command("install")
def docker_install(
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Repository checkout from which installation commands run; auto-detected when omitted.",
    ),
) -> None:
    """Install Docker and the NVIDIA container runtime when missing."""
    from robodojo.workflows.docker import install

    raise typer.Exit(install(_paths(root)))


@docker_app.command("build")
def docker_build(
    image: str = typer.Option(
        "robodojo:cuda12.8",
        "--image",
        help="Repository and tag to assign to the built simulator image.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Repository checkout used as the Docker build context; auto-detected when omitted.",
    ),
) -> None:
    """Build the RoboDojo simulator container image."""
    from robodojo.workflows.docker import build

    raise typer.Exit(build(_paths(root), image))


@docker_app.command("smoke")
def docker_smoke(
    port: int = typer.Option(
        ...,
        "--policy-port",
        help="Host TCP port of the already running external policy server.",
    ),
    image: str = typer.Option(
        "robodojo:cuda12.8",
        "--image",
        help="Simulator image to run for the one-episode container check.",
    ),
    task: str = typer.Option(
        "stack_bowls",
        "--task",
        help="Canonical task name to evaluate inside the container.",
    ),
    policy: str = typer.Option(
        "demo_policy",
        "--policy",
        help="XPolicyLab policy name expected by the simulator client.",
    ),
    env_config: str = typer.Option(
        "arx_x5",
        "--env-cfg",
        help="Environment profile selecting the robot, cameras, simulator, and default scene.",
    ),
    scene_config: str | None = typer.Option(
        None,
        "--scene",
        help="Scene configuration override for the container evaluation.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Repository checkout whose storage and settings are mounted into the container.",
    ),
) -> None:
    """Run a one-episode GPU and policy-connectivity check in Docker."""
    from robodojo.workflows.docker import smoke

    raise typer.Exit(smoke(_paths(root), image, task, policy, port, env_config, scene_config))


@docker_app.command("monitor")
def docker_monitor(
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Repository checkout containing docker/smoke_logs; auto-detected when omitted.",
    ),
) -> None:
    """Follow the newest Docker smoke log."""
    from robodojo.workflows.docker import monitor

    raise typer.Exit(monitor(_paths(root)))


@docker_app.command("clean")
def docker_clean(
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Repository checkout whose Docker smoke state should be cleaned.",
    ),
) -> None:
    """Stop and remove RoboDojo Docker smoke-test state."""
    from robodojo.workflows.docker import clean

    raise typer.Exit(clean(_paths(root)))


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
