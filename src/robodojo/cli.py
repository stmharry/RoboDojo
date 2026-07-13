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
upstream_app = typer.Typer(no_args_is_help=True, help="Review official upstream changes.")
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
    root: Path | None = typer.Option(None, "--root"),
    from_step: str = typer.Option("system", "--from", help="system, submodules, or sync"),
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
    task: str = typer.Option("stack_bowls", "--task"),
    env_config: str = typer.Option("arx_x5", "--env-cfg"),
    policy_dir: Path | None = typer.Option(None, "--policy-dir"),
    skip_policy: bool = typer.Option(False, "--skip-policy"),
    root: Path | None = typer.Option(None, "--root"),
) -> None:
    """Validate the checkout, configuration, assets, and optional policy adapter."""
    from robodojo.workflows.doctor import run_doctor

    code = run_doctor(_paths(root), task, env_config, None if skip_policy else policy_dir)
    raise typer.Exit(code)


@app.command()
def tasks(
    format: str = typer.Option("plain", "--format", help="plain, json, or markdown"),
    only_runnable: bool = typer.Option(False, "--only-runnable"),
    check: bool = typer.Option(False, "--check"),
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
    project: UpstreamProject = typer.Option(UpstreamProject.ALL, "--project"),
    ref: str | None = typer.Option(None, "--ref"),
    source: Path | None = typer.Option(None, "--source"),
    format: UpstreamOutputFormat = typer.Option(UpstreamOutputFormat.PLAIN, "--format"),
    root: Path | None = typer.Option(None, "--root"),
) -> None:
    """Detect new official changes and map them to local ownership."""
    from robodojo.workflows.upstream import check_upstreams, format_upstream_report, json_upstream_report

    report, code = check_upstreams(_paths(root), project=project, ref=ref, source=source)
    typer.echo(json_upstream_report(report) if format == UpstreamOutputFormat.JSON else format_upstream_report(report))
    raise typer.Exit(code)


def _evaluation_request(
    *,
    policy_dir: Path,
    task: str,
    checkpoint: str,
    policy_env: str,
    dataset: str,
    env_config: str,
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
        dry_run=dry_run,
    )


@app.command("eval")
def evaluate(
    policy_dir: Path = typer.Option(..., "--policy-dir"),
    task: str = typer.Option(..., "--task"),
    checkpoint: str = typer.Option(..., "--ckpt"),
    policy_env: str = typer.Option(..., "--policy-env"),
    dataset: str = typer.Option("RoboDojo", "--dataset"),
    env_config: str = typer.Option("arx_x5", "--env-cfg"),
    expert_num: int = typer.Option(100, "--expert-num"),
    action_type: str = typer.Option("ee", "--action-type"),
    seed: int = typer.Option(0, "--seed"),
    policy_gpu: int = typer.Option(0, "--policy-gpu"),
    env_gpu: int = typer.Option(0, "--env-gpu"),
    eval_num: str | None = typer.Option(None, "--eval-num"),
    checkpoint_label: str | None = typer.Option(None, "--ckpt-label"),
    export_scene: bool = typer.Option(False, "--export-scene"),
    export_scene_only: bool = typer.Option(False, "--export-scene-only"),
    export_scene_dir: Path | None = typer.Option(None, "--export-scene-dir"),
    layout_id: int = typer.Option(0, "--layout-id"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    root: Path | None = typer.Option(None, "--root"),
) -> None:
    """Run a local policy server and simulator evaluation."""
    request = _evaluation_request(**{key: value for key, value in locals().items() if key != "root"})
    from robodojo.orchestration.evaluation import run_evaluation

    raise typer.Exit(run_evaluation(_paths(root), request))


@app.command()
def server(
    policy_dir: Path = typer.Option(..., "--policy-dir"),
    task: str = typer.Option(..., "--task"),
    checkpoint: str = typer.Option(..., "--ckpt"),
    policy_env: str = typer.Option(..., "--policy-env"),
    dataset: str = typer.Option("RoboDojo", "--dataset"),
    env_config: str = typer.Option("arx_x5", "--env-cfg"),
    action_type: str = typer.Option("ee", "--action-type"),
    seed: int = typer.Option(0, "--seed"),
    policy_gpu: int = typer.Option(0, "--policy-gpu"),
    policy_port: int | None = typer.Option(None, "--policy-port"),
    bind_host: str = typer.Option("0.0.0.0", "--bind-host"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    root: Path | None = typer.Option(None, "--root"),
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
    raise typer.Exit(run_policy_server(_paths(root), request))


def _client(
    *,
    root: Path | None,
    task: str,
    policy_name: str | None,
    policy_dir: Path | None,
    policy_host: str,
    policy_port: int,
    env_config: str,
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
        env_gpu=env_gpu,
        seed=seed,
        eval_num=parsed_eval_num,
        additional_info=f"ckpt_name={label},action_type={action_type}",
        dry_run=dry_run,
    )
    return run_simulator_session(_paths(root), request)


@app.command()
def client(
    task: str = typer.Option(..., "--task"),
    policy_name: str | None = typer.Option(None, "--policy-name"),
    policy_dir: Path | None = typer.Option(None, "--policy-dir"),
    policy_host: str = typer.Option("127.0.0.1", "--policy-host"),
    policy_port: int = typer.Option(..., "--policy-port"),
    env_config: str = typer.Option("arx_x5", "--env-cfg"),
    env_gpu: int = typer.Option(0, "--env-gpu"),
    seed: int = typer.Option(0, "--seed"),
    eval_num: str = typer.Option("1", "--eval-num"),
    checkpoint: str = typer.Option("external", "--ckpt"),
    checkpoint_label: str | None = typer.Option(None, "--ckpt-label"),
    action_type: str = typer.Option("ee", "--action-type"),
    connect_timeout: float = typer.Option(5, "--connect-timeout"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    root: Path | None = typer.Option(None, "--root"),
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
    policy_dir: Path = typer.Option(..., "--policy-dir"),
    checkpoint: str = typer.Option(..., "--ckpt"),
    policy_env: str = typer.Option(..., "--policy-env"),
    env_config: str = typer.Option("arx_x5", "--env-cfg"),
    action_type: str = typer.Option("ee", "--action-type"),
    seed: int = typer.Option(0, "--seed"),
    policy_gpu: int = typer.Option(0, "--policy-gpu"),
    env_gpu: int = typer.Option(0, "--env-gpu"),
    eval_num: str = typer.Option("1", "--eval-num"),
    only: str | None = typer.Option(None, "--only"),
    tasks_file: Path | None = typer.Option(None, "--tasks-file"),
    limit: int | None = typer.Option(None, "--limit"),
    resume: bool = typer.Option(False, "--resume"),
    fail_fast: bool = typer.Option(False, "--fail-fast"),
    run_id: str | None = typer.Option(None, "--run-id"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    root: Path | None = typer.Option(None, "--root"),
) -> None:
    """Run a one-episode sequential task sweep."""
    _sweep_options("smoke", **locals())


@app.command()
def benchmark(
    policy_dir: Path = typer.Option(..., "--policy-dir"),
    checkpoint: str = typer.Option(..., "--ckpt"),
    policy_env: str = typer.Option(..., "--policy-env"),
    eval_num: str = typer.Option(..., "--eval-num"),
    env_config: str = typer.Option("arx_x5", "--env-cfg"),
    action_type: str = typer.Option("ee", "--action-type"),
    seed: int = typer.Option(0, "--seed"),
    policy_gpu: int = typer.Option(0, "--policy-gpu"),
    env_gpu: int = typer.Option(0, "--env-gpu"),
    only: str | None = typer.Option(None, "--only"),
    tasks_file: Path | None = typer.Option(None, "--tasks-file"),
    limit: int | None = typer.Option(None, "--limit"),
    resume: bool = typer.Option(False, "--resume"),
    fail_fast: bool = typer.Option(False, "--fail-fast"),
    run_id: str | None = typer.Option(None, "--run-id"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    root: Path | None = typer.Option(None, "--root"),
) -> None:
    """Run a sequential benchmark sweep."""
    _sweep_options("benchmark", **locals())


@app.command()
def summarize(output: Path | None = typer.Option(None, "--output")) -> None:
    """Aggregate evaluation results into Markdown."""
    from robodojo.workflows.results_summary import main

    main(["--output", str(output)] if output else [])


@assets_app.command("download")
def assets_download(root: Path | None = typer.Option(None, "--root"), revision: str = typer.Option("main")) -> None:
    from robodojo.workflows.downloads import download_assets

    download_assets(_paths(root), revision)


@assets_app.command("build-openarm")
def assets_build_openarm(root: Path | None = typer.Option(None, "--root")) -> None:
    from robodojo.workflows.assets import build_openarm

    raise typer.Exit(build_openarm(_paths(root)))


@data_app.command("list")
def data_list() -> None:
    from robodojo.workflows.downloads import list_data

    list_data()


@data_app.command("download")
def data_download(
    data_format: DataFormat,
    root: Path | None = typer.Option(None, "--root"),
    revision: str = "main",
) -> None:
    from robodojo.workflows.downloads import download_data

    download_data(_paths(root), data_format, revision)


@storage_app.command("status")
@storage_app.command("doctor")
def storage_doctor() -> None:
    from robodojo.workflows.storage import doctor

    doctor()


@storage_app.command()
def publish(source: Path, relative: str, replace: bool = False, dry_run: bool = False) -> None:
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
def pull(relative: str, replace: bool = False, dry_run: bool = False) -> None:
    args = ["pull", relative]
    if replace:
        args.append("--replace")
    if dry_run:
        args.append("--dry-run")
    _storage_passthrough(args)


@storage_app.command("publish-assets")
def storage_publish_assets(source: Path, replace: bool = False, dry_run: bool = False) -> None:
    _storage_passthrough(_publish_arguments("publish-assets", [str(source)], replace, dry_run))


@storage_app.command("publish-data")
def storage_publish_data(dataset: str, source: Path, replace: bool = False, dry_run: bool = False) -> None:
    _storage_passthrough(_publish_arguments("publish-data", [dataset, str(source)], replace, dry_run))


@storage_app.command("publish-checkpoint")
def storage_publish_checkpoint(
    policy: str,
    checkpoint: str,
    source: Path,
    replace: bool = False,
    dry_run: bool = False,
) -> None:
    _storage_passthrough(_publish_arguments("publish-checkpoint", [policy, checkpoint, str(source)], replace, dry_run))


@storage_app.command("publish-model")
def storage_publish_model(
    policy: str,
    model: str,
    source: Path,
    replace: bool = False,
    dry_run: bool = False,
) -> None:
    _storage_passthrough(_publish_arguments("publish-model", [policy, model, str(source)], replace, dry_run))


@storage_app.command("publish-reference-cache")
def storage_publish_reference_cache(
    name: str,
    revision: str,
    source: Path,
    replace: bool = False,
    dry_run: bool = False,
) -> None:
    _storage_passthrough(_publish_arguments("publish-reference-cache", [name, revision, str(source)], replace, dry_run))


@storage_app.command("publish-eval")
def storage_publish_eval(
    source: Path = Path("."),
    run_id: str | None = typer.Option(None, "--run-id"),
    replace: bool = False,
    dry_run: bool = False,
) -> None:
    values = [str(source)]
    if run_id:
        values += ["--run-id", run_id]
    _storage_passthrough(_publish_arguments("publish-eval", values, replace, dry_run))


@storage_app.command("publish-run")
def storage_publish_run(
    kind: str,
    run_id: str,
    source: Path,
    replace: bool = False,
    dry_run: bool = False,
) -> None:
    _storage_passthrough(_publish_arguments("publish-run", [kind, run_id, str(source)], replace, dry_run))


@results_app.command("stats")
def results_stats(
    root: Path | None = typer.Option(None, "--root"),
    policies: list[str] | None = typer.Option(None, "--policy"),
    tasks: list[str] | None = typer.Option(None, "--task"),
    per_seed: bool = typer.Option(False, "--per-seed"),
    json_out: Path | None = typer.Option(None, "--json-out"),
) -> None:
    from robodojo.workflows.results_stats import main

    args: list[str] = []
    if root:
        args += ["--root", str(root)]
    if policies:
        args += ["--policies", *policies]
    for task in tasks or []:
        args += ["--task", task]
    if per_seed:
        args.append("--per-seed")
    if json_out:
        args += ["--json-out", str(json_out)]
    raise typer.Exit(main(args))


@docker_app.command("install")
def docker_install(root: Path | None = typer.Option(None, "--root")) -> None:
    from robodojo.workflows.docker import install

    raise typer.Exit(install(_paths(root)))


@docker_app.command("build")
def docker_build(
    image: str = typer.Option("robodojo:cuda12.8"),
    root: Path | None = typer.Option(None, "--root"),
) -> None:
    from robodojo.workflows.docker import build

    raise typer.Exit(build(_paths(root), image))


@docker_app.command("smoke")
def docker_smoke(
    port: int = typer.Option(..., "--policy-port"),
    image: str = typer.Option("robodojo:cuda12.8"),
    task: str = typer.Option("stack_bowls"),
    policy: str = typer.Option("demo_policy"),
    env_config: str = typer.Option("arx_x5", "--env-cfg"),
    root: Path | None = typer.Option(None, "--root"),
) -> None:
    from robodojo.workflows.docker import smoke

    raise typer.Exit(smoke(_paths(root), image, task, policy, port, env_config))


@docker_app.command("monitor")
def docker_monitor(root: Path | None = typer.Option(None, "--root")) -> None:
    from robodojo.workflows.docker import monitor

    raise typer.Exit(monitor(_paths(root)))


@docker_app.command("clean")
def docker_clean(root: Path | None = typer.Option(None, "--root")) -> None:
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
