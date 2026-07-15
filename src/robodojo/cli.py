"""Unified Typer command-line interface for RoboDojo."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import typer

from robodojo.commands.assets import assets_app, data_app
from robodojo.commands.common import (
    contract_values as _contract_values,
    model as _model,
    paths as _paths,
    report_format as _report_format,
)
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
    recipe: str | None = typer.Option(None, "--recipe", help="Tracked evaluation recipe."),
    policy_profile: str | None = typer.Option(None, "--policy-profile", help="Manual policy profile."),
    environment: str | None = typer.Option(None, "--environment", help="Manual environment profile."),
    scene: str | None = typer.Option(None, "--scene", help="Manual scene profile."),
    protocol: str | None = typer.Option(None, "--protocol", help="Manual task protocol."),
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

    repository = _paths(root)
    contract = _contract_values(
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
def recipes(
    format: str = typer.Option("plain", "--format", help="Output format: table, plain, or json."),
    check: bool = typer.Option(False, "--check", help="Validate every policy/protocol/recipe reference."),
    root: Path | None = typer.Option(None, "--root", help="Repository checkout to inspect."),
) -> None:
    """List and validate explicit evaluation recipes."""
    import json

    from robodojo.core.contracts import recipe_rows

    repository = _paths(root)
    try:
        rows = recipe_rows(repository)
    except (OSError, RuntimeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1 if check else 2) from exc
    if format == "json":
        typer.echo(json.dumps(rows, indent=2, sort_keys=True))
    elif format == "plain":
        for row in rows:
            typer.echo(
                f"{row['recipe']}\t{row['policy']}\t{row['environment']}\t"
                f"{row['scene']}\t{row['protocol']}\t{row['task']}"
            )
    elif format == "table":
        from robodojo.workflows.recipe_inventory import print_recipe_table

        print_recipe_table(rows)
    else:
        raise typer.BadParameter("expected table, plain, or json", param_hint="--format")


@app.command()
def preflight(
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

    repository = _paths(root)
    contract = _contract_values(
        repository,
        recipe=recipe,
        policy_profile=policy_profile,
        environment=environment,
        scene=scene,
        protocol=protocol,
    )
    request = _model(
        PreflightRequest,
        **contract,
        seed=seed,
        policy_gpu=policy_gpu,
        env_gpu=env_gpu,
        publish=publish,
        deep=deep,
        timeout=timeout,
    )
    raise typer.Exit(run_preflight(repository, request, output_format=_report_format(output_format)))


def _evaluation_request(
    *,
    contract: dict[str, Any],
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
        **contract,
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
    recipe: str | None = typer.Option(None, "--recipe", help="Tracked evaluation recipe."),
    policy_profile: str | None = typer.Option(None, "--policy-profile", help="Manual policy profile."),
    environment: str | None = typer.Option(None, "--environment", help="Manual environment profile."),
    scene: str | None = typer.Option(None, "--scene", help="Manual scene profile."),
    protocol: str | None = typer.Option(None, "--protocol", help="Manual task protocol."),
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
        "native",
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
    repository = _paths(root)
    contract = _contract_values(
        repository,
        recipe=recipe,
        policy_profile=policy_profile,
        environment=environment,
        scene=scene,
        protocol=protocol,
    )
    request = _evaluation_request(
        contract=contract,
        seed=seed,
        policy_gpu=policy_gpu,
        env_gpu=env_gpu,
        eval_num=eval_num,
        checkpoint_label=checkpoint_label,
        export_scene=export_scene,
        export_scene_only=export_scene_only,
        export_scene_dir=export_scene_dir,
        layout_id=layout_id,
        publish=publish,
        dry_run=dry_run,
    )
    from robodojo.orchestration.evaluation import run_evaluation

    raise typer.Exit(run_evaluation(repository, request))


@app.command()
def server(
    recipe: str | None = typer.Option(None, "--recipe", help="Tracked evaluation recipe."),
    policy_profile: str | None = typer.Option(None, "--policy-profile", help="Manual policy profile."),
    environment: str | None = typer.Option(None, "--environment", help="Manual environment profile."),
    scene: str | None = typer.Option(None, "--scene", help="Manual scene profile."),
    protocol: str | None = typer.Option(None, "--protocol", help="Manual task protocol."),
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

    repository = _paths(root)
    contract = _contract_values(
        repository,
        recipe=recipe,
        policy_profile=policy_profile,
        environment=environment,
        scene=scene,
        protocol=protocol,
    )
    request = _model(
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


def _client(
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
    connect_timeout: float | None = None,
) -> int:
    from robodojo.core.storage import checkpoint_label as safe_checkpoint_label
    from robodojo.orchestration.split import run_client

    repository = _paths(root)
    contract = _contract_values(
        repository,
        recipe=recipe,
        policy_profile=policy_profile,
        environment=environment,
        scene=scene,
        protocol=protocol,
    )
    resolved_name = Path(contract["policy_dir"]).name
    parsed_eval_num: int | str = eval_num if eval_num == "native" else int(eval_num)
    label = safe_checkpoint_label(contract["checkpoint"], checkpoint_label)
    from robodojo.core.gpu import GpuSelectionError, parse_gpu_selector, resolve_gpus

    try:
        assignment = resolve_gpus(env_gpu=parse_gpu_selector(env_gpu))
    except GpuSelectionError as exc:
        typer.echo(f"GPU selection failed: {exc}", err=True)
        return 2
    request = _model(
        SimulatorLaunchRequest,
        task=contract["task"],
        protocol_name=contract["protocol"],
        episode_horizon=contract["episode_horizon"],
        native_eval_num=contract["native_eval_num"],
        recipe=contract["recipe"],
        contract_hash=contract["contract_hash"],
        policy_name=resolved_name,
        host=policy_host,
        port=policy_port,
        env_config=contract["env_config"],
        scene_config=contract["scene_config"],
        env_gpu=assignment.env_gpu,
        seed=seed,
        eval_num=parsed_eval_num,
        additional_info=f"ckpt_name={label},action_type={contract['action_type']}",
        dry_run=dry_run,
    )
    return run_client(repository, request, connect_timeout=connect_timeout if connect_timeout is not None else 5)


@app.command()
def client(
    recipe: str | None = typer.Option(None, "--recipe", help="Tracked evaluation recipe."),
    policy_profile: str | None = typer.Option(None, "--policy-profile", help="Manual policy profile."),
    environment: str | None = typer.Option(None, "--environment", help="Manual environment profile."),
    scene: str | None = typer.Option(None, "--scene", help="Manual scene profile."),
    protocol: str | None = typer.Option(None, "--protocol", help="Manual task protocol."),
    policy_host: str = typer.Option(
        "127.0.0.1",
        "--policy-host",
        help="Hostname or IP address of the external policy server.",
    ),
    policy_port: int = typer.Option(..., "--policy-port", help="TCP port of the external policy WebSocket server."),
    env_gpu: str = typer.Option(
        "auto",
        "--env-gpu",
        envvar="ENV_GPU",
        help="Simulator GPU as a zero-based index or auto; ENV_GPU is used when the flag is omitted.",
    ),
    seed: int = typer.Option(0, "--seed", help="Nonnegative evaluation seed used for task layout."),
    eval_num: str = typer.Option(
        "native",
        "--eval-num",
        help="Episode count as a positive integer, or native to keep the simulator config value.",
    ),
    checkpoint_label: str | None = typer.Option(
        None,
        "--ckpt-label",
        help="Filesystem-safe result label; defaults to the checkpoint name or path basename.",
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
    recipe: list[str],
    seed: int,
    policy_gpu: str,
    env_gpu: str,
    eval_num: str,
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
        recipes=tuple(recipe),
        seed=seed,
        policy_gpu=policy_gpu,
        env_gpu=env_gpu,
        eval_num=parsed_eval_num,
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
    recipe: list[str] = typer.Option(..., "--recipe", help="Recipe to sweep; repeat for multiple recipes."),
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
    """Run a one-episode sequential recipe sweep."""
    _sweep_options("smoke", **locals())


@app.command()
def benchmark(
    recipe: list[str] = typer.Option(..., "--recipe", help="Recipe to sweep; repeat for multiple recipes."),
    eval_num: str = typer.Option(
        ...,
        "--eval-num",
        help="Episodes per task as a positive integer, or native for each simulator config.",
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

    request = _model(
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
    raise typer.Exit(run_simulator_session(_paths(root_dir), request))


if __name__ == "__main__":
    app()
