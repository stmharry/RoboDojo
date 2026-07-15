"""Container support command group."""

from __future__ import annotations

from pathlib import Path

import typer

from robodojo.commands.common import paths

docker_app = typer.Typer(no_args_is_help=True, help="Build and validate RoboDojo containers.")


@docker_app.command("install")
def install_command(
    root: Path | None = typer.Option(None, "--root", help="Repository checkout to use; auto-detected when omitted."),
) -> None:
    """Install Docker and the NVIDIA container runtime when missing."""
    from robodojo.workflows.docker import install

    raise typer.Exit(install(paths(root)))


@docker_app.command("build")
def build_command(
    image: str = typer.Option("robodojo:cuda12.8", "--image", help="Repository and tag for the simulator image."),
    root: Path | None = typer.Option(None, "--root", help="Repository checkout to use; auto-detected when omitted."),
) -> None:
    """Build the RoboDojo simulator container image."""
    from robodojo.workflows.docker import build

    raise typer.Exit(build(paths(root), image))


@docker_app.command("smoke")
def smoke_command(
    port: int = typer.Option(..., "--policy-port", help="Host TCP port of the external policy server."),
    image: str = typer.Option("robodojo:cuda12.8", "--image", help="Simulator image for the container check."),
    task: str = typer.Option("stack_bowls", "--task", help="Canonical task to evaluate."),
    policy: str = typer.Option("demo_policy", "--policy", help="XPolicyLab policy name expected by the client."),
    env_config: str = typer.Option("arx_x5", "--env-cfg", help="Environment profile to evaluate."),
    scene_config: str | None = typer.Option(None, "--scene", help="Optional scene profile override."),
    root: Path | None = typer.Option(None, "--root", help="Repository checkout to use; auto-detected when omitted."),
) -> None:
    """Run a one-episode GPU and policy-connectivity check in Docker."""
    from robodojo.workflows.docker import smoke

    raise typer.Exit(smoke(paths(root), image, task, policy, port, env_config, scene_config))


@docker_app.command("monitor")
def monitor_command(
    root: Path | None = typer.Option(None, "--root", help="Repository checkout to use; auto-detected when omitted."),
) -> None:
    """Follow the newest Docker smoke log."""
    from robodojo.workflows.docker import monitor

    raise typer.Exit(monitor(paths(root)))


@docker_app.command("clean")
def clean_command(
    root: Path | None = typer.Option(None, "--root", help="Repository checkout whose Docker state is cleaned."),
) -> None:
    """Stop and remove RoboDojo Docker smoke-test state."""
    from robodojo.workflows.docker import clean

    raise typer.Exit(clean(paths(root)))
