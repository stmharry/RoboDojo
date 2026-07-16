"""Container support command group."""

from __future__ import annotations

import typer

from robodojo.commands.common import paths
from robodojo.commands.options import DockerImageOption, RepositoryRootOption

docker_app = typer.Typer(no_args_is_help=True, help="Build and validate RoboDojo containers.")


@docker_app.command("install")
def install_command(
    root: RepositoryRootOption = None,
) -> None:
    """Install Docker and the NVIDIA container runtime when missing."""
    from robodojo.workflows.docker import install

    raise typer.Exit(install(paths(root)))


@docker_app.command("build")
def build_command(
    image: DockerImageOption = "robodojo:cuda12.8",
    root: RepositoryRootOption = None,
) -> None:
    """Build the RoboDojo simulator container image."""
    from robodojo.workflows.docker import build

    raise typer.Exit(build(paths(root), image))


@docker_app.command("smoke")
def smoke_command(
    port: int = typer.Option(..., "--policy-port", help="Host TCP port of the external policy server."),
    image: DockerImageOption = "robodojo:cuda12.8",
    task_protocol: str = typer.Option("stack_bowls", "--task-protocol", help="Task protocol to evaluate."),
    policy: str = typer.Option("pi05_arx_x5", "--policy", help="Tracked policy profile expected by the client."),
    environment: str = typer.Option("arx_x5", "--environment", help="Environment profile to evaluate."),
    scene: str | None = typer.Option(None, "--scene", help="Optional scene profile override."),
    root: RepositoryRootOption = None,
) -> None:
    """Run a one-episode GPU and policy-connectivity check in Docker."""
    from robodojo.workflows.docker import smoke

    raise typer.Exit(smoke(paths(root), image, task_protocol, policy, port, environment, scene))


@docker_app.command("monitor")
def monitor_command(
    root: RepositoryRootOption = None,
) -> None:
    """Follow the newest Docker smoke log."""
    from robodojo.workflows.docker import monitor

    raise typer.Exit(monitor(paths(root)))


@docker_app.command("clean")
def clean_command(
    root: RepositoryRootOption = None,
) -> None:
    """Stop and remove RoboDojo Docker smoke-test state."""
    from robodojo.workflows.docker import clean

    raise typer.Exit(clean(paths(root)))
