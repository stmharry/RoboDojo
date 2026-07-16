import os
from pathlib import Path
import shutil
import subprocess
import sys

from pydantic import ValidationError
import pytest
import typer
from typer.main import get_command
from typer.testing import CliRunner

from robodojo.cli import app
from robodojo.core.experiments.selection import resolve_recipe
from robodojo.core.models.experiment import ExperimentSpec
from robodojo.core.models.requests import (
    EvaluationRequest,
    PolicyServerLaunchRequest,
    SimulatorLaunchRequest,
)
from robodojo.core.paths import RepositoryPaths, discover_repository_root
from robodojo.core.settings import RuntimeSettings
from robodojo.policy.adapter import policy_server_command
from robodojo.sim.launcher import load_simulator_config, simulator_command
from robodojo.workflows.task_inventory import build_inventory

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()
RECIPE = "pi05-bimanual_yam-molmo_yam-general_pickup"


def _experiment_values(recipe: str = RECIPE) -> dict:
    paths = RepositoryPaths.resolve(ROOT)
    return {"experiment": resolve_recipe(paths, recipe).spec(paths)}


def _experiment(**updates) -> ExperimentSpec:
    values = {
        "policy_dir": ROOT / "XPolicyLab/policy/Pi_05",
        "task": "stack_bowls",
        "checkpoint": "run-1",
        "policy_profile": "test_policy",
        "policy_runtime": "policy-env",
        "environment": "arx_x5",
        "embodiment": "arx_x5",
        "scene": "default",
        "action_type": "joint",
        "task_protocol": "stack_bowls",
        "episode_horizon": 800,
        "evaluation_episodes": 25,
    }
    values.update(updates)
    return ExperimentSpec(**values)


def _simulator_request(**updates) -> SimulatorLaunchRequest:
    experiment_keys = {
        "task",
        "checkpoint",
        "policy_profile",
        "policy_runtime",
        "environment",
        "embodiment",
        "scene",
        "action_type",
        "task_protocol",
        "episode_horizon",
        "evaluation_episodes",
    }
    experiment_updates = {key: updates.pop(key) for key in tuple(updates) if key in experiment_keys}
    values = {
        "experiment": _experiment(**experiment_updates),
        "policy_name": "TestPolicy",
        "port": 19000,
        "additional_info": "test",
    }
    values.update(updates)
    return SimulatorLaunchRequest(**values)


def test_cli_exposes_the_unified_command_surface():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "setup",
        "preflight",
        "eval",
        "server",
        "client",
        "snapshots",
        "smoke",
        "storage",
        "assets",
        "data",
        "docker",
        "results",
    ):
        assert command in result.stdout
    commands = get_command(app).commands
    assert set(commands) == {
        "setup",
        "doctor",
        "tasks",
        "recipes",
        "snapshots",
        "preflight",
        "eval",
        "server",
        "client",
        "smoke",
        "benchmark",
        "_adapter-client",
        "assets",
        "data",
        "storage",
        "results",
        "docker",
    }
    assert "upstream" not in result.stdout

    assert set(commands["assets"].commands) == {
        "download",
        "build-moonlake-office",
        "build-moonlake-packing",
        "build-yam",
        "build-openarm",
    }
    assert set(commands["data"].commands) == {"list", "download"}
    assert set(commands["storage"].commands) == {"doctor", "publish", "pull", "publish-eval"}
    assert set(commands["results"].commands) == {"summarize", "stats"}
    assert set(commands["docker"].commands) == {"install", "build", "smoke", "monitor", "clean"}


def test_every_public_command_and_parameter_has_human_readable_help():
    root = get_command(app)
    visited: list[tuple[str, ...]] = []

    def visit(command, path: tuple[str, ...]) -> None:
        assert command.help and command.help.strip(), f"missing command help: {' '.join(path) or 'robodojo'}"
        for parameter in command.params:
            if not parameter.hidden:
                assert getattr(parameter, "help", None), (
                    f"missing parameter help: {' '.join(path) or 'robodojo'} {parameter.name}"
                )
        visited.append(path)
        for name, child in getattr(command, "commands", {}).items():
            if not child.hidden:
                visit(child, (*path, name))

    visit(root, ())

    for path in visited:
        result = runner.invoke(app, [*path, "--help"])
        assert result.exit_code == 0, f"help failed for {' '.join(path) or 'robodojo'}: {result.output}"


def test_eval_help_explains_publication_and_evaluation_inputs():
    result = runner.invoke(app, ["eval", "--help"])

    assert result.exit_code == 0
    assert "--publish" in result.stdout
    assert "ROBODOJO_S3_URI" in result.stdout
    assert "--recipe" in result.stdout
    assert "--task-protocol" in result.stdout
    assert "--checkpoint-label" in result.stdout
    assert "--ckpt-label" not in result.stdout
    assert "Episode count" in result.stdout
    assert "native" in result.stdout


def test_clean_break_help_only_lists_new_option_names():
    results = runner.invoke(app, ["results", "stats", "--help"])
    docker = runner.invoke(app, ["docker", "smoke", "--help"])

    assert results.exit_code == 0
    assert "--results-root" in results.stdout
    assert "--environment" in results.stdout
    assert "--env-cfg" not in results.stdout
    assert "--root" not in results.stdout
    assert docker.exit_code == 0
    assert "--environment" in docker.stdout
    assert "--task-protocol" in docker.stdout
    assert "--task                 " not in docker.stdout
    assert "--env-cfg" not in docker.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ["eval", "--ckpt-label", "checkpoint"],
        ["results", "summarize", "--root", "/tmp/results"],
        ["results", "stats", "--env-cfg", "arx_x5"],
        ["docker", "smoke", "--env-cfg", "arx_x5"],
    ],
)
def test_clean_break_rejects_removed_option_names(arguments):
    result = runner.invoke(app, arguments)

    assert result.exit_code == 2
    assert "No such option" in result.output


def test_docker_smoke_uses_the_typed_client_interface(monkeypatch, tmp_path):
    from robodojo.workflows import docker

    calls = []
    monkeypatch.setenv("ROBODOJO_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.delenv("ROBODOJO_S3_URI", raising=False)
    monkeypatch.setattr(
        docker.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)) or subprocess.CompletedProcess(command, 0),
    )

    assert (
        docker.smoke(
            RepositoryPaths.resolve(ROOT),
            "robodojo:test",
            "stack_bowls",
            "pi05_arx_x5",
            19000,
            "arx_x5",
        )
        == 0
    )

    command = calls[0][0]
    assert command[command.index("client") :] == [
        "client",
        "--policy-profile",
        "pi05_arx_x5",
        "--environment",
        "arx_x5",
        "--scene",
        "default",
        "--task-protocol",
        "stack_bowls",
        "--policy-host",
        "127.0.0.1",
        "--policy-port",
        "19000",
        "--eval-num",
        "1",
    ]


@pytest.mark.parametrize("value", ["0", "-1", "all"])
def test_evaluation_count_has_one_validation_rule(value):
    from robodojo.commands.options import parse_evaluation_count

    with pytest.raises(typer.BadParameter, match="positive integer or native"):
        parse_evaluation_count(value)

    assert parse_evaluation_count("native") == "native"
    assert parse_evaluation_count("3") == 3


@pytest.mark.parametrize(
    "arguments",
    [
        ["eval", "--recipe", RECIPE],
        ["client", "--recipe", RECIPE, "--policy-port", "19000"],
        ["smoke", "--recipe", RECIPE],
        ["benchmark", "--recipe", RECIPE],
    ],
)
def test_evaluation_count_validation_is_shared_by_all_consumers(arguments):
    result = runner.invoke(app, [*arguments, "--eval-num", "0", "--root", str(ROOT)])

    assert result.exit_code == 2
    assert "positive integer or native" in result.output


def test_gpu_cli_precedence_is_flag_then_environment_then_auto(monkeypatch):
    from robodojo.orchestration import evaluation

    monkeypatch.delenv("POLICY_GPU", raising=False)
    monkeypatch.delenv("ENV_GPU", raising=False)
    requests = []
    monkeypatch.setattr(evaluation, "run_evaluation", lambda paths, request: requests.append(request) or 0)
    arguments = [
        "eval",
        "--recipe",
        RECIPE,
        "--root",
        str(ROOT),
    ]

    default = runner.invoke(app, arguments)
    environment = runner.invoke(app, arguments, env={"POLICY_GPU": "4", "ENV_GPU": "5"})
    flags = runner.invoke(
        app,
        [*arguments, "--policy-gpu", "2", "--env-gpu", "3"],
        env={"POLICY_GPU": "4", "ENV_GPU": "5"},
    )

    assert [default.exit_code, environment.exit_code, flags.exit_code] == [0, 0, 0]
    assert [(request.policy_gpu, request.environment_gpu) for request in requests] == [
        ("auto", "auto"),
        (4, 5),
        (2, 3),
    ]


def test_gpu_cli_rejects_noncanonical_auto(monkeypatch):
    from robodojo.orchestration import evaluation

    monkeypatch.setattr(evaluation, "run_evaluation", lambda *args: pytest.fail("invalid selector reached workflow"))
    result = runner.invoke(
        app,
        [
            "eval",
            "--recipe",
            RECIPE,
            "--policy-gpu",
            "AUTO",
            "--env-gpu",
            "0",
            "--root",
            str(ROOT),
        ],
    )

    assert result.exit_code == 2
    assert "auto" in result.output


def test_client_resolves_only_the_simulator_gpu(monkeypatch):
    from robodojo.core import gpu
    from robodojo.core.gpu import GpuAssignment
    from robodojo.orchestration import split

    monkeypatch.delenv("ENV_GPU", raising=False)
    selections = []
    launched = []

    def resolve(**selectors):
        selections.append(selectors)
        return GpuAssignment(env_gpu=7, env_source="auto")

    monkeypatch.setattr(gpu, "resolve_gpus", resolve)
    monkeypatch.setattr(
        split,
        "run_client",
        lambda paths, request, *, connect_timeout: launched.append(request) or 0,
    )
    result = runner.invoke(
        app,
        [
            "client",
            "--recipe",
            RECIPE,
            "--policy-port",
            "19000",
            "--root",
            str(ROOT),
        ],
    )

    assert result.exit_code == 0
    assert selections == [{"env_gpu": "auto"}]
    assert launched[0].environment_gpu == 7
    assert launched[0].experiment.policy_profile == "pi05_bimanual_yam"
    assert launched[0].experiment.policy_descriptor_hash
    assert launched[0].experiment.policy_reference_match == "reference_match"


def test_publish_is_incompatible_with_scene_only_export(tmp_path):
    with pytest.raises(ValidationError, match="--publish cannot be combined with --export-scene-only"):
        EvaluationRequest(
            **_experiment_values(),
            publish=True,
            export_scene_only=True,
        )


def test_make_eval_defaults_to_local_info_with_opt_in_overrides(tmp_path):
    makefile = tmp_path / "Makefile"
    shutil.copy2(ROOT / "Makefile", makefile)
    common = [
        "make",
        "-f",
        str(makefile),
        "-n",
        "eval",
        f"RECIPE={RECIPE}",
    ]
    default = subprocess.run(common, cwd=tmp_path, check=True, capture_output=True, text=True)
    opted_in = subprocess.run(
        [*common, "PUBLISH=true", "EXPORT_SCENE=true"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    debug = subprocess.run(
        [*common, "VERBOSITY=DEBUG"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    invalid_publish = subprocess.run(
        [*common, "PUBLISH=maybe"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    invalid_scene_export = subprocess.run(
        [*common, "EXPORT_SCENE=maybe"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    invalid_gpu = subprocess.run(
        ["make", "-f", str(makefile), "eval", *common[5:], "POLICY_GPU=AUTO"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert "--publish" not in default.stdout
    assert "--export-scene" not in default.stdout
    assert '--log-level "INFO"' in default.stdout
    assert "--publish" in opted_in.stdout
    assert "--export-scene" in opted_in.stdout
    assert '--log-level "DEBUG"' in debug.stdout
    assert f'--recipe "{RECIPE}"' in default.stdout
    assert '--seed "0"' in default.stdout
    assert '--policy-gpu "auto"' in default.stdout
    assert '--env-gpu "auto"' in default.stdout
    assert '--eval-num "native"' in default.stdout
    assert invalid_publish.returncode != 0
    assert "PUBLISH must be true or false" in invalid_publish.stderr
    assert invalid_scene_export.returncode != 0
    assert "EXPORT_SCENE must be true or false" in invalid_scene_export.stderr
    assert invalid_gpu.returncode != 0
    assert "POLICY_GPU must be 'auto' or a nonnegative integer" in invalid_gpu.stderr


def test_make_setup_and_preflight_forward_experiment_contract():
    common = [
        f"RECIPE={RECIPE}",
        "SEED=0",
        "POLICY_GPU=0",
        "ENV_GPU=1",
        "EVAL_NUM=1",
        "PUBLISH=false",
    ]
    setup = subprocess.run(
        ["make", "-n", "setup", *common],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    deep = subprocess.run(
        ["make", "-n", "preflight", "DEEP=true", *common],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "uv run --locked robodojo --log-level" in setup.stdout
    assert " setup --recipe" in setup.stdout
    assert f'--recipe "{RECIPE}"' in setup.stdout
    assert "--no-sync robodojo --log-level" in deep.stdout
    assert " preflight --recipe" in deep.stdout
    assert '--env-gpu "1"' in deep.stdout
    assert "--deep" in deep.stdout
    assert "--publish" not in deep.stdout


def test_make_dry_run_toggle_and_local_sweeps():
    experiment = [
        f"RECIPE={RECIPE}",
        "SEED=0",
        "ENV_GPU=1",
        "POLICY_GPU=0",
        "EVAL_NUM=1",
    ]

    rendered = {
        target: subprocess.run(
            [
                "make",
                "-n",
                target,
                *experiment,
                "DRY_RUN=true",
                "PUBLISH=true",
                "EXPORT_SCENE=true",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for target in ("eval", "smoke", "benchmark")
    }

    assert "--dry-run" in rendered["eval"]
    assert "--publish" in rendered["eval"]
    assert "--export-scene" in rendered["eval"]
    for target in ("smoke", "benchmark"):
        assert "--dry-run" in rendered[target]
        assert "--publish" not in rendered[target]
        assert "--export-scene" not in rendered[target]


def test_make_help_exposes_only_the_supported_target_surface():
    result = subprocess.run(["make", "help"], cwd=ROOT, check=True, capture_output=True, text=True)
    for target in ("recipes", "setup", "preflight", "eval", "snapshots", "smoke", "benchmark", "results", "check"):
        assert target in result.stdout
    assert "make recipes -> make eval RECIPE=<name>" in result.stdout
    assert "optional machine defaults: .env (?= assignments)" in result.stdout
    for removed in ("init", "policy-setup", "eval-dry-run", "storage-publish", "docker-build", "assets-yam"):
        assert removed not in result.stdout


def test_task_inventory_reads_the_simulator_task_package():
    inventory = build_inventory()
    tasks = {item["name"]: item for item in inventory["tasks"]}
    assert inventory["config_dir"] == "configs/task"
    assert tasks["stack_bowls"]["runnable"] is True


def test_removed_openarm_cloth_profile_is_rejected():
    request = _simulator_request(
        task="fold_clothes",
        task_protocol="fold_clothes",
        environment="openarm_cloth_folding",
    )
    with pytest.raises(ValueError, match="environment config not found"):
        load_simulator_config(RepositoryPaths.resolve(ROOT), request)


def test_removed_generic_openarm_profile_is_rejected():
    request = _simulator_request(
        task="fold_clothes",
        task_protocol="fold_clothes",
        environment="openarm",
    )
    with pytest.raises(ValueError, match="environment config not found"):
        load_simulator_config(RepositoryPaths.resolve(ROOT), request)


@pytest.mark.parametrize("profile", ["openarm_wowrobo_v1_1", "openarm_anvil_v2"])
def test_unmeasured_openarm_profiles_are_release_blocked(profile):
    request = _simulator_request(
        task="fold_clothes",
        task_protocol="fold_clothes",
        environment=profile,
    )
    with pytest.raises(ValueError, match="calibration is not release-ready"):
        load_simulator_config(RepositoryPaths.resolve(ROOT), request)


def test_server_dry_run_validates_and_builds_adapter_argv(tmp_path):
    policy = tmp_path / "Policy"
    policy.mkdir()
    adapter = policy / "setup_eval_policy_server.sh"
    adapter.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    request = PolicyServerLaunchRequest(
        experiment=_experiment(policy_dir=policy),
        port=19000,
    )
    command = policy_server_command(request, 19000)
    assert command[:2] == ["bash", str(adapter)]
    assert command[-2:] == ["19000", "0.0.0.0"]


@pytest.mark.parametrize("profile", ["openarm_lerobot", "openarm_wowrobo_v1_1", "openarm_anvil_v2"])
def test_openarm_policy_uses_explicit_embodiment_contract(tmp_path, profile):
    policy = tmp_path / "LeRobot_Pi05_OpenArm"
    policy.mkdir()
    (policy / "setup_eval_policy_server.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    request = PolicyServerLaunchRequest(
        experiment=_experiment(
            policy_dir=policy,
            task="fold_clothes",
            checkpoint="folding_final",
            policy_runtime="lerobot-pi05",
            environment=profile,
            embodiment="openarm_lerobot",
            task_protocol="fold_clothes",
        ),
        port=19000,
    )
    command = policy_server_command(request, 19000)
    assert command[5] == "openarm_lerobot"


def test_server_cli_rejects_invalid_port():
    result = runner.invoke(
        app,
        [
            "server",
            "--recipe",
            RECIPE,
            "--policy-port",
            "70000",
            "--dry-run",
        ],
    )
    assert result.exit_code == 2
    assert "1<=x<=65535" in result.output


def test_cli_rejects_invalid_log_level():
    result = runner.invoke(app, ["--log-level", "verbose", "tasks"])
    assert result.exit_code == 2
    assert "Invalid value for --log-level" in result.output


def test_server_dry_run_separates_diagnostics_from_command_output(monkeypatch):
    from robodojo.orchestration import split

    captured = []
    monkeypatch.setattr(split, "run_server", lambda paths, request: captured.append(request) or 0)
    result = runner.invoke(
        app,
        [
            "--log-level",
            "INFO",
            "server",
            "--recipe",
            RECIPE,
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert captured[0].experiment.task == "general_pickup"
    assert captured[0].experiment.task_protocol == "general_pickup"


def test_cli_log_level_is_propagated_for_child_processes(monkeypatch):
    from robodojo.orchestration import split

    seen: list[str | None] = []
    monkeypatch.setenv("ROBODOJO_LOG_LEVEL", "WARNING")
    monkeypatch.setattr(
        split,
        "run_server",
        lambda paths, request: seen.append(os.environ.get("ROBODOJO_LOG_LEVEL")) or 0,
    )
    result = runner.invoke(
        app,
        [
            "--log-level",
            "debug",
            "server",
            "--recipe",
            RECIPE,
        ],
    )
    assert result.exit_code == 0
    assert seen == ["DEBUG"]


def test_repository_root_precedence(monkeypatch, tmp_path):
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "pyproject.toml").write_text('[project]\nname = "robodojo"\n', encoding="utf-8")
    monkeypatch.setenv("ROBODOJO_ROOT", str(fake))
    assert discover_repository_root() == fake
    assert RepositoryPaths.resolve(ROOT).root == ROOT


def test_runtime_settings_ignore_repository_dotenv(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "robodojo"\n', encoding="utf-8")
    (root / ".env").write_text(
        "ROBODOJO_EVAL_ROOT=/from-file\nROBODOJO_STORAGE_ROOT=/also-from-file\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ROBODOJO_STORAGE_ROOT", raising=False)
    for name in RuntimeSettings.REMOVED_STORAGE_VARIABLES:
        monkeypatch.delenv(name, raising=False)

    settings = RuntimeSettings.load(RepositoryPaths.resolve(root))

    assert settings.storage_root is None


def test_policy_imports_work_without_simulator_extra():
    code = (
        "import sys; from robodojo.policy.adapter import policy_server_command; "
        "assert not any(name.startswith(('isaacsim', 'isaaclab', 'torch')) for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_simulator_command_uses_the_domain_module_path():
    request = _simulator_request(
        environment_gpu=1,
        additional_info="ckpt_name=test,action_type=ee",
    )
    command, environment = simulator_command(RepositoryPaths.resolve(ROOT), request)
    assert command[command.index("-m") + 1] == "robodojo.sim.evaluation.main"
    assert command[command.index("--task") + 1] == "stack_bowls"
    assert "--layout_name" not in command
    assert "--layout-name" not in command
    assert command[command.index("--policy_server_url") + 1] == "ws://127.0.0.1:19000"
    assert command[command.index("--device") + 1] == "cuda:0"
    assert command[command.index("--device_id") + 1] == "1"
    assert command[command.index("--experience") + 1] == "isaaclab.python.kit"
    assert "--/app/extensions/registryEnabled=0" in command[command.index("--kit_args") + 1]
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"


def test_standard_and_openarm_profiles_keep_intended_parallelism():
    paths = RepositoryPaths.resolve(ROOT)
    arx = _simulator_request()
    openarm = arx.model_copy(
        update={"experiment": arx.experiment.model_copy(update={"environment": "openarm_lerobot"})}
    )

    assert load_simulator_config(paths, arx)[0] == 10
    assert load_simulator_config(paths, openarm)[0] == 1
