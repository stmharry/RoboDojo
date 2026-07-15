from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner
import yaml

from robodojo.cli import app
from robodojo.core import gpu
from robodojo.core.models import EvaluationRequest, PreflightCheck, PreflightRequest, SetupRequest, SetupStage
from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles import load_environment_profile, load_scene_profile
from robodojo.orchestration import evaluation
from robodojo.workflows import assets, preflight, sweeps

ROOT = Path(__file__).resolve().parents[1]
RUNNER = CliRunner()


def _request(tmp_path: Path, **updates) -> PreflightRequest:
    values = {
        "policy_dir": tmp_path / "Policy",
        "task": "stack_bowls",
        "checkpoint": "alias",
        "policy_env": "uv",
        "env_config": "arx_x5",
        "policy_contract": "arx_x5",
        "protocol": "stack_bowls",
        "layout": "stack_bowls",
        "episode_horizon": 800,
        "native_eval_num": 25,
        "scene_config": "default",
        "action_type": "joint",
        "policy_gpu": 0,
        "env_gpu": 1,
    }
    values.update(updates)
    return PreflightRequest(**values)


def _evaluation(tmp_path: Path, **updates) -> EvaluationRequest:
    values = _request(tmp_path).model_dump(exclude={"publish", "deep", "timeout"})
    values.update(updates)
    return EvaluationRequest(**values)


def _report(status: str = "PASS"):
    return preflight.build_report([PreflightCheck(name="test", status=status, detail="test")])


def test_report_json_has_stable_overall_and_per_check_fields(capsys):
    report = preflight.build_report(
        [
            PreflightCheck(name="ready", status="PASS", detail="ok"),
            PreflightCheck(name="legacy", status="WARN", detail="generic only", remediation="make setup"),
        ]
    )

    preflight.emit_report(report, "json")

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "status": "WARN",
        "checks": [
            {"name": "ready", "status": "PASS", "detail": "ok"},
            {
                "name": "legacy",
                "status": "WARN",
                "detail": "generic only",
                "remediation": "make setup",
            },
        ],
    }


@pytest.mark.parametrize("output_format", ["human", "json"])
def test_setup_policy_stage_invokes_eight_argument_hook_and_reports(tmp_path, output_format, capsys):
    from robodojo.workflows import setup as setup_workflow

    policy = tmp_path / "Policy"
    policy.mkdir()
    hook = policy / "prepare_eval_policy.sh"
    hook.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" > hook_args.txt\n', encoding="utf-8")
    request = SetupRequest(
        stages=(SetupStage.POLICY,),
        policy_dir=policy,
        task="stack_bowls",
        checkpoint="release",
        policy_env="runtime",
        dataset="RoboDojo",
        env_config="arx_x5",
        policy_contract="arx_x5",
        action_type="joint",
        protocol="stack_bowls",
        layout="stack_bowls",
        episode_horizon=800,
        native_eval_num=25,
        seed=4,
        policy_gpu=2,
    )
    assert setup_workflow.setup(RepositoryPaths.resolve(ROOT), request, output_format=output_format) == 0
    output = capsys.readouterr().out
    expected = "RoboDojo stack_bowls release arx_x5 joint 4 2 runtime"
    assert (policy / "hook_args.txt").read_text(encoding="utf-8").strip() == expected
    if output_format == "json":
        payload = json.loads(output)
        assert payload["status"] == "PASS"
        assert payload["stages"][-1]["name"] == "policy"
    else:
        assert "CHANGED policy:" in output


def test_setup_policy_stage_resolves_only_the_policy_gpu(monkeypatch, tmp_path):
    from robodojo.core.gpu import GpuAssignment
    from robodojo.workflows import setup as setup_workflow

    policy = tmp_path / "Policy"
    policy.mkdir()
    (policy / "prepare_eval_policy.sh").write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" > hook_args.txt\n',
        encoding="utf-8",
    )
    selections = []

    def resolve(**selectors):
        selections.append(selectors)
        return GpuAssignment(policy_gpu=6, policy_source="auto")

    monkeypatch.setattr(setup_workflow, "resolve_gpus", resolve)
    request = SetupRequest(
        stages=(SetupStage.POLICY,),
        policy_dir=policy,
        task="stack_bowls",
        checkpoint="release",
        policy_env="runtime",
        env_config="arx_x5",
        policy_contract="arx_x5",
        action_type="joint",
        protocol="stack_bowls",
        layout="stack_bowls",
        episode_horizon=800,
        native_eval_num=25,
    )

    result = setup_workflow._policy_stage(RepositoryPaths(root=ROOT), request)

    assert result.status == "CHANGED"
    assert selections == [{"policy_gpu": "auto"}]
    assert (policy / "hook_args.txt").read_text(encoding="utf-8").split()[6] == "6"


def test_missing_and_stale_uv_environments_are_actionable(monkeypatch, tmp_path):
    policy = tmp_path / "Policy"
    policy.mkdir()
    (policy / "deploy.yml").write_text("policy_uv_env_path: project\n", encoding="utf-8")
    request = _request(tmp_path)

    python, project, error = preflight._resolve_policy_python(request)
    assert python is None
    assert project == policy / "project"
    assert "Python is missing" in error

    (project / ".venv/bin").mkdir(parents=True)
    (project / ".venv/bin/python").touch()
    (project / "pyproject.toml").touch()
    (project / "uv.lock").touch()
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: preflight.subprocess.CompletedProcess(args[0], 1, "", "stale lock"),
    )

    checks = preflight._policy_runtime_checks(RepositoryPaths(root=ROOT), request)

    assert checks[0].status == "FAIL"
    assert "stale" in checks[0].detail
    assert checks[0].remediation == "make setup with the same complete manual contract"


def test_missing_conda_environment_and_failed_xpolicylab_import(monkeypatch, tmp_path):
    policy = tmp_path / "Policy"
    policy.mkdir()
    conda_request = _request(tmp_path, policy_env="missing-conda")
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/conda" if name == "conda" else None)
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: preflight.subprocess.CompletedProcess(args[0], 0, '{"envs": []}', ""),
    )
    assert "does not exist" in preflight._resolve_policy_python(conda_request)[2]

    project = policy / "project"
    (project / ".venv/bin").mkdir(parents=True)
    (project / ".venv/bin/python").touch()
    (project / "pyproject.toml").touch()
    (project / "uv.lock").touch()
    calls = 0

    def command(*args, **kwargs):
        nonlocal calls
        calls += 1
        return preflight.subprocess.CompletedProcess(args[0], 0 if calls == 1 else 1, "", "import failed")

    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(preflight.subprocess, "run", command)
    checks = preflight._policy_runtime_checks(
        RepositoryPaths(root=ROOT),
        _request(tmp_path, policy_env=str(project)),
    )
    assert [item.status for item in checks] == ["PASS", "FAIL"]
    assert checks[-1].name == "xpolicylab_import"
    assert checks[-1].remediation == "make setup with the same complete manual contract"


def test_explicit_checkpoint_must_exist_and_opaque_alias_warns(tmp_path):
    missing = preflight._checkpoint_check(_request(tmp_path, checkpoint="./missing"))
    opaque = preflight._checkpoint_check(_request(tmp_path, checkpoint="release-alias"))

    assert missing.status == "FAIL"
    assert "does not exist" in missing.detail
    assert opaque.status == "WARN"


def test_gpu_indices_are_validated(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        gpu.subprocess,
        "run",
        lambda *args, **kwargs: preflight.subprocess.CompletedProcess(args[0], 0, "0, 100\n2, 100\n", ""),
    )

    result = preflight._gpu_check(_request(tmp_path, policy_gpu=1, env_gpu=2))

    assert result.status == "FAIL"
    assert "[1]" in result.detail
    assert "POLICY_GPU" in result.remediation


def test_scene_only_preflight_skips_every_policy_side_check(monkeypatch, tmp_path):
    from robodojo.core.gpu import GpuAssignment

    selections = []

    def resolve(**selectors):
        selections.append(selectors)
        return GpuAssignment(env_gpu=4, env_source="auto")

    monkeypatch.setattr(preflight, "resolve_gpus", resolve)
    monkeypatch.setattr(
        preflight,
        "_root_runtime_check",
        lambda paths: PreflightCheck(name="root", status="PASS", detail="ok"),
    )
    monkeypatch.setattr(
        preflight,
        "_configuration_checks",
        lambda paths, request: ([PreflightCheck(name="configuration", status="PASS", detail="ok")], None, None),
    )
    monkeypatch.setattr(
        preflight,
        "_layout_check",
        lambda *args: PreflightCheck(name="layout", status="PASS", detail="ok"),
    )
    monkeypatch.setattr(
        preflight,
        "_robot_asset_check",
        lambda *args: PreflightCheck(name="robot_assets", status="PASS", detail="ok"),
    )
    monkeypatch.setattr(
        preflight,
        "_scene_asset_check",
        lambda *args: PreflightCheck(name="scene_assets", status="PASS", detail="ok"),
    )
    for name in (
        "_publication_check",
        "_adapter_files_check",
        "_policy_runtime_checks",
        "_checkpoint_check",
        "_policy_hook_check",
    ):
        monkeypatch.setattr(preflight, name, lambda *args, name=name: pytest.fail(f"scene-only ran {name}"))

    request = _request(tmp_path).model_copy(update={"policy_gpu": "auto", "env_gpu": "auto"})
    report = preflight.run_simulator_preflight(RepositoryPaths(root=ROOT), request)

    assert report.status == "PASS"
    assert selections == [{"policy_gpu": None, "env_gpu": "auto"}]
    assert [check.name for check in report.checks] == [
        "root",
        "configuration",
        "layout",
        "robot_assets",
        "scene_assets",
        "gpu_indices",
    ]


def test_yam_manifest_requires_asset_and_matching_checksums(monkeypatch, tmp_path):
    config = tmp_path / "dual_yam.yml"
    config.write_text(yaml.safe_dump({"robots": [{"robot_name": "yam"}]}), encoding="utf-8")
    profile = SimpleNamespace(component_paths={"robot": config})
    asset_root = tmp_path / "assets"
    monkeypatch.setattr(assets, "assets_root", lambda: asset_root)

    missing = preflight._robot_asset_check(profile)
    assert missing.status == "FAIL"
    assert missing.remediation == "make setup"

    robot_root = asset_root / "Robots/yam"
    robot_root.mkdir(parents=True)
    output = robot_root / "YAM.usd"
    output.write_bytes(b"valid")
    (robot_root / "manifest.json").write_text(
        json.dumps({"outputs": {"YAM.usd": hashlib.sha256(b"other").hexdigest()}}),
        encoding="utf-8",
    )
    mismatch = preflight._robot_asset_check(profile)
    assert mismatch.status == "FAIL"
    assert "checksum mismatch" in mismatch.detail

    (robot_root / "manifest.json").write_text(
        json.dumps({"outputs": {"YAM.usd": hashlib.sha256(b"valid").hexdigest()}}),
        encoding="utf-8",
    )
    assert preflight._robot_asset_check(profile).status == "PASS"


def test_layout_check_fails_when_selected_task_seed_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "assets_root", lambda: tmp_path / "assets")
    paths = RepositoryPaths(root=tmp_path)
    scene = SimpleNamespace(document=SimpleNamespace(layout_set="molmo_yam"))

    result = preflight._layout_check(
        paths,
        _request(tmp_path, task="general_pickup", protocol="general_pickup", layout="general_pickup"),
        scene,
    )

    assert result.status == "FAIL"
    assert "general_pickup_*.json" in result.detail
    assert result.remediation == "make setup with the same complete manual contract"
    assert "same complete manual contract" in result.remediation


@pytest.mark.parametrize(
    ("environment", "scene_name"),
    [
        ("bimanual_yam_molmoact2", "molmo_yam"),
        ("bimanual_yam_moonlake_office", "moonlake_office"),
    ],
)
def test_general_pickup_preflight_validates_the_same_role_and_workspace_contract(environment, scene_name):
    paths = RepositoryPaths.resolve(ROOT)
    profile = load_environment_profile(paths, environment)
    scene = load_scene_profile(paths, scene_name)
    request = _request(
        ROOT,
        task="general_pickup",
        protocol="general_pickup",
        layout="general_pickup",
        episode_horizon=200,
        native_eval_num=50,
        env_config=environment,
        scene_config=scene_name,
        seed=0,
    )

    result = preflight._layout_check(paths, request, scene, profile)

    assert result.status == "PASS"
    assert result.detail.startswith("1 layout(s)")


def test_failed_launch_preflight_stops_before_port_process_and_simulator(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "run_fast_preflight", lambda paths, request: _report("FAIL"))
    monkeypatch.setattr(evaluation, "free_port", lambda: pytest.fail("free_port must not be called"))
    monkeypatch.setattr(evaluation, "start", lambda *args, **kwargs: pytest.fail("process must not start"))
    monkeypatch.setattr(
        evaluation,
        "run_simulator_session",
        lambda *args, **kwargs: pytest.fail("simulator must not start"),
    )
    request = _evaluation(tmp_path)

    assert evaluation.run_evaluation(RepositoryPaths(root=ROOT), request) == 2


def test_failed_server_preflight_stops_before_policy_runner(monkeypatch):
    from robodojo.policy import adapter

    monkeypatch.setattr(preflight, "run_fast_preflight", lambda paths, request: _report("FAIL"))
    monkeypatch.setattr(adapter, "run_policy_server", lambda request: pytest.fail("policy runner must not start"))

    result = RUNNER.invoke(
        app,
        [
            "server",
            "--recipe",
            "pi05-bimanual_yam-molmo_yam-general_pickup",
            "--policy-gpu",
            "0",
            "--env-gpu",
            "1",
            "--root",
            str(ROOT),
        ],
    )

    assert result.exit_code == 1


def test_eval_dry_run_does_not_preflight(monkeypatch, tmp_path, capsys):
    policy = tmp_path / "Policy"
    policy.mkdir()
    (policy / "setup_eval_policy_server.sh").touch()
    monkeypatch.setattr(preflight, "run_fast_preflight", lambda *args: pytest.fail("dry run preflighted"))
    request = _evaluation(tmp_path, policy_dir=policy, dry_run=True)

    assert evaluation.run_evaluation(RepositoryPaths(root=ROOT), request) == 0
    assert "setup_eval_policy_server.sh" in capsys.readouterr().out


def test_sweep_preflights_each_explicit_recipe(monkeypatch, tmp_path):
    from robodojo.core.gpu import GpuAssignment
    from robodojo.core.models import SweepRequest

    children = []
    gpu_calls = 0

    def resolve(**selectors):
        nonlocal gpu_calls
        gpu_calls += 1
        return GpuAssignment(policy_gpu=4, env_gpu=2, policy_source="auto", env_source="auto")

    monkeypatch.setattr(sweeps, "resolve_gpus", resolve)
    monkeypatch.setattr(sweeps, "run_work_root", lambda: tmp_path)
    monkeypatch.setattr(
        sweeps,
        "run_evaluation",
        lambda paths, request, *, preflight: (
            children.append((request.recipe, request.task, preflight, request.policy_gpu, request.env_gpu)) or 0
        ),
    )
    request = SweepRequest(
        recipes=(
            "pi05-bimanual_yam-molmo_yam-general_pickup",
            "molmoact2-bimanual_yam-molmo_yam-fold_clothes",
        ),
        run_id="once",
    )

    assert sweeps.run_sweep(RepositoryPaths(root=ROOT), request) == 0
    assert gpu_calls == 1
    assert children == [
        ("pi05-bimanual_yam-molmo_yam-general_pickup", "general_pickup", True, 4, 2),
        ("molmoact2-bimanual_yam-molmo_yam-fold_clothes", "fold_clothes", True, 4, 2),
    ]


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        (None, "PASS"),
        (RuntimeError("policy exited"), "FAIL"),
        (TimeoutError("policy timed out"), "FAIL"),
    ],
)
def test_deep_preflight_always_cleans_up(monkeypatch, tmp_path, failure, expected_status):
    process = object()
    terminated = []
    monkeypatch.setattr(preflight, "run_fast_preflight", lambda paths, request: _report())
    monkeypatch.setattr(preflight, "free_port", lambda: 19001)
    monkeypatch.setattr(preflight, "policy_server_command", lambda request, port: ["policy-server"])
    monkeypatch.setattr(preflight, "start", lambda *args, **kwargs: process)

    def wait(*args, **kwargs):
        if failure:
            raise failure

    monkeypatch.setattr(preflight, "wait_for_port", wait)
    monkeypatch.setattr(preflight, "terminate_process_group", terminated.append)

    report = preflight.run_deep_preflight(RepositoryPaths(root=ROOT), _request(tmp_path, deep=True, timeout=0.1))

    assert report.status == expected_status
    assert terminated == [process]
