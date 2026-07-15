from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner
import yaml

from robodojo.cli import app
from robodojo.core.models import EvaluationRequest, PreflightCheck, PreflightRequest
from robodojo.core.paths import RepositoryPaths
from robodojo.orchestration import evaluation
from robodojo.workflows import preflight, sweeps

ROOT = Path(__file__).resolve().parents[1]
RUNNER = CliRunner()


def _request(tmp_path: Path, **updates) -> PreflightRequest:
    values = {
        "policy_dir": tmp_path / "Policy",
        "task": "stack_bowls",
        "checkpoint": "alias",
        "policy_env": "uv",
    }
    values.update(updates)
    return PreflightRequest(**values)


def _report(status: str = "PASS"):
    return preflight.build_report([PreflightCheck(name="test", status=status, detail="test")])


def test_report_json_has_stable_overall_and_per_check_fields(capsys):
    report = preflight.build_report(
        [
            PreflightCheck(name="ready", status="PASS", detail="ok"),
            PreflightCheck(name="legacy", status="WARN", detail="generic only", remediation="make policy-setup"),
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
                "remediation": "make policy-setup",
            },
        ],
    }


@pytest.mark.parametrize("output_format", ["human", "json"])
def test_policy_setup_cli_invokes_eight_argument_hook_and_reports(monkeypatch, tmp_path, output_format):
    policy = tmp_path / "Policy"
    policy.mkdir()
    hook = policy / "prepare_eval_policy.sh"
    hook.write_text('#!/usr/bin/env bash\nprintf "prepared:%s\\n" "$*"\n', encoding="utf-8")
    result = RUNNER.invoke(
        app,
        [
            "policy-setup",
            "--policy-dir",
            str(policy),
            "--task",
            "stack_bowls",
            "--ckpt",
            "release",
            "--policy-env",
            "runtime",
            "--dataset",
            "RoboDojo",
            "--env-cfg",
            "arx_x5",
            "--action-type",
            "joint",
            "--seed",
            "4",
            "--policy-gpu",
            "2",
            "--format",
            output_format,
            "--root",
            str(ROOT),
        ],
    )

    assert result.exit_code == 0
    expected = "RoboDojo stack_bowls release arx_x5 joint 4 2 runtime"
    assert expected in result.stdout
    if output_format == "json":
        payload = json.loads(result.stdout)
        assert payload["status"] == "PASS"
        assert payload["checks"][0]["name"] == "policy_setup"
    else:
        assert "PASS policy_setup:" in result.stdout


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
    assert checks[0].remediation == "make policy-setup"


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
    assert checks[-1].remediation == "make policy-setup"


def test_explicit_checkpoint_must_exist_and_opaque_alias_warns(tmp_path):
    missing = preflight._checkpoint_check(_request(tmp_path, checkpoint="./missing"))
    opaque = preflight._checkpoint_check(_request(tmp_path, checkpoint="release-alias"))

    assert missing.status == "FAIL"
    assert "does not exist" in missing.detail
    assert opaque.status == "WARN"


def test_gpu_indices_are_validated(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: preflight.subprocess.CompletedProcess(args[0], 0, "0\n2\n", ""),
    )

    result = preflight._gpu_check(_request(tmp_path, policy_gpu=1, env_gpu=2))

    assert result.status == "FAIL"
    assert "[1]" in result.detail
    assert "POLICY_GPU" in result.remediation


def test_yam_manifest_requires_asset_and_matching_checksums(monkeypatch, tmp_path):
    config = tmp_path / "dual_yam.yml"
    config.write_text(yaml.safe_dump({"robots": [{"robot_name": "yam"}]}), encoding="utf-8")
    profile = SimpleNamespace(component_paths={"robot": config})
    asset_root = tmp_path / "assets"
    monkeypatch.setattr(preflight, "assets_root", lambda: asset_root)

    missing = preflight._robot_asset_check(profile)
    assert missing.status == "FAIL"
    assert missing.remediation == "make assets-yam"

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

    result = preflight._layout_check(paths, _request(tmp_path, task="general_pickup"), scene)

    assert result.status == "FAIL"
    assert "general_pickup_*.json" in result.detail
    assert result.remediation == "make assets"


def test_failed_launch_preflight_stops_before_port_process_and_simulator(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "run_fast_preflight", lambda paths, request: _report("FAIL"))
    monkeypatch.setattr(evaluation, "free_port", lambda: pytest.fail("free_port must not be called"))
    monkeypatch.setattr(evaluation, "start", lambda *args, **kwargs: pytest.fail("process must not start"))
    monkeypatch.setattr(
        evaluation,
        "run_simulator_session",
        lambda *args, **kwargs: pytest.fail("simulator must not start"),
    )
    request = EvaluationRequest(
        policy_dir=tmp_path / "Policy",
        task="stack_bowls",
        checkpoint="alias",
        policy_env="uv",
    )

    assert evaluation.run_evaluation(RepositoryPaths(root=ROOT), request) == 2


def test_failed_server_preflight_stops_before_policy_runner(monkeypatch, tmp_path):
    from robodojo.policy import adapter

    policy = tmp_path / "Policy"
    policy.mkdir()
    monkeypatch.setattr(preflight, "run_fast_preflight", lambda paths, request: _report("FAIL"))
    monkeypatch.setattr(adapter, "run_policy_server", lambda request: pytest.fail("policy runner must not start"))

    result = RUNNER.invoke(
        app,
        [
            "server",
            "--policy-dir",
            str(policy),
            "--task",
            "stack_bowls",
            "--ckpt",
            "release",
            "--policy-env",
            "runtime",
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
    request = EvaluationRequest(
        policy_dir=policy,
        task="stack_bowls",
        checkpoint="alias",
        policy_env="uv",
        dry_run=True,
    )

    assert evaluation.run_evaluation(RepositoryPaths(root=ROOT), request) == 0
    assert "setup_eval_policy_server.sh" in capsys.readouterr().out


def test_sweep_preflights_once_and_children_skip_duplicate_gate(monkeypatch, tmp_path):
    from robodojo.core.models import SweepRequest

    calls = 0
    children = []

    def fast(paths, request):
        nonlocal calls
        calls += 1
        return _report()

    monkeypatch.setattr(preflight, "run_fast_preflight", fast)
    monkeypatch.setattr(sweeps, "run_work_root", lambda: tmp_path)
    monkeypatch.setattr(sweeps, "_selected_tasks", lambda request: ["general_pickup", "fold_clothes"])
    monkeypatch.setattr(
        sweeps,
        "run_evaluation",
        lambda paths, request, *, preflight: children.append((request.task, preflight)) or 0,
    )
    request = SweepRequest(
        policy_dir=tmp_path / "Policy",
        checkpoint="alias",
        policy_env="uv",
        env_config="bimanual_yam",
        scene_config="molmo_yam",
        run_id="once",
    )

    assert sweeps.run_sweep(RepositoryPaths(root=ROOT), request) == 0
    assert calls == 1
    assert children == [("general_pickup", False), ("fold_clothes", False)]


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
