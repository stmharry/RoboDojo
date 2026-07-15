from pathlib import Path
import subprocess

import pytest

from robodojo.core.models import EvaluationRequest, SimulatorLaunchRequest
from robodojo.core.paths import RepositoryPaths
from robodojo.orchestration import evaluation
from robodojo.workflows import storage as storage_workflow

ROOT = Path(__file__).resolve().parents[1]


def _request(policy_dir: Path) -> EvaluationRequest:
    return EvaluationRequest(
        policy_dir=policy_dir,
        task="stack_bowls",
        checkpoint="test-checkpoint",
        policy_env="test-policy-env",
    )


def _policy_dir(tmp_path: Path) -> Path:
    policy_dir = tmp_path / "TestPolicy"
    policy_dir.mkdir()
    (policy_dir / "setup_eval_policy_server.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    return policy_dir


def test_evaluation_coordinates_policy_readiness_simulator_and_cleanup(monkeypatch, tmp_path):
    policy_dir = _policy_dir(tmp_path)
    process = object()
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(evaluation, "free_port", lambda: 19000)
    monkeypatch.setattr(
        evaluation,
        "start",
        lambda argv, cwd, env: calls.append(("start", argv, cwd, env)) or process,
    )
    monkeypatch.setattr(
        evaluation,
        "wait_for_port",
        lambda child, host, port, timeout: calls.append(("wait", child, host, port, timeout)),
    )
    monkeypatch.setattr(
        evaluation,
        "run_simulator_session",
        lambda paths, request, environment, *, publish: calls.append(("simulator", request, environment, publish)) or 7,
    )
    monkeypatch.setattr(
        evaluation,
        "terminate_process_group",
        lambda child: calls.append(("terminate", child)),
    )

    code = evaluation.run_evaluation(RepositoryPaths.resolve(ROOT), _request(policy_dir), preflight=False)

    assert code == 7
    assert [call[0] for call in calls] == ["start", "wait", "simulator", "terminate"]
    assert calls[0][2] == policy_dir
    assert calls[1][1:] == (process, "127.0.0.1", 19000, 600)
    assert calls[2][3] is False


def test_evaluation_cleans_up_when_policy_readiness_fails(monkeypatch, tmp_path):
    policy_dir = _policy_dir(tmp_path)
    process = object()
    terminated: list[object] = []

    monkeypatch.setattr(evaluation, "free_port", lambda: 19000)
    monkeypatch.setattr(evaluation, "start", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        evaluation,
        "wait_for_port",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("not ready")),
    )
    monkeypatch.setattr(evaluation, "terminate_process_group", terminated.append)

    with pytest.raises(TimeoutError, match="not ready"):
        evaluation.run_evaluation(RepositoryPaths.resolve(ROOT), _request(policy_dir), preflight=False)

    assert terminated == [process]


@pytest.mark.parametrize(
    ("remote", "aws_path", "message"),
    [
        (None, "/usr/bin/aws", "ROBODOJO_S3_URI"),
        ("https://bucket.example/robodojo", "/usr/bin/aws", "s3://"),
        ("s3://bucket/robodojo", None, "AWS CLI"),
    ],
)
def test_publish_prerequisites_fail_before_policy_launch(monkeypatch, tmp_path, caplog, remote, aws_path, message):
    policy_dir = _policy_dir(tmp_path)
    if remote is None:
        monkeypatch.delenv("ROBODOJO_S3_URI", raising=False)
    else:
        monkeypatch.setenv("ROBODOJO_S3_URI", remote)
    monkeypatch.setattr(evaluation.shutil, "which", lambda name: aws_path)
    monkeypatch.setattr(evaluation, "free_port", lambda: pytest.fail("policy launch should not be prepared"))

    code = evaluation.run_evaluation(
        RepositoryPaths.resolve(ROOT),
        _request(policy_dir).model_copy(update={"publish": True}),
    )

    assert code == 2
    assert message in caplog.text


def test_publish_dry_run_skips_s3_prerequisites(monkeypatch, tmp_path, capsys):
    policy_dir = _policy_dir(tmp_path)
    monkeypatch.delenv("ROBODOJO_S3_URI", raising=False)
    monkeypatch.setattr(evaluation.shutil, "which", lambda name: None)

    code = evaluation.run_evaluation(
        RepositoryPaths.resolve(ROOT),
        _request(policy_dir).model_copy(update={"publish": True, "dry_run": True}),
    )

    assert code == 0
    assert "setup_eval_policy_server.sh" in capsys.readouterr().out


def test_simulator_session_publishes_only_when_requested(monkeypatch, tmp_path):
    request = SimulatorLaunchRequest(
        task="stack_bowls",
        policy_name="TestPolicy",
        port=19000,
        additional_info="test",
    )
    calls: list[list[str]] = []
    monkeypatch.setenv("ROBODOJO_S3_URI", "s3://bucket/robodojo")
    monkeypatch.setattr(evaluation, "run_simulator", lambda paths, request, environment: 0)
    monkeypatch.setattr(storage_workflow, "main", lambda argv: calls.append(argv) or 0)
    environment = {"ROBODOJO_RUN_ID": "2026-07-14_12-00-00"}

    assert evaluation.run_simulator_session(RepositoryPaths.resolve(ROOT), request, environment) == 0
    assert calls == []

    assert (
        evaluation.run_simulator_session(
            RepositoryPaths.resolve(ROOT),
            request,
            environment,
            publish=True,
        )
        == 0
    )
    assert calls == [["publish-eval", ".", "--run-id", "2026-07-14_12-00-00"]]


def test_simulator_session_does_not_publish_incomplete_runs(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(storage_workflow, "main", lambda argv: calls.append(argv) or 0)
    base = SimulatorLaunchRequest(
        task="stack_bowls",
        policy_name="TestPolicy",
        port=19000,
        additional_info="test",
    )

    monkeypatch.setattr(evaluation, "run_simulator", lambda paths, request, environment: 7)
    assert (
        evaluation.run_simulator_session(
            RepositoryPaths.resolve(ROOT),
            base,
            {"ROBODOJO_RUN_ID": "failed"},
            publish=True,
        )
        == 7
    )

    monkeypatch.setattr(evaluation, "run_simulator", lambda paths, request, environment: 0)
    dry_run = base.model_copy(update={"dry_run": True})
    assert evaluation.run_simulator_session(RepositoryPaths.resolve(ROOT), dry_run, publish=True) == 0
    assert (
        evaluation.run_simulator_session(
            RepositoryPaths.resolve(ROOT),
            base,
            {"ROBODOJO_RUN_ID": "scene", "ROBODOJO_EXPORT_SCENE_ONLY": "true"},
            publish=True,
        )
        == 0
    )
    assert calls == []


@pytest.mark.parametrize(
    ("failure", "expected_code", "message"),
    [
        (SystemExit("remote destination is already complete"), 1, "already complete"),
        (subprocess.CalledProcessError(5, ["aws"], stderr="access denied"), 5, "access denied"),
        (OSError("aws executable failed"), 1, "aws executable failed"),
    ],
)
def test_publication_failure_returns_nonzero_and_preserves_local_result(
    monkeypatch,
    tmp_path,
    caplog,
    failure,
    expected_code,
    message,
):
    local_result = tmp_path / "_result.json"
    local_result.write_text('{"eval_time": 1}\n', encoding="utf-8")
    request = SimulatorLaunchRequest(
        task="stack_bowls",
        policy_name="TestPolicy",
        port=19000,
        additional_info="test",
    )
    monkeypatch.setattr(evaluation, "run_simulator", lambda paths, request, environment: 0)

    def fail_publish(argv):
        raise failure

    monkeypatch.setattr(storage_workflow, "main", fail_publish)

    code = evaluation.run_simulator_session(
        RepositoryPaths.resolve(ROOT),
        request,
        {"ROBODOJO_RUN_ID": "publish-failure"},
        publish=True,
    )

    assert code == expected_code
    assert message in caplog.text
    assert local_result.read_text(encoding="utf-8") == '{"eval_time": 1}\n'
