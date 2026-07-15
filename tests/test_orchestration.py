from pathlib import Path
import subprocess

import pytest

from robodojo.core.models import EvaluationRequest, SimulatorLaunchRequest
from robodojo.core.paths import RepositoryPaths
from robodojo.orchestration import evaluation, split
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
        lambda paths, request, environment: calls.append(("simulator", request, environment)) or 7,
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


def test_simulator_session_never_performs_publication(monkeypatch):
    request = SimulatorLaunchRequest(
        task="stack_bowls",
        policy_name="TestPolicy",
        port=19000,
        additional_info="test",
    )
    monkeypatch.setattr(evaluation, "run_simulator", lambda paths, request, environment: 0)
    monkeypatch.setattr(
        storage_workflow,
        "publish_evaluation_run",
        lambda run_id: pytest.fail("simulator session invoked publication"),
    )
    assert evaluation.run_simulator_session(RepositoryPaths.resolve(ROOT), request) == 0


@pytest.mark.parametrize("dry_run", [False, True])
def test_client_reachability_is_owned_by_orchestration(monkeypatch, caplog, dry_run):
    request = SimulatorLaunchRequest(
        task="stack_bowls",
        policy_name="TestPolicy",
        host="policy.example",
        port=19000,
        additional_info="test",
        dry_run=dry_run,
    )
    reachability: list[tuple[str, int, float]] = []
    monkeypatch.setattr(
        split,
        "warn_if_server_unreachable",
        lambda host, port, timeout: reachability.append((host, port, timeout)) or "warning: not ready",
    )
    monkeypatch.setattr(evaluation, "run_simulator_session", lambda paths, request: 0)

    assert split.run_client(RepositoryPaths.resolve(ROOT), request, connect_timeout=2.5) == 0
    assert reachability == ([] if dry_run else [("policy.example", 19000, 2.5)])
    assert ("not ready" in caplog.text) == (not dry_run)


def test_evaluation_publishes_once_only_after_success(monkeypatch, tmp_path):
    policy_dir = _policy_dir(tmp_path)
    published: list[str] = []
    process = object()
    simulator_code = 0
    monkeypatch.setenv("ROBODOJO_S3_URI", "s3://bucket/robodojo")
    monkeypatch.setattr(evaluation.shutil, "which", lambda name: "/usr/bin/aws")
    monkeypatch.setattr(evaluation, "free_port", lambda: 19000)
    monkeypatch.setattr(evaluation, "start", lambda *args, **kwargs: process)
    monkeypatch.setattr(evaluation, "wait_for_port", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluation, "terminate_process_group", lambda process: None)
    monkeypatch.setattr(evaluation, "run_simulator_session", lambda *args, **kwargs: simulator_code)
    monkeypatch.setattr(evaluation, "_publish_evaluation", lambda run_id: published.append(run_id) or 0)

    request = _request(policy_dir).model_copy(update={"publish": True})
    assert evaluation.run_evaluation(RepositoryPaths.resolve(ROOT), request, preflight=False) == 0
    assert len(published) == 1

    simulator_code = 7
    assert evaluation.run_evaluation(RepositoryPaths.resolve(ROOT), request, preflight=False) == 7
    assert len(published) == 1


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

    def fail_publish(run_id):
        raise failure

    monkeypatch.setattr(storage_workflow, "publish_evaluation_run", fail_publish)

    code = evaluation._publish_evaluation("publish-failure")

    assert code == expected_code
    assert message in caplog.text
    assert local_result.read_text(encoding="utf-8") == '{"eval_time": 1}\n'
