from pathlib import Path

import pytest

from robodojo.core.models import EvaluationRequest
from robodojo.core.paths import RepositoryPaths
from robodojo.orchestration import evaluation

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

    code = evaluation.run_evaluation(RepositoryPaths.resolve(ROOT), _request(policy_dir))

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
        evaluation.run_evaluation(RepositoryPaths.resolve(ROOT), _request(policy_dir))

    assert terminated == [process]
