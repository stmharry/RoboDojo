import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from robodojo.core.scene_identity import ARTIFACT_SCHEMA_VERSION
from robodojo.workflows import storage as storage_cli


def _result_payload(**overrides):
    payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "task_name": "general_pickup",
        "protocol_name": "moonlake_office_general_pickup",
        "episode_horizon": 400,
        "native_eval_num": 50,
        "scene_config": "moonlake_office",
        "layout_config_name": "moonlake_office",
        "layout_source": "bundled",
        "layout_set_hash": "a" * 64,
        "eval_time": 1,
        "details": {
            "0": {
                "layout_id": 0,
                "layout_file": "general_pickup_0.json",
                "layout_sha256": "b" * 64,
                "success": False,
                "score": 0.0,
            }
        },
    }
    payload.update(overrides)
    return payload


def test_storage_cli_help_runs_through_typer():
    environment = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-m", "robodojo.cli", "storage", "--help"],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert "publish-checkpoint" in result.stdout
    assert "pull" in result.stdout


def test_storage_dry_runs_preserve_stdout(monkeypatch, tmp_path, capsys):
    source = tmp_path / "source"
    source.mkdir()
    local = tmp_path / "local"
    monkeypatch.setenv("ROBODOJO_STORAGE_ROOT", str(local))
    monkeypatch.setenv("ROBODOJO_S3_URI", "s3://bucket/robodojo")

    storage_cli.publish(source, "datasets/example", dry_run=True)
    storage_cli.pull("datasets/example", dry_run=True)

    captured = capsys.readouterr()
    assert captured.out == (
        f"publish {source.resolve()} -> s3://bucket/robodojo/datasets/example\n"
        f"pull s3://bucket/robodojo/datasets/example -> {local / 'datasets/example'}\n"
    )
    assert captured.err == ""


def test_publish_uses_canonical_destination_and_completion_is_last(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "video.mp4").write_bytes(b"video")
    (source / "_result.json").write_text(json.dumps(_result_payload()), encoding="utf-8")
    monkeypatch.setenv("ROBODOJO_S3_URI", "s3://bucket/robodojo")
    monkeypatch.setenv("ROBODOJO_STORAGE_ROOT", str(tmp_path / "local"))
    calls = []

    def fake_aws(*args, check=True):
        calls.append(args)
        return subprocess.CompletedProcess(args, 1 if args[:2] == ("s3api", "head-object") else 0, "", "")

    monkeypatch.setattr(storage_cli, "_aws", fake_aws)
    storage_cli.publish(source, "runs/eval_result/RoboDojo/task/run")

    assert calls[1][:4] == (
        "s3",
        "sync",
        str(source.resolve()),
        "s3://bucket/robodojo/runs/eval_result/RoboDojo/task/run",
    )
    assert ".cache/*" in calls[1]
    assert "*/.cache/*" in calls[1]
    assert "*.lock" in calls[1]
    assert "*.partial.*" in calls[1]
    assert calls[-2][-2].endswith("/_result.json")
    assert calls[-1][-2].endswith("/_COMPLETE.json")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("artifact_schema_version"), "artifact_schema_version mismatch"),
        (lambda payload: payload.update(artifact_schema_version=1), "artifact_schema_version mismatch"),
        (lambda payload: payload.update(layout_name="general_pickup"), "removed layout_name selector"),
        (lambda payload: payload.update(details={}), "incomplete episode details"),
    ],
)
def test_evaluation_publication_strictly_rejects_legacy_or_incomplete_results(
    monkeypatch,
    tmp_path,
    mutation,
    message,
):
    source = tmp_path / "source"
    source.mkdir()
    payload = _result_payload()
    mutation(payload)
    (source / "_result.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("ROBODOJO_S3_URI", "s3://bucket/robodojo")

    with pytest.raises(SystemExit, match=message):
        storage_cli.publish(source, "runs/eval_result/RoboDojo/general_pickup/run", dry_run=True)


def test_publish_evaluation_run_requires_current_result_before_dry_run(monkeypatch, tmp_path):
    storage = tmp_path / "storage"
    run = (
        storage
        / "runs/eval_result/RoboDojo/general_pickup/TestPolicy/bimanual_yam"
        / "0_ckpt_name=test,action_type=joint/legacy-run"
    )
    run.mkdir(parents=True)
    (run / "_result.json").write_text(json.dumps({"eval_time": 1}), encoding="utf-8")
    monkeypatch.setenv("ROBODOJO_STORAGE_ROOT", str(storage))
    monkeypatch.setenv("ROBODOJO_S3_URI", "s3://bucket/robodojo")

    with pytest.raises(SystemExit, match="artifact_schema_version mismatch"):
        storage_cli.publish_evaluation_run("legacy-run", dry_run=True)


def test_publish_rejects_completed_destination(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setenv("ROBODOJO_S3_URI", "s3://bucket/robodojo")

    def already_complete(*args, check=True):
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(storage_cli, "_aws", already_complete)
    with pytest.raises(SystemExit, match="already complete"):
        storage_cli.publish(source, "assets")


def test_payload_excludes_cache_git_locks_partials_and_rejects_symlinks(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    git = source / ".git"
    git.mkdir()
    (git / "index").write_bytes(b"ignored")
    cache = source / "nested" / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "weights.metadata").write_bytes(b"ignored")
    (source / "download.lock").write_bytes(b"ignored")
    (source / "weights.partial.123").write_bytes(b"ignored")
    (source / "weights.tmp").write_bytes(b"ignored")
    (source / "payload").write_bytes(b"kept")
    assert storage_cli._payload_files(source) == [source / "payload"]

    (source / "link").symlink_to(source / "payload")
    with pytest.raises(SystemExit, match="unsupported symlink"):
        storage_cli._payload_files(source)


def test_destination_cannot_escape_prefix(monkeypatch):
    monkeypatch.setenv("ROBODOJO_S3_URI", "s3://bucket/robodojo")
    with pytest.raises(SystemExit, match="invalid storage destination"):
        storage_cli._destination("../other")


def test_pull_verifies_and_installs_canonical_payload(monkeypatch, tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "weights.bin").write_bytes(b"weights")
    manifest, complete = storage_cli._metadata(remote, "datasets/example")
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    complete["manifest_sha256"] = hashlib.sha256(manifest_text.encode()).hexdigest()
    (remote / "_MANIFEST.json").write_text(manifest_text, encoding="utf-8")
    (remote / "_COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")
    local = tmp_path / "local"
    monkeypatch.setenv("ROBODOJO_STORAGE_ROOT", str(local))
    monkeypatch.setenv("ROBODOJO_S3_URI", "s3://bucket/robodojo")

    def fake_aws(*args, check=True):
        if args[:2] == ("s3", "sync"):
            shutil.copytree(remote, Path(args[3]), dirs_exist_ok=True)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(storage_cli, "_aws", fake_aws)
    storage_cli.pull("datasets/example")
    assert (local / "datasets/example/weights.bin").read_bytes() == b"weights"

    (remote / "weights.bin").write_bytes(b"replacement")
    manifest, complete = storage_cli._metadata(remote, "datasets/example")
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    complete["manifest_sha256"] = hashlib.sha256(manifest_text.encode()).hexdigest()
    (remote / "_MANIFEST.json").write_text(manifest_text, encoding="utf-8")
    (remote / "_COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")
    storage_cli.pull("datasets/example", replace=True)
    assert (local / "datasets/example/weights.bin").read_bytes() == b"replacement"


def test_pull_preserves_existing_destination_without_replace(monkeypatch, tmp_path):
    local = tmp_path / "local"
    (local / "assets").mkdir(parents=True)
    monkeypatch.setenv("ROBODOJO_STORAGE_ROOT", str(local))
    monkeypatch.setenv("ROBODOJO_S3_URI", "s3://bucket/robodojo")
    with pytest.raises(SystemExit, match="already exists"):
        storage_cli.pull("assets")


def test_status_publish_and_pull_aliases(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(storage_cli, "doctor", lambda: seen.append("status"))
    monkeypatch.setattr(
        storage_cli,
        "publish",
        lambda source, relative, **kwargs: seen.append((source, relative, kwargs)),
    )
    monkeypatch.setattr(storage_cli, "pull", lambda relative, **kwargs: seen.append((relative, kwargs)))

    assert storage_cli.main(["status"]) == 0
    assert storage_cli.main(["publish", str(tmp_path), "datasets/example", "--dry-run"]) == 0
    assert storage_cli.main(["pull", "datasets/example", "--dry-run"]) == 0
    assert seen[0] == "status"
    assert seen[1][1] == "datasets/example"
    assert seen[1][2]["dry_run"] is True
    assert seen[2] == ("datasets/example", {"replace": False, "dry_run": True})
