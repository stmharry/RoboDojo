import os
from pathlib import Path
import subprocess

import pytest

from robodojo.workflows import storage as storage_cli


def test_storage_cli_help_runs_through_typer():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    result = subprocess.run(
        [str(root / ".venv/bin/robodojo"), "storage", "--help"],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert "publish-checkpoint" in result.stdout
    assert "publish-checkpoint" in result.stdout


def test_publish_uses_canonical_destination_and_completion_is_last(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "video.mp4").write_bytes(b"video")
    (source / "_result.json").write_text('{"eval_time": 1}\n', encoding="utf-8")
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("ROBODOJO_S3_URI", "s3://bucket/robodojo")
    monkeypatch.setenv("ROBODOJO_LOCAL_SCRATCH_ROOT", str(scratch))
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


def test_link_payload_creates_idempotent_local_link(tmp_path):
    source = tmp_path / "mount" / "assets"
    source.mkdir(parents=True)
    destination = tmp_path / "repo" / "Assets"

    storage_cli.link_payload(source, destination)
    assert destination.is_symlink()
    assert destination.resolve() == source.resolve()
    storage_cli.link_payload(source, destination)


def test_link_payload_refuses_real_directory_and_mismatched_link(tmp_path):
    source = tmp_path / "mount" / "datasets"
    source.mkdir(parents=True)
    real_destination = tmp_path / "data"
    real_destination.mkdir()
    with pytest.raises(SystemExit, match="refusing to replace existing path"):
        storage_cli.link_payload(source, real_destination)

    other = tmp_path / "other"
    other.mkdir()
    link_destination = tmp_path / "dataset-link"
    link_destination.symlink_to(other, target_is_directory=True)
    with pytest.raises(SystemExit, match="refusing to replace existing symlink"):
        storage_cli.link_payload(source, link_destination)


def test_status_publish_and_hydrate_aliases(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(storage_cli, "doctor", lambda: seen.append("status"))
    monkeypatch.setattr(
        storage_cli,
        "publish",
        lambda source, relative, **kwargs: seen.append((source, relative, kwargs)),
    )
    monkeypatch.setattr(
        storage_cli,
        "materialize",
        lambda source, destination: seen.append((source, destination)),
    )

    assert storage_cli.main(["status"]) == 0
    assert storage_cli.main(["publish", str(tmp_path), "datasets/example", "--dry-run"]) == 0
    assert storage_cli.main(["hydrate", str(tmp_path / "source"), str(tmp_path / "dest")]) == 0
    assert seen[0] == "status"
    assert seen[1][1] == "datasets/example"
    assert seen[1][2]["dry_run"] is True


@pytest.mark.parametrize(
    ("arguments", "relative_source"),
    [
        (["assets"], "assets"),
        (["datasets"], "datasets"),
        (["checkpoint", "--policy", "SmolVLA", "--checkpoint", "run-1"], "model_weights/SmolVLA/run-1"),
    ],
)
def test_link_cli_resolves_supported_durable_sources(monkeypatch, tmp_path, arguments, relative_source):
    mount = tmp_path / "mount"
    monkeypatch.setenv("ROBODOJO_STORAGE_ROOT", str(mount))
    seen = []
    monkeypatch.setattr(storage_cli, "link_payload", lambda source, destination: seen.append((source, destination)))
    destination = tmp_path / "destination"

    command = ["link", arguments[0], str(destination), *arguments[1:]]
    assert storage_cli.main(command) == 0
    assert seen == [(mount / relative_source, destination)]
