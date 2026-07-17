import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from robodojo.cli import app
from robodojo.core.artifacts.scene_exports import SCENE_EXPORT_FORMAT_VERSION, ExportIdentity
from robodojo.core.models.reports import (
    SnapshotRecord,
    SnapshotSummary,
)
from robodojo.workflows import storage as storage_cli
from robodojo.workflows.errors import StorageError

RUNNER = CliRunner()


def _result_payload(**overrides):
    payload = {
        "artifact_schema_version": 3,
        "task_name": "general_pickup",
        "protocol_name": "moonlake_office_general_pickup",
        "episode_horizon": 400,
        "native_eval_num": 50,
        "robodojo_revision": "r" * 40,
        "xpolicylab_revision": "x" * 40,
        "policy_profile": "pi05_bimanual_yam_pickup",
        "policy_descriptor_hash": "d" * 64,
        "environment_profile": "bimanual_yam_moonlake_office",
        "environment_profile_hash": "e" * 64,
        "environment_asset_hash": "f" * 64,
        "policy_contract": "bimanual_yam",
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


def _write_scene_only_export(
    run: Path,
    *,
    directory: str = "scene_snapshot",
    scene_export_only: bool | None = True,
    complete: bool = True,
    format_version: int = SCENE_EXPORT_FORMAT_VERSION,
) -> Path:
    scene = run / directory
    scene.mkdir(parents=True)
    for filename in ("scene_referenced.usda", "scene_flattened.usdc", "scene_preview.usdz"):
        (scene / filename).write_bytes(filename.encode())
    manifest = {
        "format_version": format_version,
        "complete": complete,
        "identity": ExportIdentity(
            task="general_pickup",
            task_protocol="moonlake_office_general_pickup",
            episode_horizon=400,
            evaluation_episodes=50,
            recipe="pi05-bimanual_yam-moonlake_office-general_pickup",
            experiment_hash="d" * 64,
            environment="bimanual_yam_moonlake_office",
            scene="moonlake_office",
            seed=0,
            layout_id=0,
            repository_revision="r" * 40,
            environment_profile_hash="e" * 64,
            embodiment="bimanual_yam",
            scene_profile_hash="f" * 64,
            layout_set_hash="a" * 64,
            scene_asset_hash="c" * 64,
        ).to_dict(),
        "artifacts": {
            "referenced_usda": {"path": "scene_referenced.usda", "sha256": "a" * 64},
            "flattened_usdc": {"path": "scene_flattened.usdc", "sha256": "b" * 64},
            "preview_usdz": {"path": "scene_preview.usdz", "sha256": "c" * 64},
        },
        "preview": {
            "preserved_materials": 0,
            "translated_materials": 0,
            "fallback_materials": 0,
            "missing_textures": [],
            "unsupported_inputs": [],
            "excluded_guide_meshes": 0,
            "approximation": "portable approximation",
        },
    }
    if scene_export_only is not None:
        manifest["scene_export_only"] = scene_export_only
    (scene / "scene_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return scene


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
    for command in ("doctor", "publish", "pull", "publish-eval"):
        assert command in result.stdout
    for removed in (
        "status",
        "publish-assets",
        "publish-data",
        "publish-checkpoint",
        "publish-model",
        "publish-reference-cache",
        "publish-run",
    ):
        assert removed not in result.stdout


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


def test_publish_scene_only_evaluation_without_result(monkeypatch, tmp_path):
    source = tmp_path / "scene-only-run"
    _write_scene_only_export(source)
    monkeypatch.setenv("ROBODOJO_S3_URI", "s3://bucket/robodojo")
    monkeypatch.setenv("ROBODOJO_STORAGE_ROOT", str(tmp_path / "local"))
    calls = []

    def fake_aws(*args, check=True):
        calls.append(args)
        return subprocess.CompletedProcess(args, 1 if args[:2] == ("s3api", "head-object") else 0, "", "")

    monkeypatch.setattr(storage_cli, "_aws", fake_aws)
    storage_cli.publish(source, "runs/eval_result/RoboDojo/task/scene-only-run")

    assert calls[1][:4] == (
        "s3",
        "sync",
        str(source.resolve()),
        "s3://bucket/robodojo/runs/eval_result/RoboDojo/task/scene-only-run",
    )
    assert not any(any(str(value).endswith("/_result.json") for value in call) for call in calls)
    assert calls[-1][-2].endswith("/_COMPLETE.json")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scene_export_only": None}, "not marked scene_export_only"),
        ({"scene_export_only": False}, "not marked scene_export_only"),
        ({"complete": False}, "manifest is incomplete"),
        ({"format_version": 6}, "unsupported scene export format"),
        ({"directory": "custom_scene"}, "manifest is missing or invalid"),
    ],
)
def test_scene_only_evaluation_publication_rejects_invalid_completion(monkeypatch, tmp_path, kwargs, message):
    source = tmp_path / "invalid-scene-run"
    _write_scene_only_export(source, **kwargs)
    monkeypatch.setenv("ROBODOJO_S3_URI", "s3://bucket/robodojo")

    with pytest.raises(StorageError, match=message):
        storage_cli.publish(source, "runs/eval_result/RoboDojo/task/invalid-scene-run", dry_run=True)


def test_invalid_result_never_falls_back_to_completed_scene_export(monkeypatch, tmp_path):
    source = tmp_path / "invalid-result-run"
    _write_scene_only_export(source)
    (source / "_result.json").write_text(json.dumps({"eval_time": 0}), encoding="utf-8")
    monkeypatch.setenv("ROBODOJO_S3_URI", "s3://bucket/robodojo")

    with pytest.raises(StorageError, match="artifact_schema_version mismatch"):
        storage_cli.publish(source, "runs/eval_result/RoboDojo/task/invalid-result-run", dry_run=True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("artifact_schema_version"), "artifact_schema_version mismatch"),
        (lambda payload: payload.update(artifact_schema_version=2), "artifact_schema_version mismatch"),
        (lambda payload: payload.update(layout_name="general_pickup"), "removed layout_name selector"),
        (lambda payload: payload.update(details={}), "incomplete episode details"),
    ],
)
def test_evaluation_publication_rejects_unsupported_or_incomplete_results(
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

    with pytest.raises(StorageError, match=message):
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

    with pytest.raises(StorageError, match="artifact_schema_version mismatch"):
        storage_cli.publish_evaluation_run("legacy-run", dry_run=True)


def test_publish_evaluation_run_discovers_scene_only_manifest(monkeypatch, tmp_path):
    storage = tmp_path / "storage"
    run = (
        storage
        / "runs/eval_result/RoboDojo/general_pickup/TestPolicy/bimanual_yam"
        / "0_ckpt_name=test,action_type=joint/scene-run"
    )
    _write_scene_only_export(run)
    monkeypatch.setenv("ROBODOJO_STORAGE_ROOT", str(storage))
    published = []
    monkeypatch.setattr(
        storage_cli,
        "publish",
        lambda path, relative, **kwargs: published.append((path, relative, kwargs)),
    )

    storage_cli.publish_evaluation_run("scene-run", dry_run=True)

    assert published == [
        (
            run,
            (
                "runs/eval_result/RoboDojo/general_pickup/TestPolicy/bimanual_yam/"
                "0_ckpt_name=test,action_type=joint/scene-run"
            ),
            {"replace": False, "dry_run": True},
        )
    ]


def test_publish_snapshot_run_requires_success_and_uses_snapshot_destination(monkeypatch, tmp_path):
    source = tmp_path / "snapshot-run"
    source.mkdir()
    record = SnapshotRecord(
        status="PASS",
        recipe="pi05-arx_x5-default-fold_clothes",
        policy="pi05_arx_x5",
        environment="arx_x5",
        scene="default",
        task_protocol="fold_clothes",
        task="fold_clothes",
        experiment_hash="a" * 64,
        exit_code=0,
        output_dir=str(source / "pi05-arx_x5-default-fold_clothes"),
    )
    summary = SnapshotSummary(
        run_id="snapshot-run",
        output_dir=str(source.resolve()),
        seed=0,
        layout_id=0,
        export_scene=False,
        recipes=(record.recipe,),
        complete=True,
        results=[record],
    )
    (source / "summary.json").write_text(summary.model_dump_json(), encoding="utf-8")
    published = []
    monkeypatch.setattr(
        storage_cli,
        "publish",
        lambda path, relative, **kwargs: published.append((path, relative, kwargs)),
    )

    storage_cli.publish_snapshot_run("snapshot-run", source)
    assert published == [(source.resolve(), "runs/snapshots/snapshot-run", {"replace": False, "dry_run": False})]

    failed = summary.model_copy(update={"results": [record.model_copy(update={"status": "FAIL", "exit_code": 1})]})
    (source / "summary.json").write_text(failed.model_dump_json(), encoding="utf-8")
    with pytest.raises(StorageError, match="not complete and successful"):
        storage_cli.publish_snapshot_run("snapshot-run", source)


def test_publish_rejects_completed_destination(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setenv("ROBODOJO_S3_URI", "s3://bucket/robodojo")

    def already_complete(*args, check=True):
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(storage_cli, "_aws", already_complete)
    with pytest.raises(StorageError, match="already complete"):
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
    with pytest.raises(StorageError, match="unsupported symlink"):
        storage_cli._payload_files(source)


def test_destination_cannot_escape_prefix(monkeypatch):
    monkeypatch.setenv("ROBODOJO_S3_URI", "s3://bucket/robodojo")
    with pytest.raises(StorageError, match="invalid storage destination"):
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
    with pytest.raises(StorageError, match="already exists"):
        storage_cli.pull("assets")


def test_publish_evaluation_accepts_source_or_run_id_but_not_both(monkeypatch, tmp_path):
    local = tmp_path / "local"
    source = local / "runs/eval_result/RoboDojo/task/run"
    source.mkdir(parents=True)
    monkeypatch.setenv("ROBODOJO_STORAGE_ROOT", str(local))
    published = []
    monkeypatch.setattr(
        storage_cli,
        "publish",
        lambda path, relative, **kwargs: published.append((path, relative, kwargs)),
    )

    storage_cli.publish_evaluation(source, dry_run=True)

    assert published == [
        (
            source.resolve(),
            "runs/eval_result/RoboDojo/task/run",
            {"replace": False, "dry_run": True},
        )
    ]
    with pytest.raises(StorageError, match="mutually exclusive"):
        storage_cli.publish_evaluation(source, run_id="run")


def test_storage_domain_errors_are_rendered_without_tracebacks(tmp_path):
    missing = tmp_path / "missing"
    result = RUNNER.invoke(
        app,
        ["storage", "publish", str(missing), "assets"],
        env={"ROBODOJO_S3_URI": "s3://bucket/robodojo"},
    )

    assert result.exit_code == 1
    assert "publish source is not a directory" in result.stderr
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (subprocess.CalledProcessError(5, ["aws"], stderr="access denied"), "AWS command failed: access denied"),
        (OSError("aws is unavailable"), "could not run AWS CLI: aws is unavailable"),
    ],
)
def test_aws_failures_are_storage_errors(monkeypatch, failure, message):
    monkeypatch.setattr(storage_cli.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(failure))

    with pytest.raises(StorageError, match=message):
        storage_cli._aws("s3", "ls")


def test_publish_evaluation_cli_rejects_conflicting_selectors(tmp_path):
    result = RUNNER.invoke(
        app,
        ["storage", "publish-eval", "--source", str(tmp_path), "--run-id", "run"],
    )

    assert result.exit_code == 1
    assert "mutually exclusive" in result.stderr
    assert "Traceback" not in result.output
