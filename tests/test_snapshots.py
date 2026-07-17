from hashlib import sha256
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
from typer.testing import CliRunner

from robodojo.cli import app
from robodojo.core.artifacts.snapshots import normalize_snapshot_summary
from robodojo.core.experiments.selection import resolve_recipe
from robodojo.core.gpu import GpuAssignment
from robodojo.core.models.reports import (
    SnapshotRecord,
    SnapshotSummary,
)
from robodojo.core.models.requests import (
    SnapshotBatchRequest,
    SnapshotCaptureRequest,
)
from robodojo.core.paths import RepositoryPaths
from robodojo.orchestration import snapshots as snapshot_orchestration
from robodojo.sim.scene_export.contracts import SCENE_EXPORT_FORMAT_VERSION, ExportIdentity
from robodojo.sim.scene_export.first_frame import (
    FirstFrameIdentity,
    capture_first_frame,
    completed_first_frame_matches,
)
from robodojo.workflows import snapshots
from robodojo.workflows.errors import StorageError
from robodojo.workflows.snapshot_gallery import render_snapshot_gallery

ROOT = Path(__file__).resolve().parents[1]
RECIPE = "pi05-arx_x5-default-fold_clothes"
RUNNER = CliRunner()


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _fake_env() -> SimpleNamespace:
    frame_a = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    frame_b = np.full((2, 2, 3), 91, dtype=np.uint8)
    return SimpleNamespace(
        recipe=RECIPE,
        experiment_hash="a" * 64,
        task_name="fold_clothes",
        task_protocol="fold_clothes",
        environment="arx_x5",
        scene="default",
        eval_seed=0,
        camera_rig=SimpleNamespace(
            cameras=[
                SimpleNamespace(observation_key="cam_head", role="top"),
                SimpleNamespace(observation_key="cam_wrist", role="wrist"),
            ]
        ),
        get_obs=lambda: {
            "instruction": "fold the cloth",
            "vision": {
                "cam_head": {"color": frame_a},
                "cam_wrist": {"color": frame_b},
            },
        },
    )


def test_first_frame_capture_writes_exact_camera_pixels_and_reuses(tmp_path):
    env = _fake_env()
    output = tmp_path / "first_frame"
    capture_first_frame(env, output, 0)

    assert np.array_equal(np.asarray(Image.open(output / "cam_head.png")), env.get_obs()["vision"]["cam_head"]["color"])
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    identity = FirstFrameIdentity(**metadata["identity"])
    assert metadata["snapshot_boundary"] == "post_reset_first_rollout_observation_pre_action"
    assert metadata["camera_order"] == ["cam_head", "cam_wrist"]
    assert metadata["instruction"] == "fold the cloth"
    assert completed_first_frame_matches(output, identity)
    assert capture_first_frame(env, output, 0) == output.resolve()

    (output / "cam_head.png").write_bytes(b"changed")
    assert not completed_first_frame_matches(output, identity)


def test_first_frame_capture_rejects_observation_without_rgb(tmp_path):
    env = _fake_env()
    env.get_obs = lambda: {"vision": {"cam_head": {}}}
    try:
        capture_first_frame(env, tmp_path / "first_frame", 0)
    except RuntimeError as exc:
        assert "has no RGB" in str(exc)
    else:
        raise AssertionError("capture should reject a camera without RGB")


def _capture_request(tmp_path: Path, **updates) -> SnapshotCaptureRequest:
    paths = RepositoryPaths.resolve(ROOT)
    values = {
        "experiment": resolve_recipe(paths, RECIPE).spec(paths),
        "environment_gpu": 0,
        "output_dir": tmp_path / RECIPE,
        "layout_id": 0,
        "export_scene": False,
        "run_id": "snapshot-test",
        "dry_run": True,
    }
    values.update(updates)
    return SnapshotCaptureRequest(**values)


def test_snapshot_orchestration_is_policy_free_and_propagates_scene_export(monkeypatch, tmp_path):
    launches = []
    monkeypatch.setattr(
        snapshot_orchestration,
        "resolve_gpus",
        lambda **selectors: GpuAssignment(env_gpu=3, env_source="explicit"),
    )
    monkeypatch.setattr(
        snapshot_orchestration,
        "run_simulator_session",
        lambda paths, request, environment: launches.append((request, environment)) or 0,
    )

    request = _capture_request(tmp_path, environment_gpu="auto", export_scene=True)
    assert snapshot_orchestration.run_snapshot_capture(RepositoryPaths.resolve(ROOT), request) == 0
    simulator, environment = launches[0]
    assert simulator.environment_gpu == 3
    assert simulator.port == 1
    assert environment["ROBODOJO_CAPTURE_FIRST_FRAME"] == "true"
    assert environment["ROBODOJO_EXPORT_SCENE"] == "true"
    assert environment["ROBODOJO_FIRST_FRAME_DIR"].endswith(f"{RECIPE}/first_frame")
    assert "ROBODOJO_EXPORT_SCENE_DIR" not in environment


def test_snapshots_cli_defaults_to_all_and_accepts_explicit_recipes(monkeypatch, tmp_path):
    requests = []
    monkeypatch.setattr(snapshots, "run_snapshot_batch", lambda paths, request: requests.append(request) or 0)

    default = RUNNER.invoke(app, ["snapshots", "--root", str(ROOT), "--dry-run"])
    selected = RUNNER.invoke(
        app,
        [
            "snapshots",
            "--recipe",
            RECIPE,
            "--layout-id",
            "2",
            "--env-gpu",
            "4",
            "--output-dir",
            str(tmp_path),
            "--export-scene",
            "--publish",
            "--root",
            str(ROOT),
        ],
    )

    assert default.exit_code == selected.exit_code == 0
    assert requests[0].recipes == ()
    assert requests[0].dry_run is True
    assert requests[0].publish is False
    assert requests[1].recipes == (RECIPE,)
    assert requests[1].layout_id == 2
    assert requests[1].environment_gpu == 4
    assert requests[1].export_scene is True
    assert requests[1].publish is True


def test_snapshots_cli_rejects_duplicate_recipe_selection():
    result = RUNNER.invoke(app, ["snapshots", "--recipe", RECIPE, "--recipe", RECIPE])
    assert result.exit_code == 2
    assert "recipe selections must be unique" in result.output


def _write_fake_capture(request: SnapshotCaptureRequest) -> int:
    first_frame = request.output_dir / "first_frame"
    first_frame.mkdir(parents=True)
    camera = first_frame / "cam_head.png"
    sheet = first_frame / "contact_sheet.png"
    Image.fromarray(np.full((2, 3, 3), 40, dtype=np.uint8)).save(camera)
    Image.fromarray(np.full((2, 3, 3), 50, dtype=np.uint8)).save(sheet)
    experiment = request.experiment
    metadata = {
        "format_version": 1,
        "complete": True,
        "identity": {
            "recipe": str(experiment.recipe),
            "contract_hash": str(experiment.experiment_hash),
            "task": experiment.task,
            "protocol": experiment.task_protocol,
            "profile": experiment.environment,
            "scene_config": experiment.scene,
            "seed": request.seed,
            "layout_id": request.layout_id,
        },
        "camera_order": ["cam_head"],
        "artifacts": {
            "contact_sheet": {"path": sheet.name, "sha256": _sha(sheet)},
            "cameras": {"cam_head": {"path": camera.name, "sha256": _sha(camera)}},
        },
    }
    (first_frame / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    if request.export_scene:
        scene = request.output_dir / "scene_snapshot"
        scene.mkdir()
        for name in ("scene_referenced.usda", "scene_flattened.usdc", "scene_preview.usdz"):
            (scene / name).write_bytes(name.encode())
        export_identity = ExportIdentity(
            task=experiment.task,
            task_protocol=experiment.task_protocol,
            episode_horizon=experiment.episode_horizon,
            evaluation_episodes=experiment.evaluation_episodes,
            recipe=experiment.recipe,
            experiment_hash=experiment.experiment_hash,
            environment=experiment.environment,
            scene=experiment.scene,
            seed=request.seed,
            layout_id=request.layout_id,
            repository_revision="abc123",
            environment_profile_hash="b" * 64,
            embodiment=experiment.embodiment,
            scene_profile_hash="c" * 64,
            layout_set_hash="d" * 64,
            scene_asset_hash="e" * 64,
        )
        manifest = {
            "format_version": SCENE_EXPORT_FORMAT_VERSION,
            "complete": True,
            "identity": export_identity.to_dict(),
            "artifacts": {
                "referenced_usda": {"path": "scene_referenced.usda", "sha256": "1" * 64},
                "flattened_usdc": {"path": "scene_flattened.usdc", "sha256": "2" * 64},
                "preview_usdz": {"path": "scene_preview.usdz", "sha256": "3" * 64},
            },
            "preview": {
                "preserved_materials": 1,
                "translated_materials": 1,
                "fallback_materials": 0,
                "missing_textures": [],
                "unsupported_inputs": [],
                "excluded_guide_meshes": 0,
                "approximation": "test",
            },
        }
        (scene / "scene_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return 0


def test_snapshot_batch_writes_gallery_and_resumes_exact_bundle(monkeypatch, tmp_path):
    calls = []
    published = []

    def capture(paths, request):
        calls.append(request)
        return _write_fake_capture(request)

    output = tmp_path / "batch"
    monkeypatch.setattr(snapshots, "run_snapshot_capture", capture)
    monkeypatch.setattr(snapshots, "_publication_prerequisites", lambda: 0)
    monkeypatch.setattr(
        snapshots,
        "publish_snapshot_run",
        lambda run_id, source: published.append((run_id, source)),
    )
    request = SnapshotBatchRequest(
        recipes=(RECIPE,),
        environment_gpu=0,
        output_dir=output,
        export_scene=True,
    )
    paths = RepositoryPaths.resolve(ROOT)
    assert snapshots.run_snapshot_batch(paths, request) == 0
    assert len(calls) == 1
    assert (output / RECIPE / "metadata.json").is_file()
    page = (output / "index.html").read_text(encoding="utf-8")
    assert RECIPE in page
    assert "scene_preview.usdz" in page
    assert "cam_head.png" in page

    resumed = request.model_copy(update={"resume": True, "publish": True})
    assert snapshots.run_snapshot_batch(paths, resumed) == 0
    assert len(calls) == 1
    assert published == [("batch", output.resolve())]
    summary = SnapshotSummary.model_validate_json((output / "summary.json").read_text(encoding="utf-8"))
    assert summary.results[0].status == "SKIP"


def test_zero_exit_without_artifacts_is_a_failed_snapshot(monkeypatch, tmp_path):
    published = []
    monkeypatch.setattr(snapshots, "_publication_prerequisites", lambda: 0)
    monkeypatch.setattr(
        snapshots,
        "publish_snapshot_run",
        lambda *args: published.append(args),
    )
    monkeypatch.setattr(snapshots, "run_snapshot_capture", lambda paths, request: 0)
    request = SnapshotBatchRequest(
        recipes=(RECIPE,),
        environment_gpu=0,
        output_dir=tmp_path / "empty",
        publish=True,
    )
    assert snapshots.run_snapshot_batch(RepositoryPaths.resolve(ROOT), request) == 1
    summary = SnapshotSummary.model_validate_json((tmp_path / "empty/summary.json").read_text(encoding="utf-8"))
    assert summary.results[0].status == "FAIL"
    assert "incomplete" in summary.results[0].message
    assert published == []


def test_snapshot_dry_run_reports_launch_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(snapshots, "run_snapshot_capture", lambda paths, request: 2)
    request = SnapshotBatchRequest(
        recipes=(RECIPE,),
        environment_gpu=0,
        output_dir=tmp_path / "dry-run",
        dry_run=True,
    )
    assert snapshots.run_snapshot_batch(RepositoryPaths.resolve(ROOT), request) == 2
    assert not request.output_dir.exists()


@pytest.mark.parametrize(
    ("remote", "aws_path", "message"),
    [
        (None, "/usr/bin/aws", "ROBODOJO_S3_URI"),
        ("https://bucket/robodojo", "/usr/bin/aws", "ROBODOJO_S3_URI"),
        ("s3://bucket/robodojo", None, "AWS CLI"),
    ],
)
def test_snapshot_publish_prerequisites_fail_before_capture(
    monkeypatch,
    tmp_path,
    caplog,
    remote,
    aws_path,
    message,
):
    monkeypatch.setattr(snapshots, "s3_uri", lambda: remote)
    monkeypatch.setattr(snapshots.shutil, "which", lambda name: aws_path)
    monkeypatch.setattr(
        snapshots,
        "run_snapshot_capture",
        lambda *args: pytest.fail("snapshot capture should not start"),
    )
    output = tmp_path / "publish-prerequisite"
    request = SnapshotBatchRequest(
        recipes=(RECIPE,),
        environment_gpu=0,
        output_dir=output,
        publish=True,
    )

    assert snapshots.run_snapshot_batch(RepositoryPaths.resolve(ROOT), request) == 2
    assert message in caplog.text
    assert not output.exists()


def test_snapshot_publish_dry_run_skips_prerequisites_and_publication(monkeypatch, tmp_path):
    captures = []
    monkeypatch.setattr(snapshots, "s3_uri", lambda: pytest.fail("S3 should not be inspected"))
    monkeypatch.setattr(
        snapshots.shutil,
        "which",
        lambda name: pytest.fail("AWS CLI should not be inspected"),
    )
    monkeypatch.setattr(
        snapshots,
        "publish_snapshot_run",
        lambda *args: pytest.fail("dry run should not publish"),
    )
    monkeypatch.setattr(
        snapshots,
        "run_snapshot_capture",
        lambda paths, request: captures.append(request) or 0,
    )
    output = tmp_path / "publish-dry-run"
    request = SnapshotBatchRequest(
        recipes=(RECIPE,),
        environment_gpu=0,
        output_dir=output,
        publish=True,
        dry_run=True,
    )

    assert snapshots.run_snapshot_batch(RepositoryPaths.resolve(ROOT), request) == 0
    assert len(captures) == 1
    assert not output.exists()


def test_successful_snapshot_batch_publishes_once(monkeypatch, tmp_path):
    published = []
    output = tmp_path / "published-batch"
    monkeypatch.setattr(snapshots, "_publication_prerequisites", lambda: 0)
    monkeypatch.setattr(snapshots, "run_snapshot_capture", lambda paths, request: _write_fake_capture(request))
    monkeypatch.setattr(
        snapshots,
        "publish_snapshot_run",
        lambda run_id, source: published.append((run_id, source)),
    )
    request = SnapshotBatchRequest(
        recipes=(RECIPE,),
        environment_gpu=0,
        output_dir=output,
        publish=True,
    )

    assert snapshots.run_snapshot_batch(RepositoryPaths.resolve(ROOT), request) == 0
    assert published == [("published-batch", output.resolve())]


@pytest.mark.parametrize(
    ("failure", "expected_code", "message"),
    [
        (StorageError("remote destination is already complete"), 1, "already complete"),
        (subprocess.CalledProcessError(5, ["aws"], stderr="access denied"), 5, "access denied"),
        (OSError("aws executable failed"), 1, "aws executable failed"),
    ],
)
def test_snapshot_publication_failure_preserves_local_batch(
    monkeypatch,
    tmp_path,
    caplog,
    failure,
    expected_code,
    message,
):
    output = tmp_path / "publish-failure"
    monkeypatch.setattr(snapshots, "_publication_prerequisites", lambda: 0)
    monkeypatch.setattr(snapshots, "run_snapshot_capture", lambda paths, request: _write_fake_capture(request))

    def fail_publish(run_id, source):
        raise failure

    monkeypatch.setattr(snapshots, "publish_snapshot_run", fail_publish)
    request = SnapshotBatchRequest(
        recipes=(RECIPE,),
        environment_gpu=0,
        output_dir=output,
        publish=True,
    )

    assert snapshots.run_snapshot_batch(RepositoryPaths.resolve(ROOT), request) == expected_code
    assert message in caplog.text
    assert (output / "summary.json").is_file()
    assert (output / RECIPE / "first_frame/contact_sheet.png").is_file()


def test_gallery_escapes_failure_messages_and_requires_no_network(tmp_path):
    record = SnapshotRecord(
        status="FAIL",
        recipe=RECIPE,
        policy="pi05_arx_x5",
        environment="arx_x5",
        scene="default",
        task_protocol="fold_clothes",
        task="fold_clothes",
        experiment_hash="a" * 64,
        output_dir=str(tmp_path / RECIPE),
        message="<script>alert(1)</script>",
    )
    summary = SnapshotSummary(
        run_id="gallery",
        output_dir=str(tmp_path),
        seed=0,
        layout_id=0,
        export_scene=False,
        recipes=(RECIPE,),
        results=[record],
    )
    page = render_snapshot_gallery(summary)
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "https://" not in page
    assert "prefers-reduced-motion" in page


def test_v1_snapshot_summary_is_normalized_in_memory():
    legacy = {
        "run_id": "legacy",
        "output_dir": "/tmp/legacy",
        "seed": 0,
        "layout_id": 0,
        "export_scene": False,
        "recipes": [RECIPE],
        "complete": True,
        "results": [
            {
                "status": "PASS",
                "recipe": RECIPE,
                "policy": "pi05_arx_x5",
                "environment": "arx_x5",
                "scene": "default",
                "protocol": "fold_clothes",
                "task": "fold_clothes",
                "contract_hash": "a" * 64,
                "exit_code": 0,
                "output_dir": f"/tmp/legacy/{RECIPE}",
            }
        ],
    }

    summary = SnapshotSummary.model_validate(normalize_snapshot_summary(legacy))

    assert summary.format_version == 2
    assert summary.results[0].task_protocol == "fold_clothes"
    assert summary.results[0].experiment_hash == "a" * 64
    assert "format_version" not in legacy
    assert "contract_hash" in legacy["results"][0]
