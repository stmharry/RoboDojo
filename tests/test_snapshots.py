from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from robodojo.cli import app
from robodojo.core.contracts import resolve_recipe
from robodojo.core.gpu import GpuAssignment
from robodojo.core.models import (
    SnapshotBatchRequest,
    SnapshotCaptureRequest,
    SnapshotRecord,
    SnapshotSummary,
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
        recipe_name=RECIPE,
        contract_hash="a" * 64,
        task_name="fold_clothes",
        protocol_name="fold_clothes",
        config_name="arx_x5",
        scene_config="default",
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
        **resolve_recipe(paths, RECIPE).request_values(paths),
        "env_gpu": 0,
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

    request = _capture_request(tmp_path, env_gpu="auto", export_scene=True)
    assert snapshot_orchestration.run_snapshot_capture(RepositoryPaths.resolve(ROOT), request) == 0
    simulator, environment = launches[0]
    assert simulator.env_gpu == 3
    assert simulator.port == 1
    assert environment["ROBODOJO_CAPTURE_FIRST_FRAME"] == "true"
    assert environment["ROBODOJO_EXPORT_SCENE"] == "true"
    assert environment["ROBODOJO_EXPORT_SCENE_DIR"].endswith(f"{RECIPE}/scene_snapshot")


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
            "--root",
            str(ROOT),
        ],
    )

    assert default.exit_code == selected.exit_code == 0
    assert requests[0].recipes == ()
    assert requests[0].dry_run is True
    assert requests[1].recipes == (RECIPE,)
    assert requests[1].layout_id == 2
    assert requests[1].env_gpu == 4
    assert requests[1].export_scene is True


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
    identity = FirstFrameIdentity(
        recipe=str(request.recipe),
        contract_hash=str(request.contract_hash),
        task=request.task,
        protocol=request.protocol,
        profile=request.env_config,
        scene_config=request.scene_config,
        seed=request.seed,
        layout_id=request.layout_id,
    )
    metadata = {
        "format_version": 1,
        "complete": True,
        "identity": identity.to_dict(),
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
            task=request.task,
            protocol=request.protocol,
            episode_horizon=request.episode_horizon,
            native_eval_num=request.native_eval_num,
            recipe=request.recipe,
            contract_hash=request.contract_hash,
            profile=request.env_config,
            scene_config=request.scene_config,
            seed=request.seed,
            layout_id=request.layout_id,
            repository_revision="abc123",
            environment_profile_hash="b" * 64,
            policy_contract=request.policy_contract,
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

    def capture(paths, request):
        calls.append(request)
        return _write_fake_capture(request)

    output = tmp_path / "batch"
    monkeypatch.setattr(snapshots, "run_snapshot_capture", capture)
    request = SnapshotBatchRequest(
        recipes=(RECIPE,),
        env_gpu=0,
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

    resumed = request.model_copy(update={"resume": True})
    assert snapshots.run_snapshot_batch(paths, resumed) == 0
    assert len(calls) == 1
    summary = SnapshotSummary.model_validate_json((output / "summary.json").read_text(encoding="utf-8"))
    assert summary.results[0].status == "SKIP"


def test_zero_exit_without_artifacts_is_a_failed_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(snapshots, "run_snapshot_capture", lambda paths, request: 0)
    request = SnapshotBatchRequest(recipes=(RECIPE,), env_gpu=0, output_dir=tmp_path / "empty")
    assert snapshots.run_snapshot_batch(RepositoryPaths.resolve(ROOT), request) == 1
    summary = SnapshotSummary.model_validate_json((tmp_path / "empty/summary.json").read_text(encoding="utf-8"))
    assert summary.results[0].status == "FAIL"
    assert "incomplete" in summary.results[0].message


def test_snapshot_dry_run_reports_launch_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(snapshots, "run_snapshot_capture", lambda paths, request: 2)
    request = SnapshotBatchRequest(
        recipes=(RECIPE,),
        env_gpu=0,
        output_dir=tmp_path / "dry-run",
        dry_run=True,
    )
    assert snapshots.run_snapshot_batch(RepositoryPaths.resolve(ROOT), request) == 2
    assert not request.output_dir.exists()


def test_gallery_escapes_failure_messages_and_requires_no_network(tmp_path):
    record = SnapshotRecord(
        status="FAIL",
        recipe=RECIPE,
        policy="pi05_arx_x5",
        environment="arx_x5",
        scene="default",
        protocol="fold_clothes",
        task="fold_clothes",
        contract_hash="a" * 64,
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


def test_first_frame_hook_is_after_reset_and_before_rollout():
    source = (ROOT / "src/robodojo/sim/evaluation/main.py").read_text(encoding="utf-8")
    reset = source.index("env.reset(seed=env.env_seeds)")
    scene_export = source.index("export_scene_snapshot(env, export_dir", reset)
    first_frame = source.index("capture_first_frame(env, FIRST_FRAME_CAPTURE_DIR", scene_export)
    rollout = source.index("env.run_eval()", first_frame)
    assert reset < scene_export < first_frame < rollout
