from dataclasses import fields
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from robodojo.core.artifacts.scene_exports import SceneExportArtifactError, require_completed_scene_export
from robodojo.core.models.experiment import ExperimentSpec
from robodojo.sim.scene_export.contracts import (
    SCENE_EXPORT_FORMAT_VERSION,
    ExportIdentity,
    calculate_fisheye_fov_degrees,
    calculate_fov_degrees,
    completed_export_matches,
    package_member_exists,
    scene_input_paths,
    split_package_asset_path,
)

ROOT = Path(__file__).resolve().parents[1]


def _identity(**overrides) -> ExportIdentity:
    values = {
        "task": "fold_clothes",
        "task_protocol": "fold_clothes",
        "episode_horizon": 500,
        "evaluation_episodes": 25,
        "recipe": "pi05-arx_x5-default-fold_clothes",
        "experiment_hash": "d" * 64,
        "environment": "arx_x5",
        "scene": "default",
        "seed": 0,
        "layout_id": 0,
        "repository_revision": "abc123",
        "environment_profile_hash": "e" * 64,
        "embodiment": "arx_x5",
        "scene_profile_hash": "a" * 64,
        "layout_set_hash": "b" * 64,
        "scene_asset_hash": "c" * 64,
    }
    values.update(overrides)
    return ExportIdentity(**values)


def _complete_manifest(identity: ExportIdentity, *, scene_export_only: bool = False) -> dict[str, object]:
    return {
        "format_version": SCENE_EXPORT_FORMAT_VERSION,
        "complete": True,
        "scene_export_only": scene_export_only,
        "identity": identity.to_dict(),
        "artifacts": {
            "referenced_usda": {"path": "scene_referenced.usda", "sha256": "a" * 64},
            "flattened_usdc": {"path": "scene_flattened.usdc", "sha256": "b" * 64},
            "preview_usdz": {"path": "scene_preview.usdz", "sha256": "c" * 64},
        },
        "preview": {
            "preserved_materials": 1,
            "translated_materials": 1,
            "fallback_materials": 1,
            "missing_textures": [],
            "unsupported_inputs": [],
            "excluded_guide_meshes": 0,
            "approximation": "portable approximation",
        },
    }


def test_camera_fov_contract():
    fov = calculate_fov_degrees(640, 480, 320, 320)
    assert fov["horizontal"] == pytest.approx(90.0)
    assert fov["vertical"] == pytest.approx(73.739795, abs=1e-6)
    assert fov["diagonal"] == pytest.approx(102.680383, abs=1e-6)
    with pytest.raises(ValueError, match="positive"):
        calculate_fov_degrees(640, 480, 0, 320)


def test_openarm_fisheye_intrinsics_match_fitted_and_published_diagonal_fov():
    base = calculate_fisheye_fov_degrees(640, 480, 416.7350, 416.7350, [0.0] * 4)
    wrist = calculate_fisheye_fov_degrees(1280, 720, 824.9654, 824.9654, [0.0] * 4)
    assert base["diagonal"] == pytest.approx(110.0, abs=0.1)
    assert wrist["diagonal"] == pytest.approx(102.0, abs=0.1)


def test_completed_export_requires_exact_identity(tmp_path):
    identity = _identity(environment="openarm_wowrobo_v1_1")
    assert not completed_export_matches(tmp_path, identity)
    (tmp_path / "scene_manifest.json").write_text(
        json.dumps(_complete_manifest(identity)),
        encoding="utf-8",
    )
    assert not completed_export_matches(tmp_path, identity)
    for name in ("scene_referenced.usda", "scene_flattened.usdc", "scene_preview.usdz"):
        (tmp_path / name).touch()
    assert completed_export_matches(tmp_path, identity)
    assert not completed_export_matches(
        tmp_path,
        _identity(environment="openarm_wowrobo_v1_1", layout_id=1),
    )
    assert not completed_export_matches(
        tmp_path,
        _identity(environment="openarm_wowrobo_v1_1", scene="molmo_yam"),
    )
    assert not completed_export_matches(
        tmp_path,
        _identity(environment="openarm_wowrobo_v1_1", layout_set_hash="d" * 64),
    )
    assert not completed_export_matches(
        tmp_path,
        _identity(environment="openarm_wowrobo_v1_1", scene_asset_hash="e" * 64),
    )


def test_scene_only_completion_requires_explicit_manifest_marker(tmp_path):
    identity = _identity()
    for name in ("scene_referenced.usda", "scene_flattened.usdc", "scene_preview.usdz"):
        (tmp_path / name).touch()
    (tmp_path / "scene_manifest.json").write_text(
        json.dumps(_complete_manifest(identity, scene_export_only=True)),
        encoding="utf-8",
    )

    manifest = require_completed_scene_export(tmp_path, require_scene_export_only=True)
    assert manifest["scene_export_only"] is True
    assert completed_export_matches(tmp_path, identity, scene_export_only=True)
    assert not completed_export_matches(tmp_path, identity, scene_export_only=False)

    manifest["scene_export_only"] = False
    (tmp_path / "scene_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SceneExportArtifactError, match="not marked scene_export_only"):
        require_completed_scene_export(tmp_path, require_scene_export_only=True)


def test_scene_export_v8_identity_has_no_layout_selector():
    assert SCENE_EXPORT_FORMAT_VERSION == 8
    assert "layout" not in {field.name for field in fields(ExportIdentity)}
    assert "layout" not in _identity().to_dict()


def test_completed_export_rejects_incomplete_v8_manifest(tmp_path):
    identity = _identity()
    for name in ("scene_referenced.usda", "scene_flattened.usdc", "scene_preview.usdz"):
        (tmp_path / name).touch()
    incomplete = _complete_manifest(identity)
    del incomplete["preview"]
    (tmp_path / "scene_manifest.json").write_text(json.dumps(incomplete), encoding="utf-8")
    assert not completed_export_matches(tmp_path, identity)


def test_completed_export_rejects_legacy_manifest(tmp_path):
    identity = _identity()
    for name in ("scene_referenced.usda", "scene_flattened.usdc", "scene_preview.usdz"):
        (tmp_path / name).touch()
    (tmp_path / "scene_manifest.json").write_text(
        json.dumps(
            {
                **_complete_manifest(identity),
                "format_version": 6,
                "identity": {**identity.to_dict(), "layout": "fold_clothes"},
                "layout_name": "fold_clothes",
            }
        ),
        encoding="utf-8",
    )
    assert not completed_export_matches(tmp_path, identity)


def test_completed_export_normalizes_v7_manifest_in_memory(tmp_path):
    identity = _identity()
    legacy_identity = identity.to_dict()
    for new, old in {
        "task_protocol": "protocol",
        "evaluation_episodes": "native_eval_num",
        "experiment_hash": "contract_hash",
        "environment": "profile",
        "scene": "scene_config",
        "embodiment": "policy_contract",
    }.items():
        legacy_identity[old] = legacy_identity.pop(new)
    manifest = _complete_manifest(identity)
    manifest["format_version"] = 7
    manifest["identity"] = legacy_identity
    (tmp_path / "scene_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in ("scene_referenced.usda", "scene_flattened.usdc", "scene_preview.usdz"):
        (tmp_path / name).touch()

    assert completed_export_matches(tmp_path, identity)
    assert json.loads((tmp_path / "scene_manifest.json").read_text(encoding="utf-8"))["format_version"] == 7


def test_package_asset_member_validation(tmp_path):
    import zipfile

    package = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("textures/base_color.png", b"png")

    assert split_package_asset_path("asset.usdz[textures/base_color.png]") == (
        "asset.usdz",
        "textures/base_color.png",
    )
    assert split_package_asset_path("texture.png") == ("texture.png", "")
    assert package_member_exists(package, "textures/base_color.png")
    assert not package_member_exists(package, "textures/missing.png")


def test_scene_export_inputs_follow_canonical_config_domains():
    paths = scene_input_paths(
        ROOT,
        "arx_x5",
        "default",
        "stack_bowls",
        {
            "sim": "sim_config",
            "robot": "dual_x5",
            "camera": "camera_config",
        },
    )

    assert {str(path.relative_to(ROOT)) for path in paths} == {
        "configs/environment/arx_x5.yml",
        "configs/camera/camera_config.yml",
        "configs/scene/profiles/default.yml",
        "configs/scene/components/default.yml",
        "configs/robot/dual_x5.yml",
        "configs/sim/sim_config.yml",
        "configs/task/stack_bowls.yml",
    }


def test_scene_only_eval_dry_run_bypasses_policy_orchestrator():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "robodojo.cli",
            "eval",
            "run",
            "--recipe",
            "pi05-arx_x5-default-fold_clothes",
            "--env-gpu",
            "0",
            "--seed",
            "0",
            "--layout-id",
            "0",
            "--export-scene-only",
            "--publish",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "robodojo.sim.evaluation.main" in result.stdout
    assert "ROBODOJO_EXPORT_SCENE_ONLY=true" in result.stdout
    assert "ROBODOJO_EXPORT_SCENE_DIR" not in result.stdout
    assert "setup_eval_policy_server.sh" not in result.stdout


def test_scene_visual_audit_dry_run_is_propagated_only_through_scene_only_path():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "robodojo.cli",
            "eval",
            "run",
            "--recipe",
            "pi05-arx_x5-default-fold_clothes",
            "--env-gpu",
            "0",
            "--export-scene-only",
            "--dry-run",
        ],
        cwd=ROOT,
        env={**os.environ, "ROBODOJO_SCENE_VISUAL_AUDIT": "1"},
        check=True,
        capture_output=True,
        text=True,
    )
    assert "ROBODOJO_SCENE_VISUAL_AUDIT=1" in result.stdout
    assert "setup_eval_policy_server.sh" not in result.stdout


def test_scene_visual_audit_rejects_non_scene_only_evaluation(monkeypatch, tmp_path):
    from robodojo.core.models.requests import EvaluationRequest
    from robodojo.core.paths import RepositoryPaths
    from robodojo.orchestration import evaluation

    policy_dir = tmp_path / "TestPolicy"
    policy_dir.mkdir()
    monkeypatch.setenv("ROBODOJO_SCENE_VISUAL_AUDIT", "1")
    request = EvaluationRequest(
        experiment=ExperimentSpec(
            policy_dir=policy_dir,
            task="fold_clothes",
            checkpoint="folding_final",
            policy_profile="test-policy",
            policy_runtime="test-policy-env",
            environment="arx_x5",
            embodiment="arx_x5",
            scene="default",
            action_type="joint",
            task_protocol="fold_clothes",
            episode_horizon=500,
            evaluation_episodes=25,
        ),
        export_scene=True,
    )
    with pytest.raises(ValueError, match="valid only with --export-scene-only"):
        evaluation.run_evaluation(RepositoryPaths.resolve(ROOT), request)


def test_export_and_continue_keeps_policy_orchestrator():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "robodojo.cli",
            "eval",
            "run",
            "--recipe",
            "pi05-arx_x5-default-fold_clothes",
            "--policy-gpu",
            "0",
            "--env-gpu",
            "1",
            "--export-scene",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "setup_eval_policy_server.sh" in result.stdout
    assert "robodojo.sim.evaluation.main" in result.stdout
