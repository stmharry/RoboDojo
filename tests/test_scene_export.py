import json
from pathlib import Path
import subprocess

import pytest

from robodojo.sim.scene_export.contracts import (
    ExportIdentity,
    calculate_fisheye_fov_degrees,
    calculate_fov_degrees,
    completed_export_matches,
)

ROOT = Path(__file__).resolve().parents[1]


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
    identity = ExportIdentity("fold_clothes", "openarm_wowrobo_v1_1", 0, 0, "abc123")
    assert not completed_export_matches(tmp_path, identity)
    (tmp_path / "scene_manifest.json").write_text(
        json.dumps({"complete": True, "identity": identity.to_dict()}), encoding="utf-8"
    )
    assert completed_export_matches(tmp_path, identity)
    assert not completed_export_matches(
        tmp_path,
        ExportIdentity("fold_clothes", "openarm_wowrobo_v1_1", 0, 1, "abc123"),
    )


def test_scene_only_eval_dry_run_bypasses_policy_orchestrator(tmp_path):
    policy_dir = tmp_path / "TestPolicy"
    policy_dir.mkdir()
    (policy_dir / "setup_eval_policy_server.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    result = subprocess.run(
        [
            str(ROOT / ".venv/bin/robodojo"),
            "eval",
            "--policy-dir",
            str(policy_dir),
            "--task",
            "fold_clothes",
            "--ckpt",
            "folding_final",
            "--policy-env",
            "unused-in-scene-only",
            "--env-cfg",
            "arx_x5",
            "--seed",
            "0",
            "--layout-id",
            "0",
            "--export-scene-only",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "robodojo.sim.evaluation.main" in result.stdout
    assert "ROBODOJO_EXPORT_SCENE_ONLY=true" in result.stdout
    assert "setup_eval_policy_server.sh" not in result.stdout


def test_export_and_continue_keeps_policy_orchestrator(tmp_path):
    policy_dir = tmp_path / "TestPolicy"
    policy_dir.mkdir()
    (policy_dir / "setup_eval_policy_server.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    result = subprocess.run(
        [
            str(ROOT / ".venv/bin/robodojo"),
            "eval",
            "--policy-dir",
            str(policy_dir),
            "--task",
            "fold_clothes",
            "--ckpt",
            "folding_final",
            "--policy-env",
            "test-policy-env",
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


def test_export_hook_precedes_rollout():
    source = (ROOT / "src/robodojo/sim/evaluation/main.py").read_text(encoding="utf-8")
    reset = source.index("env.reset(seed=env.env_seeds)")
    export = source.index("export_scene_snapshot(env, export_dir", reset)
    rollout = source.index("env.run_eval()", export)
    assert reset < export < rollout


def test_direct_simulator_entrypoint_validates_calibration_before_kit_startup():
    source = (ROOT / "src/robodojo/sim/evaluation/main.py").read_text(encoding="utf-8")
    validation = source.index("_validate_hardware_calibration(args_cli.env_cfg_type)")
    app_launch = source.index("app_launcher = AppLauncher(args_cli)")
    assert validation < app_launch
