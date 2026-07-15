import json
import os
from pathlib import Path
import subprocess
import sys

from robodojo.core.scene_identity import ARTIFACT_SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]


def _completed_result(storage_root: Path) -> Path:
    run = (
        storage_root
        / "runs/eval_result/RoboDojo/stack_bowls/demo_policy/arx_x5"
        / "0_ckpt_name=demo,action_type=joint/2026-07-11_00-00-00"
    )
    run.mkdir(parents=True)
    (run / "_result.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "task_name": "stack_bowls",
                "protocol_name": "stack_bowls",
                "episode_horizon": 800,
                "native_eval_num": 25,
                "scene_config": "default",
                "layout_config_name": "default",
                "layout_source": "bundled",
                "layout_set_hash": "a" * 64,
                "success_rate": 0.0,
                "eval_time": 1,
                "score": 0.0,
                "details": {
                    "0": {
                        "layout_id": 0,
                        "layout_file": "stack_bowls_0.json",
                        "layout_sha256": "b" * 64,
                        "success": False,
                        "score": 0.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "_COMPLETE.json").write_text("{}\n", encoding="utf-8")
    return run


def test_summarize_reads_and_writes_single_local_root(tmp_path):
    local = tmp_path / "local"
    _completed_result(local)
    environment = os.environ.copy()
    environment["ROBODOJO_STORAGE_ROOT"] = str(local)

    result = subprocess.run(
        [sys.executable, "-m", "robodojo.cli", "results", "summarize"],
        cwd=ROOT,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )

    output = local / "runs/reports/_summary.md"
    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert "RoboDojo Evaluation Summary" in output.read_text(encoding="utf-8")
    assert not (local / "runs/eval_result/RoboDojo/_summary.md").exists()


def test_cli_output_overrides_environment_and_creates_parents(tmp_path):
    local = tmp_path / "local"
    _completed_result(local)
    environment = os.environ.copy()
    environment["ROBODOJO_STORAGE_ROOT"] = str(local)
    output = tmp_path / "nested/cli/summary.md"

    result = subprocess.run(
        [sys.executable, "-m", "robodojo.cli", "results", "summarize", "--output", str(output)],
        cwd=ROOT,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert not (local / "runs/reports/_summary.md").exists()


def test_summarize_strictly_rejects_legacy_results(tmp_path):
    local = tmp_path / "local"
    run = _completed_result(local)
    result_path = run / "_result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload.pop("artifact_schema_version")
    payload["layout_name"] = "stack_bowls"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    environment = os.environ.copy()
    environment["ROBODOJO_STORAGE_ROOT"] = str(local)

    result = subprocess.run(
        [sys.executable, "-m", "robodojo.cli", "results", "summarize"],
        cwd=ROOT,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "removed layout_name selector" in result.stderr
    assert not (local / "runs/reports/_summary.md").exists()
