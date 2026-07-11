import json
import os
from pathlib import Path
import subprocess

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
                "success_rate": 0.0,
                "eval_time": 1,
                "score": 0.0,
                "details": {"0": {"success": False, "score": 0.0}},
            }
        ),
        encoding="utf-8",
    )
    (run / "_COMPLETE.json").write_text("{}\n", encoding="utf-8")
    return run


def _read_only_tree(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def test_summarize_reads_durable_tree_and_writes_local_scratch(tmp_path):
    durable = tmp_path / "durable"
    scratch = tmp_path / "scratch"
    _completed_result(durable)
    _read_only_tree(durable)
    before = sorted(str(path.relative_to(durable)) for path in durable.rglob("*"))
    environment = os.environ.copy()
    environment.update(
        {
            "ROBODOJO_STORAGE_ROOT": str(durable),
            "ROBODOJO_LOCAL_SCRATCH_ROOT": str(scratch),
        }
    )
    environment.pop("ROBODOJO_SUMMARY_PATH", None)

    result = subprocess.run(
        ["bash", str(ROOT / "scripts/robodojo.sh"), "summarize"],
        cwd=ROOT,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )

    output = scratch / "runs/reports/_summary.md"
    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert "RoboDojo Evaluation Summary" in output.read_text(encoding="utf-8")
    assert not (durable / "runs/eval_result/RoboDojo/_summary.md").exists()
    assert sorted(str(path.relative_to(durable)) for path in durable.rglob("*")) == before


def test_cli_output_overrides_environment_and_creates_parents(tmp_path):
    durable = tmp_path / "durable"
    _completed_result(durable)
    environment = os.environ.copy()
    environment.update(
        {
            "ROBODOJO_STORAGE_ROOT": str(durable),
            "ROBODOJO_LOCAL_SCRATCH_ROOT": str(tmp_path / "scratch"),
            "ROBODOJO_SUMMARY_PATH": str(tmp_path / "environment.md"),
        }
    )
    output = tmp_path / "nested/cli/summary.md"

    result = subprocess.run(
        ["bash", str(ROOT / "scripts/robodojo.sh"), "summarize", "--output", str(output)],
        cwd=ROOT,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert not (tmp_path / "environment.md").exists()
