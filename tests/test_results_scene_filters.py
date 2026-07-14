import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from robodojo.cli import app
from robodojo.workflows import results_stats, results_summary

RUNNER = CliRunner()


def _write_result(
    root: Path,
    *,
    task: str,
    embodiment: str,
    scene: str | None,
    count: int,
    score: float = 1.0,
    policy: str = "TestPolicy",
    seed: int = 0,
    timestamp: str = "2026-07-14_00-00-00",
) -> Path:
    run = root / task / policy / embodiment / f"{seed}_ckpt_name=test,action_type=joint" / timestamp
    run.mkdir(parents=True)
    payload = {
        "eval_time": count,
        "success_rate": 1.0,
        "score": score,
        "details": {str(index): {"layout_id": index, "success": score > 0.0, "score": score} for index in range(count)},
    }
    if scene is not None:
        payload["scene_config"] = scene
    (run / "_result.json").write_text(json.dumps(payload), encoding="utf-8")
    return run


def test_summary_fails_on_ambiguous_environment_scene_and_filters_cleanly(monkeypatch, tmp_path):
    _write_result(
        tmp_path,
        task="fasten_screws",
        embodiment="arx_x5",
        scene="default",
        count=50,
    )
    _write_result(
        tmp_path,
        task="fasten_screws",
        embodiment="bimanual_yam",
        scene="molmo_yam",
        count=50,
        score=0.0,
    )
    monkeypatch.setattr(results_summary, "ROOT", str(tmp_path))

    with pytest.raises(SystemExit, match="Pass --env-cfg and/or --scene"):
        results_summary.main(["--output", str(tmp_path / "ambiguous.md")])

    output = tmp_path / "filtered.md"
    results_summary.main(["--output", str(output), "--env-cfg", "arx_x5", "--scene", "default"])
    assert output.is_file()
    assert "RoboDojo Evaluation Summary" in output.read_text(encoding="utf-8")


def test_stats_filters_preserve_unambiguous_output_schema(tmp_path):
    _write_result(
        tmp_path,
        task="fasten_screws",
        embodiment="arx_x5",
        scene="default",
        count=50,
    )
    _write_result(
        tmp_path,
        task="fasten_screws",
        embodiment="bimanual_yam",
        scene="molmo_yam",
        count=50,
        score=0.5,
    )

    with pytest.raises(SystemExit, match="Ambiguous results"):
        results_stats.collect_distributions(
            str(tmp_path),
            ["TestPolicy"],
            tasks=["fasten_screws"],
        )

    result = results_stats.collect_distributions(
        str(tmp_path),
        ["TestPolicy"],
        tasks=["fasten_screws"],
        env_config="arx_x5",
        scene_config="default",
    )
    assert result == {
        "root": str(tmp_path),
        "policies": ["TestPolicy"],
        "tasks": ["fasten_screws"],
        "aggregated": {"fasten_screws": {"TestPolicy": {"1": 50}}},
    }


def test_paired_results_require_the_same_environment_and_scene(tmp_path):
    _write_result(
        tmp_path,
        task="fold_clothes",
        embodiment="arx_x5",
        scene="default",
        count=25,
    )
    _write_result(
        tmp_path,
        task="fold_clothes_random",
        embodiment="arx_x5",
        scene="molmo_yam",
        count=25,
    )

    with pytest.raises(SystemExit, match="Ambiguous results"):
        results_stats.collect_distributions(
            str(tmp_path),
            ["TestPolicy"],
            tasks=["fold_clothes"],
        )

    filtered = results_stats.collect_distributions(
        str(tmp_path),
        ["TestPolicy"],
        tasks=["fold_clothes"],
        env_config="arx_x5",
        scene_config="default",
    )
    assert filtered["aggregated"] == {}

    _write_result(
        tmp_path,
        task="fold_clothes_random",
        embodiment="arx_x5",
        scene="default",
        count=25,
        timestamp="2026-07-14_00-01-00",
    )
    matched = results_stats.collect_distributions(
        str(tmp_path),
        ["TestPolicy"],
        tasks=["fold_clothes"],
        env_config="arx_x5",
        scene_config="default",
    )
    assert matched["aggregated"] == {"fold_clothes": {"TestPolicy": {"1": 50}}}


@pytest.mark.parametrize("command", [["summarize", "--help"], ["results", "stats", "--help"]])
def test_result_commands_expose_environment_and_scene_filters(command):
    result = RUNNER.invoke(app, command)
    assert result.exit_code == 0
    assert "--env-cfg" in result.stdout
    assert "--scene" in result.stdout
