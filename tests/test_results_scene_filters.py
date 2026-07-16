import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from robodojo.cli import app
from robodojo.core.scene_identity import ARTIFACT_SCHEMA_VERSION, ArtifactSchemaError
from robodojo.workflows import results_stats, results_summary
from robodojo.workflows.errors import ResultsError

RUNNER = CliRunner()


def _write_result(
    root: Path,
    *,
    task: str,
    base_task: str | None = None,
    embodiment: str,
    scene: str | None,
    count: int,
    score: float = 1.0,
    policy: str = "TestPolicy",
    seed: int = 0,
    timestamp: str = "2026-07-14_00-00-00",
) -> Path:
    base_task = base_task or task
    run = root / task / policy / embodiment / f"{seed}_ckpt_name=test,action_type=joint" / timestamp
    run.mkdir(parents=True)
    payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "task_name": base_task,
        "protocol_name": task,
        "episode_horizon": 200,
        "native_eval_num": count,
        "robodojo_revision": "r" * 40,
        "xpolicylab_revision": "x" * 40,
        "policy_profile": policy,
        "policy_descriptor_hash": "d" * 64,
        "environment_profile": embodiment,
        "environment_profile_hash": "e" * 64,
        "environment_asset_hash": "f" * 64,
        "policy_contract": embodiment,
        "scene_config": scene,
        "layout_config_name": scene,
        "layout_source": "bundled",
        "layout_set_hash": "a" * 64,
        "eval_time": count,
        "success_rate": 1.0,
        "score": score,
        "details": {
            str(index): {
                "layout_id": index,
                "layout_file": f"{base_task}_{index}.json",
                "layout_sha256": f"{index:064x}",
                "success": score > 0.0,
                "score": score,
            }
            for index in range(count)
        },
    }
    (run / "_result.json").write_text(json.dumps(payload), encoding="utf-8")
    return run


def test_summary_fails_on_ambiguous_environment_scene_and_filters_cleanly(tmp_path):
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
    with pytest.raises(ResultsError, match="Pass --environment and/or --scene"):
        results_summary.summarize_results(
            results_root=tmp_path,
            output=tmp_path / "ambiguous.md",
        )

    output = tmp_path / "filtered.md"
    result = results_summary.summarize_results(
        results_root=tmp_path,
        output=output,
        environment="arx_x5",
        scene="default",
    )
    assert result == output.resolve()
    assert output.is_file()
    assert "RoboDojo Evaluation Summary" in output.read_text(encoding="utf-8")


def test_summary_reports_named_protocols_without_folding_them_into_the_upstream_scorecard(tmp_path):
    _write_result(
        tmp_path,
        task="moonlake_office_general_pickup",
        base_task="general_pickup",
        embodiment="bimanual_yam_moonlake_office",
        scene="moonlake_office",
        count=1,
        score=0.0,
        policy="Pi_05",
    )
    output = tmp_path / "summary.md"

    results_summary.summarize_results(results_root=tmp_path, output=output)

    report = output.read_text(encoding="utf-8")
    assert "Additional named protocols" in report
    assert "`moonlake_office_general_pickup`" in report
    assert "| 0 | 1 | 0 | 0.00 |" in report


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

    with pytest.raises(ResultsError, match="Ambiguous results"):
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

    with pytest.raises(ResultsError, match="Ambiguous results"):
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


@pytest.mark.parametrize("loader", [results_summary.load_completed_result, results_stats.load_completed_result])
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("artifact_schema_version"), "artifact_schema_version mismatch"),
        (lambda payload: payload.update(artifact_schema_version=1), "artifact_schema_version mismatch"),
        (lambda payload: payload.update(layout_name="fold_clothes"), "removed layout_name selector"),
    ],
)
def test_result_loaders_strictly_reject_legacy_artifacts(tmp_path, loader, mutation, message):
    run = _write_result(
        tmp_path,
        task="fold_clothes",
        embodiment="arx_x5",
        scene="default",
        count=1,
    )
    result_path = run / "_result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    mutation(payload)
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactSchemaError, match=message):
        loader(str(result_path))


@pytest.mark.parametrize("command", [["results", "summarize", "--help"], ["results", "stats", "--help"]])
def test_result_commands_expose_environment_and_scene_filters(command):
    result = RUNNER.invoke(app, command)
    assert result.exit_code == 0
    assert "--results-root" in result.stdout
    assert "--environment" in result.stdout
    assert "--scene" in result.stdout
    assert "--env-cfg" not in result.stdout


def test_result_commands_use_explicit_results_root(tmp_path):
    _write_result(
        tmp_path,
        task="fasten_screws",
        embodiment="arx_x5",
        scene="default",
        count=50,
    )
    output = tmp_path / "report.md"

    summary = RUNNER.invoke(
        app,
        [
            "results",
            "summarize",
            "--results-root",
            str(tmp_path),
            "--output",
            str(output),
            "--environment",
            "arx_x5",
            "--scene",
            "default",
        ],
    )
    stats = RUNNER.invoke(
        app,
        [
            "results",
            "stats",
            "--results-root",
            str(tmp_path),
            "--policy",
            "TestPolicy",
            "--task",
            "fasten_screws",
            "--environment",
            "arx_x5",
            "--scene",
            "default",
        ],
    )

    assert summary.exit_code == 0, summary.output
    assert output.is_file()
    assert stats.exit_code == 0, stats.output
    assert "TestPolicy (50 episodes)" in stats.stdout


def test_stats_uses_default_policies_and_writes_json(tmp_path):
    _write_result(
        tmp_path,
        task="fasten_screws",
        embodiment="arx_x5",
        scene="default",
        count=50,
        policy="pi05_bimanual_yam",
    )
    output = tmp_path / "nested/score-stats.json"

    result = RUNNER.invoke(
        app,
        [
            "results",
            "stats",
            "--results-root",
            str(tmp_path),
            "--task",
            "fasten_screws",
            "--json-out",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["policies"] == list(results_stats.DEFAULT_POLICIES)
    assert report["aggregated"]["fasten_screws"]["pi05_bimanual_yam"] == {"1": 50}
