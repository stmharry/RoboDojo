from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from pydantic import ValidationError
import pytest
import yaml

from robodojo.core.models import SetupRequest, SetupStage, SetupStageResult
from robodojo.core.paths import RepositoryPaths
from robodojo.workflows import assets, setup


def _request(tmp_path: Path, *stages: SetupStage) -> SetupRequest:
    return SetupRequest(
        stages=stages,
        policy_dir=tmp_path / "Policy",
        task="general_pickup",
        checkpoint="release",
        policy_env="uv",
        env_config="bimanual_yam",
        scene_config="molmo_yam",
        action_type="joint",
    )


def test_setup_stage_arguments_are_conditional(tmp_path):
    assert SetupRequest(stages=(SetupStage.ROOT,)).selected_stages() == (SetupStage.ROOT,)
    with pytest.raises(ValidationError, match="env_config, task"):
        SetupRequest(stages=(SetupStage.ASSETS,))
    with pytest.raises(ValidationError, match="policy_dir"):
        SetupRequest(stages=(SetupStage.POLICY,))


def test_setup_json_report_has_stable_stage_fields(capsys):
    report = setup.build_report(
        [
            SetupStageResult(name="root_environment", status="READY", detail="ready"),
            SetupStageResult(name="policy", status="WARN", detail="legacy", remediation="read the adapter README"),
        ]
    )
    setup.emit_report(report, "json")
    assert json.loads(capsys.readouterr().out) == {
        "status": "WARN",
        "stages": [
            {"name": "root_environment", "status": "READY", "detail": "ready"},
            {
                "name": "policy",
                "status": "WARN",
                "detail": "legacy",
                "remediation": "read the adapter README",
            },
        ],
    }


def test_full_setup_runs_ordered_stages_once(monkeypatch, tmp_path):
    calls: list[str] = []

    def ready(name):
        return SetupStageResult(name=name, status="READY", detail="ready")

    monkeypatch.setattr(
        setup,
        "_prerequisite_stage",
        lambda selected: calls.append("prerequisites") or ready("prerequisites"),
    )
    monkeypatch.setattr(
        setup,
        "_asset_selection_stage",
        lambda paths, request: calls.append("experiment_selection") or ready("experiment_selection"),
    )
    monkeypatch.setattr(setup, "_submodules_stage", lambda paths: calls.append("submodules") or ready("submodules"))
    monkeypatch.setattr(
        setup,
        "_root_environment_stage",
        lambda paths: calls.append("root_environment") or ready("root_environment"),
    )
    monkeypatch.setattr(setup, "_base_assets_stage", lambda paths: calls.append("base_assets") or ready("base_assets"))
    monkeypatch.setattr(
        setup,
        "_generated_assets_stages",
        lambda paths, request: calls.append("generated_assets") or [ready("generated_assets")],
    )
    monkeypatch.setattr(setup, "_policy_stage", lambda paths, request: calls.append("policy") or ready("policy"))

    report = setup.run_setup(RepositoryPaths(root=tmp_path), _request(tmp_path))

    assert report.status == "PASS"
    assert calls == [
        "prerequisites",
        "experiment_selection",
        "submodules",
        "root_environment",
        "base_assets",
        "generated_assets",
        "policy",
    ]


def test_setup_failure_stops_before_mutation(monkeypatch, tmp_path):
    monkeypatch.setattr(
        setup,
        "_prerequisite_stage",
        lambda selected: SetupStageResult(name="prerequisites", status="FAIL", detail="uv missing"),
    )
    monkeypatch.setattr(setup, "_submodules_stage", lambda paths: pytest.fail("setup continued after failure"))
    report = setup.run_setup(RepositoryPaths(root=tmp_path), _request(tmp_path))
    assert report.status == "FAIL"
    assert [stage.name for stage in report.stages] == ["prerequisites"]


def test_invalid_task_selection_stops_before_submodule_or_asset_mutation(monkeypatch, tmp_path):
    request = _request(tmp_path).model_copy(update={"task": "missing_task"})
    monkeypatch.setattr(
        setup,
        "_prerequisite_stage",
        lambda selected: SetupStageResult(name="prerequisites", status="READY", detail="ready"),
    )
    monkeypatch.setattr(setup, "build_inventory", lambda: {"tasks": [{"name": "general_pickup", "runnable": True}]})
    monkeypatch.setattr(setup, "_submodules_stage", lambda paths: pytest.fail("submodule mutation started"))
    monkeypatch.setattr(setup, "_base_assets_stage", lambda paths: pytest.fail("asset mutation started"))

    report = setup.run_setup(RepositoryPaths(root=tmp_path), request)

    assert report.status == "FAIL"
    assert [stage.name for stage in report.stages] == ["prerequisites", "experiment_selection"]
    assert report.stages[-1].remediation == "run make tasks and select a valid TASK"


def test_dirty_submodule_is_preserved_and_blocks_update(monkeypatch, tmp_path):
    submodule = tmp_path / "XPolicyLab"
    submodule.mkdir()
    (submodule / ".git").touch()
    monkeypatch.setattr(setup, "_submodule_paths", lambda paths: [submodule])
    calls: list[list[str]] = []

    def command(argv, **kwargs):
        calls.append(argv)
        assert argv == ["git", "status", "--porcelain"]
        return subprocess.CompletedProcess(argv, 0, " M policy/model.py\n", "")

    monkeypatch.setattr(setup.subprocess, "run", command)
    result = setup._submodules_stage(RepositoryPaths(root=tmp_path))
    assert result.status == "FAIL"
    assert "submodule has tracked" in result.detail
    assert calls == [["git", "status", "--porcelain"]]


def test_generated_asset_stage_infers_robot_fixture_and_task_assets(monkeypatch, tmp_path):
    profile = object()
    scene = SimpleNamespace(name="moonlake_office", document=SimpleNamespace(asset_builds=["moonlake_office"]))
    calls: list[str] = []
    monkeypatch.setattr(setup, "_asset_context", lambda paths, request: (profile, scene))
    monkeypatch.setattr(setup, "required_robot_builds", lambda profile: ("yam",))
    monkeypatch.setattr(setup, "required_fixture_builds", lambda scene, task: ("moonlake_office",))
    monkeypatch.setattr(
        setup,
        "ensure_generated_robot",
        lambda paths, name: calls.append(f"robot:{name}") or False,
    )
    monkeypatch.setattr(
        setup,
        "ensure_generated_fixture",
        lambda paths, name: calls.append(f"fixture:{name}") or True,
    )
    monkeypatch.setattr(
        setup,
        "inspect_scene_assets",
        lambda scene, task: calls.append(f"task:{task}") or SimpleNamespace(artifacts=()),
    )

    stages = setup._generated_assets_stages(RepositoryPaths(root=tmp_path), _request(tmp_path))

    assert calls == ["robot:yam", "fixture:moonlake_office", "task:general_pickup"]
    assert [stage.status for stage in stages] == ["READY", "CHANGED", "READY"]


@pytest.mark.parametrize(("robot_name", "expected"), [("yam", ("yam",)), ("openarm", ("openarm",))])
def test_generated_robot_builder_is_inferred_from_environment_config(tmp_path, robot_name, expected):
    config = tmp_path / "robot.yml"
    config.write_text(yaml.safe_dump({"robots": [{"robot_name": robot_name}]}), encoding="utf-8")
    profile = SimpleNamespace(component_paths={"robot": config})

    assert assets.required_robot_builds(profile) == expected


def test_scene_fixture_manifest_requires_current_tooling_and_checksum(monkeypatch, tmp_path):
    tooling = tmp_path / "configs/tooling/moonlake_office.yml"
    tooling.parent.mkdir(parents=True)
    tooling.write_text("fixture:\n  category: Moonlake\n", encoding="utf-8")
    asset_root = tmp_path / "assets"
    output_root = asset_root / "Object/RoboDojo/Geometry/Moonlake"
    output_root.mkdir(parents=True)
    output = output_root / "office.usd"
    output.write_bytes(b"fixture")
    monkeypatch.setattr(assets, "assets_root", lambda: asset_root)
    paths = RepositoryPaths(root=tmp_path)

    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "build_manifest_sha256": hashlib.sha256(tooling.read_bytes()).hexdigest(),
                "output": output.name,
                "output_sha256": hashlib.sha256(b"wrong").hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    assert "checksum mismatch" in assets.generated_fixture_error(paths, "moonlake_office")

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    manifest["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    (output_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert assets.generated_fixture_error(paths, "moonlake_office") is None

    tooling.write_text("fixture:\n  category: Moonlake\n# changed\n", encoding="utf-8")
    assert "stale" in assets.generated_fixture_error(paths, "moonlake_office")


def test_missing_task_scene_asset_is_prepared_during_setup(monkeypatch, tmp_path):
    profile = object()
    scene = SimpleNamespace(name="molmo_yam", document=SimpleNamespace(asset_builds=[]))
    prepared: list[str] = []
    monkeypatch.setattr(setup, "_asset_context", lambda paths, request: (profile, scene))
    monkeypatch.setattr(setup, "required_robot_builds", lambda profile: ())
    monkeypatch.setattr(setup, "required_fixture_builds", lambda scene, task: ())
    monkeypatch.setattr(
        setup,
        "inspect_scene_assets",
        lambda *args: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    monkeypatch.setattr(setup, "prepare_scene_assets", lambda scene, task: prepared.append(task))

    [stage] = setup._generated_assets_stages(RepositoryPaths(root=tmp_path), _request(tmp_path))
    assert stage.status == "CHANGED"
    assert prepared == ["general_pickup"]


def test_legacy_policy_without_prepare_hook_warns(monkeypatch, tmp_path):
    policy = tmp_path / "Policy"
    policy.mkdir()
    monkeypatch.setattr(setup, "policy_hook_command", lambda request, hook: None)
    result = setup._policy_stage(RepositoryPaths(root=tmp_path), _request(tmp_path, SetupStage.POLICY))
    assert result.status == "WARN"
    assert "legacy setup" in result.remediation
