from pathlib import Path

import pytest

from robodojo.core.models import EnvironmentConfigDocument, SceneConfigDocument
from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles import load_environment_profile, load_scene_profile
from robodojo.sim import scene_preparation

ROOT = Path(__file__).resolve().parents[1]


def test_environment_profile_rejects_scene_owned_fields():
    with pytest.raises(ValueError, match="scene/task-owned fields"):
        EnvironmentConfigDocument.model_validate(
            {
                "config_name": "invalid",
                "config": {"sim": "sim", "robot": "robot", "camera": "camera"},
                "task_instruction_overrides": {"fold_clothes": []},
            }
        )


def test_scene_profile_rejects_empty_asset_preparers():
    with pytest.raises(ValueError, match="non-empty task and preparer"):
        SceneConfigDocument.model_validate(
            {
                "config_name": "invalid",
                "component": "default",
                "layout_set": "arx_x5",
                "task_asset_preparers": {"fold_clothes": []},
            }
        )


def test_environment_profiles_resolve_policy_and_diagnostic_metadata():
    paths = RepositoryPaths.resolve(ROOT)
    openarm = load_environment_profile(paths, "openarm_lerobot")
    arx = load_environment_profile(paths, "arx_x5")

    assert openarm.path == ROOT / "configs/environment/openarm_lerobot.yml"
    assert openarm.component_paths["robot"] == ROOT / "configs/robot/dual_openarm_lerobot.yml"
    assert openarm.matched_replay_manifest == ROOT / "configs/reference/openarm_lerobot_wrist_calibration.yml"
    assert arx.matched_replay_manifest is None
    assert set(arx.component_paths) == {"sim", "robot", "camera"}

    scene = load_scene_profile(paths, "molmo_yam")
    assert scene.document.layout_set == "molmo_yam"
    assert scene.component_path == ROOT / "configs/scene/components/molmo_yam.yml"
    assert scene.document.task_asset_preparers == {"fold_clothes": ["yam_short_sleeve_garment"]}


def test_runtime_yaml_domains_use_the_canonical_config_root():
    paths = RepositoryPaths.resolve(ROOT)

    assert paths.environment_configs == ROOT / "configs"
    assert paths.environment_profiles == ROOT / "configs/environment"
    assert paths.scene_profiles == ROOT / "configs/scene/profiles"
    assert paths.scene_components == ROOT / "configs/scene/components"
    assert paths.task_configs == ROOT / "configs/task"
    assert (paths.task_configs / "_task.yml").is_file()
    assert not (ROOT / "configs/arx_x5.yml").exists()
    assert not (ROOT / "task/RoboDojo/config").exists()


def test_scene_asset_preparation_is_named_and_task_scoped(monkeypatch):
    scene = load_scene_profile(RepositoryPaths.resolve(ROOT), "molmo_yam")
    calls = []
    monkeypatch.setitem(scene_preparation._PREPARERS, "yam_short_sleeve_garment", lambda: calls.append("run"))

    scene_preparation.prepare_scene_assets(scene, "general_pickup")
    assert calls == []
    scene_preparation.prepare_scene_assets(scene, "fold_clothes")
    assert calls == ["run"]


@pytest.mark.parametrize("name", ["openarm_wowrobo_v1_1", "openarm_anvil_v2"])
def test_pending_hardware_profiles_can_be_inspected_but_not_launched(name):
    paths = RepositoryPaths.resolve(ROOT)
    profile = load_environment_profile(paths, name, validate_calibration=False)

    assert profile.document.hardware_calibration == name
    with pytest.raises(ValueError, match="not release-ready"):
        load_environment_profile(paths, name)
