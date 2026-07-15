from pathlib import Path

import pytest

from robodojo.core.models import EnvironmentConfigDocument
from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles import load_environment_profile

ROOT = Path(__file__).resolve().parents[1]


def test_environment_profile_rejects_empty_instruction_overrides():
    with pytest.raises(ValueError, match="must contain non-empty templates"):
        EnvironmentConfigDocument.model_validate(
            {
                "config_name": "invalid",
                "config": {"sim": "sim", "scene": "scene", "robot": "robot", "camera": "camera"},
                "task_instruction_overrides": {"fold_clothes": []},
            }
        )


def test_environment_profiles_resolve_policy_and_diagnostic_metadata():
    paths = RepositoryPaths.resolve(ROOT)
    openarm = load_environment_profile(paths, "openarm_lerobot")
    arx = load_environment_profile(paths, "arx_x5")

    assert openarm.path == ROOT / "configs/environment/openarm_lerobot.yml"
    assert openarm.component_paths["robot"] == ROOT / "configs/robot/dual_openarm_lerobot.yml"
    assert openarm.document.layout_config_name == "arx_x5"
    assert openarm.matched_replay_manifest == ROOT / "configs/reference/openarm_lerobot_wrist_calibration.yml"
    assert arx.matched_replay_manifest is None
    assert arx.document.task_instruction_overrides == {}


def test_runtime_yaml_domains_use_the_canonical_config_root():
    paths = RepositoryPaths.resolve(ROOT)

    assert paths.environment_configs == ROOT / "configs"
    assert paths.environment_profiles == ROOT / "configs/environment"
    assert paths.task_configs == ROOT / "configs/task"
    assert (paths.task_configs / "_task.yml").is_file()
    assert not (ROOT / "configs/arx_x5.yml").exists()
    assert not (ROOT / "task/RoboDojo/config").exists()


@pytest.mark.parametrize("name", ["openarm_wowrobo_v1_1", "openarm_anvil_v2"])
def test_pending_hardware_profiles_can_be_inspected_but_not_launched(name):
    paths = RepositoryPaths.resolve(ROOT)
    profile = load_environment_profile(paths, name, validate_calibration=False)

    assert profile.document.hardware_calibration == name
    with pytest.raises(ValueError, match="not release-ready"):
        load_environment_profile(paths, name)
