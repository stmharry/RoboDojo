from pathlib import Path

import pytest

from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles import load_environment_profile

ROOT = Path(__file__).resolve().parents[1]


def test_environment_profiles_resolve_policy_and_diagnostic_metadata():
    paths = RepositoryPaths.resolve(ROOT)
    openarm = load_environment_profile(paths, "openarm_lerobot")
    arx = load_environment_profile(paths, "arx_x5")

    assert openarm.xpolicylab_env_cfg_type == "openarm_cloth_folding"
    assert openarm.document.layout_config_name == "arx_x5"
    assert openarm.matched_replay_manifest == ROOT / "configs/reference/openarm_lerobot_wrist_calibration.yml"
    assert arx.xpolicylab_env_cfg_type == "arx_x5"
    assert arx.matched_replay_manifest is None


@pytest.mark.parametrize("name", ["openarm_wowrobo_v1_1", "openarm_anvil_v2"])
def test_pending_hardware_profiles_can_be_inspected_but_not_launched(name):
    paths = RepositoryPaths.resolve(ROOT)
    profile = load_environment_profile(paths, name, validate_calibration=False)

    assert profile.document.hardware_calibration == name
    assert profile.xpolicylab_env_cfg_type == "openarm_cloth_folding"
    with pytest.raises(ValueError, match="not release-ready"):
        load_environment_profile(paths, name)
