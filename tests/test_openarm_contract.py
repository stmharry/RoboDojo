import json
from pathlib import Path

import pytest
import yaml

from robodojo.core.calibration import load_hardware_calibration

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("openarm_wowrobo_v1_1", "openarm_anvil_v2")


@pytest.mark.parametrize("profile", PROFILES)
def test_hardware_profiles_share_only_upstream_contracts(profile):
    env_cfg = yaml.safe_load((ROOT / "configs" / f"{profile}.yml").read_text())
    sim_cfg = yaml.safe_load((ROOT / "configs/sim" / f"{profile}.yml").read_text())
    robot_info = json.loads((ROOT / "configs/robot/_robot_info.json").read_text())[
        f"dual_{profile}"
    ]

    assert env_cfg["config"] == {
        "sim": profile,
        "scene": "default",
        "robot": f"dual_{profile}",
        "camera": profile,
    }
    assert env_cfg["layout_config_name"] == "arx_x5"
    assert env_cfg["hardware_calibration"] == profile
    assert env_cfg["observation"]["collect_freq"] == 30
    assert 1.0 / (sim_cfg["dt"] * 30) == 8.0
    assert sum(robot_info["arm_dim"]) + sum(robot_info["ee_dim"]) == 16


def test_ambiguous_and_cloth_specific_profiles_are_removed():
    for path in (
        "configs/openarm.yml",
        "configs/camera/openarm.yml",
        "configs/robot/dual_openarm.yml",
        "configs/openarm_cloth_folding.yml",
        "configs/camera/openarm_cloth_folding.yml",
        "configs/scene/openarm_cloth_folding.yml",
        "configs/sim/openarm_cloth_folding.yml",
    ):
        assert not (ROOT / path).exists()


@pytest.mark.parametrize(
    ("profile", "vendor", "revision"),
    [
        ("openarm_wowrobo_v1_1", "WowRobo", "v1.1"),
        ("openarm_anvil_v2", "Anvil Robotics", "v2"),
    ],
)
def test_unmeasured_profiles_are_explicitly_blocked(profile, vendor, revision):
    manifest = yaml.safe_load((ROOT / "configs/calibration" / f"{profile}.yml").read_text())
    assert manifest["status"] == "pending_measurement"
    assert manifest["identity"]["vendor"] == vendor
    assert manifest["identity"]["hardware_revision"] == revision
    assert manifest["identity"]["robot_serial"] is None
    assert manifest["sources"] == []
    with pytest.raises(ValueError, match="not release-ready"):
        load_hardware_calibration(ROOT / "configs", profile)


def test_openarm_asset_inputs_remain_pinned_and_shared():
    manifest = yaml.safe_load((ROOT / "configs/tooling/openarm.yml").read_text())
    sources = manifest["sources"]
    hashes = sources["hardware_modifications"]["sha256"]
    assert sources["openarm_isaac_lab"]["revision"] == "bad82e23716e6941c2de78ccb978f57c78b37734"
    assert sources["hardware_modifications"]["revision"] == "ffe34b93c070343042eb9412fbfeffce16139947"
    assert hashes["jaw_normal.stl"] == "6ae41c9fbba411333954b8f4d1c6867b61fad1be7d7b936899c27d43410a2137"
    assert manifest["asset"]["upper_arm_extension_m"] == 0.05


def test_policy_reference_is_right_first_30hz_for_both_profiles():
    for profile in PROFILES:
        manifest = yaml.safe_load((ROOT / "configs/calibration" / f"{profile}.yml").read_text())
        reference = manifest["policy_reference"]
        assert reference["lerobot_contract"] == "v0.5.1"
        assert reference["state_action_order"] == "right_first_16d"
        assert reference["observation_frequency_hz"] == 30
