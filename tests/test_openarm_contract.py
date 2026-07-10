import json
from pathlib import Path

import pytest
import yaml

from env.scene_manager.layout_overrides import apply_fixture_overrides

ROOT = Path(__file__).resolve().parents[1]


def test_openarm_timing_and_dimensions():
    env_cfg = yaml.safe_load((ROOT / "env_cfg/openarm_cloth_folding.yml").read_text())
    sim_cfg = yaml.safe_load((ROOT / "env_cfg/sim/openarm_cloth_folding.yml").read_text())
    robot_info = json.loads((ROOT / "env_cfg/robot/_robot_info.json").read_text())
    assert env_cfg["observation"]["collect_freq"] == 30
    assert 1.0 / (sim_cfg["dt"] * 30) == 8.0
    assert sum(robot_info["dual_openarm"]["arm_dim"]) + sum(robot_info["dual_openarm"]["ee_dim"]) == 16


def test_layout_override_is_fixture_only():
    layout = {"Table": {"default": "wood"}, "Garment": {"pose": [1, 2, 3]}}
    mirrored = apply_fixture_overrides(layout, {"Table": {"default": "white"}})
    assert mirrored["Table"]["default"] == "white"
    assert mirrored["Garment"] == layout["Garment"]
    with pytest.raises(ValueError, match="non-fixture"):
        apply_fixture_overrides(layout, {"Garment": {"pose": [0, 0, 0]}})
