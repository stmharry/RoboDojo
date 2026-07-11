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


def test_official_camera_hardware_and_mount_contract():
    camera = yaml.safe_load((ROOT / "env_cfg/camera/openarm_cloth_folding.yml").read_text())
    robot = yaml.safe_load((ROOT / "scripts/assets/openarm_robot_config.yml").read_text())
    base = camera["cam_head"]["camera"]
    assert base["sensor_model"] == "Fafeicy_HBV-1716WA_OV2710"
    assert base["mount_link"] == "robot0/openarm_body_link"
    assert base["mount_basis"] == "bimanual_centerline_extrusion_from_robot_root"
    assert base["projection_backend"] == "pinhole_postprocess"
    assert base["published_diagonal_fov_deg"] == 140.0
    for side in ("left", "right"):
        wrist = robot[side]["camera"][0]
        assert wrist["link"] == f"openarm_{side}_link7"
        assert wrist["sensor_model"] == "Arducam_IMX708_102deg_fixed_focus"
        assert wrist["mount_basis"] == "end_effector_contact_axis"
        assert wrist["projection_backend"] == "pinhole_postprocess"
        assert wrist["published_diagonal_fov_deg"] == 102.0


def test_layout_override_is_fixture_only():
    layout = {
        "Table": {"default": "wood"},
        "Geometry": {"camera_stand": [{"label": "camera_stand"}], "rail": [{}]},
        "Garment": {"pose": [1, 2, 3]},
    }
    mirrored = apply_fixture_overrides(
        layout,
        {"Table": {"default": "white"}, "remove_fixtures": ["Geometry.camera_stand"]},
    )
    assert mirrored["Table"]["default"] == "white"
    assert mirrored["Geometry"] == {"rail": [{}]}
    assert mirrored["Garment"] == layout["Garment"]
    with pytest.raises(ValueError, match="non-fixture"):
        apply_fixture_overrides(layout, {"Garment": {"pose": [0, 0, 0]}})
    with pytest.raises(ValueError, match="forbidden fixture removals"):
        apply_fixture_overrides(layout, {"remove_fixtures": ["Geometry.rail"]})
    with pytest.raises(ValueError, match="must be a list"):
        apply_fixture_overrides(layout, {"remove_fixtures": "Geometry.camera_stand"})
