import ast
from pathlib import Path
import xml.etree.ElementTree as ET

from omegaconf import OmegaConf
import pytest
import yaml

from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles import load_environment_profile
from robodojo.sim.camera_template import YAM_TOP, YAM_WRIST
from robodojo.sim.environment.camera_manager.rig_spec import normalize_camera_rig
from robodojo.sim.utils.pipeline_utils import process_config
from robodojo.workflows.assets_yam import derive_yam_urdf

ROOT = Path(__file__).resolve().parents[1]


def test_bimanual_yam_profile_preserves_30hz_component_graph():
    profile = load_environment_profile(RepositoryPaths.resolve(ROOT), "bimanual_yam")
    assert profile.document.layout_config_name == "arx_x5"
    assert profile.document.config.model_dump() == {
        "sim": "real_time_30hz",
        "scene": "default",
        "robot": "dual_yam",
        "camera": "bimanual_yam",
    }
    assert profile.document.observation["collect_freq"] == 30
    assert profile.num_envs == 1

    robot_info = yaml.safe_load((ROOT / "configs/robot/_robot_info.json").read_text())["dual_yam"]
    assert robot_info == {"arm_dim": [6, 6], "ee_dim": [1, 1]}


def test_fold_clothes_does_not_replace_yam_profile_components():
    profile = load_environment_profile(RepositoryPaths.resolve(ROOT), "bimanual_yam")
    payload = profile.payload
    env_cfg = OmegaConf.create(
        {
            "sim": yaml.safe_load(profile.component_paths["sim"].read_text()),
            "scene": yaml.safe_load(profile.component_paths["scene"].read_text()),
            "robot": yaml.safe_load(profile.component_paths["robot"].read_text()),
            "camera": yaml.safe_load(profile.component_paths["camera"].read_text()),
            "eval_cfg": payload,
        }
    )
    processed, eval_num = process_config(env_cfg, "fold_clothes")
    assert [robot.robot_name for robot in processed.robot.robots] == ["yam", "yam"]
    assert processed.camera.camera_rig.profile_id == "bimanual_yam"
    assert processed.sim.render_interval == 8
    assert eval_num == 25


def test_dual_yam_order_pose_and_no_planner_are_explicit():
    robots = yaml.safe_load((ROOT / "configs/robot/dual_yam.yml").read_text())["robots"]
    assert [robot["default_root_pos"] for robot in robots] == [
        [-0.24, -0.45, 0.765],
        [0.24, -0.45, 0.765],
    ]
    assert all(robot["default_root_rot"] == pytest.approx([0.70710678, 0.0, 0.0, 0.70710678]) for robot in robots)
    assert all(robot["robot_name"] == "yam" for robot in robots)
    assert all(robot["coupled"] is False and robot["need_planner"] is False for robot in robots)


def test_yam_camera_rig_matches_molmo_contract_and_runtime_templates():
    rig = normalize_camera_rig(yaml.safe_load((ROOT / "configs/camera/bimanual_yam.yml").read_text()))
    assert rig.profile_id == "bimanual_yam"
    assert rig.default_frequency == 30
    assert [camera.observation_key for camera in rig.cameras] == [
        "cam_head",
        "cam_left_wrist",
        "cam_right_wrist",
    ]
    top, left, right = rig.cameras
    assert top.sensor["stream_resolution"] == [640, 360]
    assert top.mount["position"] == [0.0, -0.30, 1.565]
    assert top.mount["orientation"] == pytest.approx([0.54167522, -0.45451948, 0.45451948, 0.54167522])
    assert top.projection["fx"] == pytest.approx(462.1386898729645)
    assert top.projection["horizontal_fov_deg"] == 69.4
    assert [left.mount["target"], right.mount["target"]] == ["robot0/link6", "robot1/link6"]
    for wrist in (left, right):
        assert wrist.mount["position"] == [0.0, 0.09, 0.06]
        assert wrist.mount["orientation"] == pytest.approx([0.61237243, -0.35355339, -0.35355340, -0.61237244])
        assert wrist.projection["fx"] == pytest.approx(337.20964008990796)
        assert wrist.projection["horizontal_fov_deg"] == 87.0

    assert 640 * YAM_TOP["focal_length"] / YAM_TOP["horizontal_aperture"] == pytest.approx(top.projection["fx"])
    assert 640 * YAM_WRIST["focal_length"] / YAM_WRIST["horizontal_aperture"] == pytest.approx(left.projection["fx"])


def test_yam_tooling_and_policy_reference_are_revision_pinned():
    tooling = yaml.safe_load((ROOT / "configs/tooling/yam.yml").read_text())
    reference = yaml.safe_load((ROOT / "configs/reference/bimanual_yam.yml").read_text())
    assert tooling["sources"]["i2rt"]["revision"] == "ac096928d6899ddf852a71c5e8fbaa6055cd9745"
    assert reference["sources"]["simulator_contract"]["revision"] == ("c2282820f9b188b60e66ea1636b3efd81c45cbb4")
    assert reference["sources"]["policy_checkpoint"]["revision"] == ("8dcbed66f2380e4393189c303ea72488eb9e63c2")
    assert reference["state_action_contract"]["dimension"] == 14
    assert reference["state_action_contract"]["gripper"]["formula"] == "g=-q/0.0475"
    assert reference["state_action_contract"]["predicted_horizon"] == 30
    assert reference["state_action_contract"]["executed_horizon"] == 25

    robot = tooling["robot_config"]
    assert robot["arm_joints_name"] == [f"dof_joint{i}" for i in range(1, 7)]
    assert robot["gripper_joints_name"] == ["dof_joint7", "dof_joint8"]
    assert robot["gripper_move"] == {"base": "dof_joint7", "sign": -1.0, "mimic": ["dof_joint8", 1.0, 0.0]}
    assert robot["gripper_scale"] == [-0.0475, 0.0]
    assert robot["camera_mount_links"] == {"link6": "gripper"}


def _fake_i2rt_checkout(root: Path) -> None:
    arm = root / "i2rt/robot_models/arm/yam"
    gripper = root / "i2rt/robot_models/gripper/linear_3507"
    (arm / "assets").mkdir(parents=True)
    (gripper / "assets").mkdir(parents=True)
    (root / "LICENSE").write_text("MIT fixture\n")
    for name in ("base.stl", "link1.stl", "link2.stl", "link3.stl", "link4.stl", "link5.stl"):
        (arm / "assets" / name).write_bytes(f"solid {name}\nendsolid\n".encode())
    for name in ("gripper.stl", "tip_left.stl", "tip_right.stl"):
        (gripper / "assets" / name).write_bytes(f"solid {name}\nendsolid\n".encode())

    links = ["root", "base", "link1", "link2", "link3", "link4", "link5", "gripper", "tip_left", "tip_right"]
    mesh_by_link = {name: name for name in links if name != "root"}
    parts = ['<robot name="source">']
    for link in links:
        parts.append(f'<link name="{link}">')
        if link in mesh_by_link:
            mesh_name = mesh_by_link[link]
            source_name = "Base.stl" if mesh_name == "base" else f"{mesh_name}.stl"
            parts.append(
                '<visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry>'
                f'<mesh filename="package://source/meshes/{source_name}"/>'
                "</geometry></visual>"
            )
        parts.append("</link>")
    parts.append(
        '<joint name="dof_joint0" type="revolute"><origin xyz="0 0 0" rpy="0 1.5708 1.5708"/>'
        '<axis xyz="-1 0 0"/><parent link="root"/><child link="base"/>'
        '<limit effort="1" velocity="1" lower="0" upper="0"/></joint>'
    )
    chain = ["base", "link1", "link2", "link3", "link4", "link5", "gripper", "tip_left", "tip_right"]
    for index, (parent, child) in enumerate(zip(chain, chain[1:]), start=1):
        joint_type = "prismatic" if index >= 7 else "revolute"
        parts.append(
            f'<joint name="dof_joint{index}" type="{joint_type}"><parent link="{parent}"/>'
            f'<child link="{child}"/><limit effort="1" velocity="1" lower="-0.01" upper="0"/></joint>'
        )
    parts.append("</robot>")
    (arm / "yam.urdf").write_text("".join(parts))


def test_derived_yam_urdf_is_normalized_and_collision_complete(tmp_path):
    _fake_i2rt_checkout(tmp_path)
    tooling = yaml.safe_load((ROOT / "configs/tooling/yam.yml").read_text())
    output = tmp_path / "output"
    contract = derive_yam_urdf(tmp_path, output, tooling)
    tree = ET.parse(output / "YAM.urdf")
    robot = tree.getroot()

    base_joint = next(joint for joint in robot.findall("joint") if joint.get("name") == "dof_joint0")
    assert base_joint.get("type") == "fixed"
    assert base_joint.find("origin").get("rpy") == "0 1.5708 1.5708"
    assert base_joint.find("axis") is None and base_joint.find("limit") is None
    assert contract["collision_geometry_count"] == 9
    assert len(robot.findall(".//collision")) == 9
    assert {mesh.get("filename") for mesh in robot.findall(".//mesh")} == {
        f"meshes/{name}"
        for name in (
            "base.stl",
            "link1.stl",
            "link2.stl",
            "link3.stl",
            "link4.stl",
            "link5.stl",
            "gripper.stl",
            "tip_left.stl",
            "tip_right.stl",
        )
    }
    for name in ("dof_joint7", "dof_joint8"):
        joint = next(joint for joint in robot.findall("joint") if joint.get("name") == name)
        assert joint.find("limit").get("lower") == "-0.0475"
        assert joint.find("limit").get("upper") == "0.0"
    assert (output / "LICENSE-I2RT").read_text() == "MIT fixture\n"
    assert yaml.safe_load((output / "robot_config.yml").read_text())["base_link"] == "base"


def test_yam_initial_state_and_control_seed_preserve_partial_opening():
    config_source = ast.parse((ROOT / "src/robodojo/sim/environment/robot_manager/robot_config/yam.py").read_text())
    values = {}
    for node in ast.walk(config_source):
        if not isinstance(node, ast.Dict):
            continue
        try:
            candidate = ast.literal_eval(node)
        except (TypeError, ValueError):
            continue
        if isinstance(candidate, dict) and {"dof_joint7", "dof_joint8"}.issubset(candidate):
            values = {key: candidate[key] for key in ("dof_joint7", "dof_joint8")}
            break
    assert values == {"dof_joint7": -0.02, "dof_joint8": -0.02}

    manager_source = ast.parse((ROOT / "src/robodojo/sim/environment/robot_manager/robot_manager.py").read_text())
    method = next(
        node
        for node in ast.walk(manager_source)
        if isinstance(node, ast.FunctionDef) and node.name == "set_robot_init_state"
    )
    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
    assert any(isinstance(call.func, ast.Attribute) and call.func.attr == "get_end_effector_real_val" for call in calls)


@pytest.mark.parametrize("physical", [-0.0475, -0.02, 0.0])
def test_yam_gripper_contract_round_trip(physical):
    tooling = yaml.safe_load((ROOT / "configs/tooling/yam.yml").read_text())
    robot = tooling["robot_config"]
    lower, upper = robot["gripper_scale"]
    assert robot["gripper_move"]["sign"] == -1.0

    policy = (upper - physical) / (upper - lower)
    restored = (1.0 - policy) * (upper - lower) + lower

    assert policy == pytest.approx(-physical / 0.0475)
    assert restored == pytest.approx(physical)
