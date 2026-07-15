import ast
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from omegaconf import OmegaConf
import pytest
import yaml

from robodojo.core.models import PreflightRequest
from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles import bind_policy_contract, load_environment_profile, load_scene_profile
from robodojo.sim.camera_template import YAM_TOP, YAM_WRIST, resolve_pinhole_lens
from robodojo.sim.environment.camera_manager.mount_registry import (
    apply_optical_roll,
    convert_mount_orientation,
    mount_orientation,
    orientation_quaternion,
    require_camera_mount_prim,
    robot_link_prim_path,
)
from robodojo.sim.environment.camera_manager.rig_spec import normalize_camera_rig
from robodojo.sim.environment.description_manager.desc_manager import DescManager
from robodojo.sim.general_pickup_contract import STEP_LIMIT, instruction_templates
from robodojo.sim.utils.pipeline_utils import process_config
from robodojo.workflows.assets_yam import (
    _appearance_contract,
    _finger_collider_contract,
    _fixed_camera_frame_contract,
    _visual_proxy_contracts,
    derive_yam_urdf,
)

ROOT = Path(__file__).resolve().parents[1]


def test_named_yam_profiles_inherit_one_30hz_policy_contract():
    paths = RepositoryPaths.resolve(ROOT)
    with pytest.raises(ValueError, match="internal contract"):
        load_environment_profile(paths, "bimanual_yam")
    profile = load_environment_profile(paths, "bimanual_yam_molmoact2")
    assert profile.document.config.model_dump() == {
        "sim": "real_time_30hz",
        "robot": "dual_yam_molmoact2",
        "camera": "bimanual_yam_molmoact2",
    }
    assert profile.document.extends == "bimanual_yam"
    assert profile.policy_contract == "bimanual_yam"
    assert profile.document.observation["collect_freq"] == 30
    assert profile.num_envs == 1

    robot_info = yaml.safe_load((ROOT / "configs/robot/_robot_info.json").read_text())
    expected_dimensions = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
    assert robot_info["dual_yam"] == expected_dimensions
    assert robot_info["dual_yam_molmoact2"] == expected_dimensions
    assert robot_info["dual_yam_moonlake_office"] == expected_dimensions

    request = PreflightRequest(
        policy_dir=ROOT / "XPolicyLab/policy/MolmoACT2",
        task="general_pickup",
        checkpoint="molmoact2_bimanual_yam",
        policy_env="molmoact2",
        env_config="bimanual_yam_molmoact2",
        action_type="joint",
    )
    assert bind_policy_contract(paths, request).policy_contract == "bimanual_yam"


def test_task_instruction_is_independent_of_environment_and_scene_profiles():
    profile = load_environment_profile(RepositoryPaths.resolve(ROOT), "bimanual_yam_molmoact2")
    layout_manager = SimpleNamespace(get_label_descriptions=lambda **kwargs: [])

    class FakeEnv:
        scene_manager = SimpleNamespace(layout_manager=layout_manager)

        def __init__(self, task_name, eval_cfg):
            self.task_name = task_name
            self.eval_cfg = eval_cfg

        def gen_instruction(self, env_idx):
            return ["Fold the clothes neatly."]

    yam_manager = DescManager(num_envs=1, description_cfg={"seen": 1}, desc_type="seen")
    yam_manager.initialize(FakeEnv("fold_clothes", profile.payload))
    assert yam_manager.templates == [["Fold the clothes neatly."]]


def test_general_pickup_preserves_public_checkpoint_prompt_contract():
    assert STEP_LIMIT == 400
    assert instruction_templates("molmo_yam") == ["Put everything into the box."]
    assert instruction_templates("moonlake_office") == ["Pick up the letter block by 10 cm."]
    assert instruction_templates() == ["Pick up the <target> by 10 cm."]

    layout_manager = SimpleNamespace(get_label_descriptions=lambda **kwargs: [])

    fake_env = SimpleNamespace(
        scene_manager=SimpleNamespace(layout_manager=layout_manager),
        gen_instruction=lambda env_idx: instruction_templates("molmo_yam"),
    )
    manager = DescManager(num_envs=1, description_cfg={"seen": 1}, desc_type="seen")
    manager.initialize(fake_env)
    assert manager.get_one_description() == ["Put everything into the box."]


def test_molmo_yam_scene_profile_uses_model_aligned_bundled_layouts():
    profile = load_scene_profile(RepositoryPaths.resolve(ROOT), "molmo_yam")
    assert profile.document.layout_set == "molmo_yam"
    assert profile.document.layout_source == "bundled"
    assert profile.document.component == "molmo_yam"
    [recipe] = profile.document.task_assets["fold_clothes"]
    assert recipe.transform == "yam_short_sleeve_v1"
    assert recipe.source.index == 9
    assert recipe.destination.index == 12
    layout_root = ROOT / "configs/layout/molmo_yam/0"
    assert {path.name for path in layout_root.glob("*.json")} == {
        "fold_clothes_0.json",
        "general_pickup_0.json",
    }
    pickup = json.loads((layout_root / "general_pickup_0.json").read_text())
    assert pickup["Rigid"]["ball"][0]["visual"]["color"] == [0.35, 0.65, 0.08]
    basket = pickup["Geometry"]["basket"][0]
    assert basket["category_idx"] == 2
    assert basket["label"] == "box"
    assert basket["default_pos"] == [0.0, 0.05, 0.8036]
    fold = json.loads((layout_root / "fold_clothes_0.json").read_text())
    garment = fold["Garment"]["Top_Long"][0]
    assert garment["category_idx"] == 12
    assert garment["default_pos"] == [0.0, -0.05, 0.95]
    assert garment["physics"]["garment_config"]["total_mass"] == pytest.approx(0.2)
    assert garment["physics"]["garment_config"]["stretch_stiffness"] == pytest.approx(1e4)
    assert garment["visual"]["color"] == [0.95, 0.95, 0.95]


def test_fold_clothes_does_not_replace_yam_profile_components():
    profile = load_environment_profile(RepositoryPaths.resolve(ROOT), "bimanual_yam_molmoact2")
    scene = load_scene_profile(RepositoryPaths.resolve(ROOT), "default")
    payload = profile.payload
    env_cfg = OmegaConf.create(
        {
            "sim": yaml.safe_load(profile.component_paths["sim"].read_text()),
            "scene": scene.component,
            "robot": yaml.safe_load(profile.component_paths["robot"].read_text()),
            "camera": yaml.safe_load(profile.component_paths["camera"].read_text()),
            "task_env": yaml.safe_load((ROOT / "configs/task/fold_clothes.yml").read_text()),
            "eval_cfg": payload,
        }
    )
    processed, eval_num = process_config(env_cfg, "fold_clothes")
    assert [robot.robot_name for robot in processed.robot.robots] == ["yam", "yam"]
    assert processed.camera.camera_rig.profile_id == "bimanual_yam_molmoact2"
    assert processed.sim.render_interval == 8
    assert processed.sim.frequency_settings["/app/runLoops/main/rateLimitFrequency"] == 150
    assert processed.sim.device == "cpu"
    assert processed.sim.use_fabric is False
    assert eval_num == 25


def test_dual_yam_order_pose_and_no_planner_are_explicit():
    robots = yaml.safe_load((ROOT / "configs/robot/dual_yam_molmoact2.yml").read_text())["robots"]
    assert [robot["default_root_pos"] for robot in robots] == [
        [-0.24, -0.45, 0.765],
        [0.24, -0.45, 0.765],
    ]
    assert all(robot["default_root_rot"] == pytest.approx([0.0, 0.0, 0.0, 1.0]) for robot in robots)
    assert all(robot["robot_name"] == "yam" for robot in robots)
    assert all(robot["usd_asset"] == "YAM_molmoact2.usd" for robot in robots)
    assert all(robot["coupled"] is False and robot["need_planner"] is False for robot in robots)


def test_yam_camera_rig_matches_embodiment_contract_and_runtime_templates():
    rig = normalize_camera_rig(yaml.safe_load((ROOT / "configs/camera/bimanual_yam_molmoact2.yml").read_text()))
    assert rig.profile_id == "bimanual_yam_molmoact2"
    assert rig.default_frequency == 30
    assert [camera.observation_key for camera in rig.cameras] == [
        "cam_head",
        "cam_left_wrist",
        "cam_right_wrist",
    ]
    top, left, right = rig.cameras
    assert top.sensor["stream_resolution"] == [640, 360]
    assert top.mount["position"] == [-0.037, -0.30, 1.635]
    assert top.mount["orientation"] == pytest.approx([0.54167522, -0.45451948, 0.45451948, 0.54167522])
    assert top.mount["pose_convention"] == "sapien_robotics"
    assert top.projection["fx"] == pytest.approx(462.1386898729645)
    assert top.projection["horizontal_fov_deg"] == 69.4
    assert [left.mount["target"], right.mount["target"]] == [
        "robot0/wrist_camera_mount",
        "robot1/wrist_camera_mount",
    ]
    assert "hardware" not in top.mount
    for wrist in (left, right):
        assert wrist.mount["position"] == [0.0, 0.09, 0.06]
        assert wrist.mount["orientation"] == pytest.approx([0.61237243, -0.35355339, -0.35355340, -0.61237244])
        assert wrist.mount["pose_convention"] == "sapien_robotics"
        assert wrist.projection["fx"] == pytest.approx(337.20964008990796)
        assert wrist.projection["horizontal_fov_deg"] == 87.0
        assert wrist.mount["hardware"] == {
            "enabled": True,
            "asset": "Robots/yam/D405_proxy_molmoact2.usd",
            "collision": False,
            "camera_frame": "OpticalFrame",
        }

    assert 640 * YAM_TOP["focal_length"] / YAM_TOP["horizontal_aperture"] == pytest.approx(top.projection["fx"])
    assert 640 * YAM_WRIST["focal_length"] / YAM_WRIST["horizontal_aperture"] == pytest.approx(left.projection["fx"])


def test_yam_setup_profiles_keep_their_source_camera_aspect_ratios():
    molmo = normalize_camera_rig(
        yaml.safe_load((ROOT / "configs/camera/bimanual_yam_molmoact2.yml").read_text())
    )
    moonlake = normalize_camera_rig(
        yaml.safe_load((ROOT / "configs/camera/bimanual_yam_moonlake_office.yml").read_text())
    )

    assert {tuple(camera.sensor["stream_resolution"]) for camera in molmo.cameras} == {(640, 360)}
    assert {tuple(camera.sensor["stream_resolution"]) for camera in moonlake.cameras} == {(640, 480)}

    classic_lens = resolve_pinhole_lens(YAM_TOP, molmo.cameras[0].runtime_camera(), (640, 360))
    moonlake_lens = resolve_pinhole_lens(YAM_TOP, moonlake.cameras[0].runtime_camera(), (640, 480))
    assert classic_lens[1] == pytest.approx(moonlake_lens[1])
    assert classic_lens[2] == pytest.approx(classic_lens[1] * 360 / 640)
    assert moonlake_lens[2] == pytest.approx(moonlake_lens[1] * 480 / 640)


def test_yam_tooling_and_embodiment_reference_are_revision_pinned():
    tooling = yaml.safe_load((ROOT / "configs/tooling/yam.yml").read_text())
    reference = yaml.safe_load((ROOT / "configs/reference/bimanual_yam.yml").read_text())
    assert tooling["sources"]["i2rt"]["revision"] == "ac096928d6899ddf852a71c5e8fbaa6055cd9745"
    assert reference["sources"]["historical_simulator_contract"]["revision"] == (
        "c2282820f9b188b60e66ea1636b3efd81c45cbb4"
    )
    assert reference["state_action_contract"]["dimension"] == 14
    assert reference["state_action_contract"]["gripper"]["formula"] == "g=-q/0.0475"
    assert "policy_checkpoint" not in reference["sources"]
    assert "predicted_horizon" not in reference["state_action_contract"]
    assert "executed_horizon" not in reference["state_action_contract"]
    assert "joint_convention_bridge" not in reference["state_action_contract"]

    robot = tooling["robot_config"]
    assert robot["arm_joints_name"] == [f"dof_joint{i}" for i in range(1, 7)]
    assert robot["gripper_joints_name"] == ["dof_joint7", "dof_joint8"]
    assert robot["gripper_move"] == {"base": "dof_joint7", "sign": -1.0, "mimic": ["dof_joint8", 1.0, 0.0]}
    assert robot["gripper_scale"] == [-0.0475, 0.0]
    assert robot["camera_mount_links"] == {"wrist_camera_mount": "gripper/wrist_camera_mount"}

    reference_source = tooling["sources"]["historical_camera_mount_reference"]
    assert reference_source["repository"] == ("https://huggingface.co/datasets/TreeePlanter/molmoact2-sim-eval-assets")
    assert reference_source["revision"] == "9332a64224ff0a813d9f77bd377b845270232513"
    assert reference_source["license"] == "undeclared"
    assert reference_source["usage"] == "reference_only"
    assert reference_source["files"] == {
        "bimanual_model": {
            "path": "assets/yam/yam_mujoco/bimanual_yam_linear_flattened.xml",
            "sha256": "459f1801fae5618e7caf3c44f257a53d61974e7708a14d83c7b60a60b65e374a",
        },
        "gripper_mesh": {
            "path": "assets/yam/yam_mujoco/assets/gripper_linear.obj",
            "sha256": "31d70e2129a3ab8e85f391785cae5bba4163e1e16fb02a3988c5ac1c549c9d78",
        },
    }
    finger_colliders = _finger_collider_contract(tooling)
    assert set(finger_colliders) == {"tip_left", "tip_right"}
    assert all(len(colliders) == 3 for colliders in finger_colliders.values())
    assert all(
        collider["size"][2] == pytest.approx(0.004) for colliders in finger_colliders.values() for collider in colliders
    )
    appearance_source = tooling["sources"]["hardware_appearance_reference"]
    assert appearance_source == {
        "repository": "https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset",
        "revision": "e9f21ae15074330839f2ac25ed4b49d76dfa1f9c",
        "license": "Apache-2.0",
        "usage": "reference_only",
        "derivation": "visually match coarse YAM link colors from public top and wrist RGB training frames",
        "inspected_modalities": [
            "observation.images.top",
            "observation.images.left",
            "observation.images.right",
        ],
    }
    assert "author_yam_hardware_preview_materials" in tooling["asset"]["transformations"]
    visual_links = ["base", "gripper", "link1", "link2", "link3", "link4", "link5", "tip_left", "tip_right"]
    appearance = _appearance_contract(tooling, visual_links)
    assert appearance == _appearance_contract(tooling, list(reversed(visual_links)))
    assert appearance["derivation_source"] == "hardware_appearance_reference"
    assert appearance["shader"] == "UsdPreviewSurface"
    assert appearance["color_space"] == "linear_rgb"
    assert appearance["link_materials"] == {
        "base": "charcoal",
        "gripper": "charcoal",
        "link1": "charcoal",
        "link2": "off_white",
        "link3": "light_gray",
        "link4": "charcoal",
        "link5": "charcoal",
        "tip_left": "charcoal",
        "tip_right": "charcoal",
    }
    assert appearance["palette"]["charcoal"]["diffuse_color"] == [0.03, 0.035, 0.04]
    assert appearance["palette"]["off_white"]["diffuse_color"] == [0.78, 0.8, 0.82]
    assert appearance["palette"]["light_gray"]["diffuse_color"] == [0.52, 0.55, 0.58]
    assert {name: material["sha256"] for name, material in appearance["palette"].items()} == {
        "charcoal": "a991aa064b75bec04a85965f2c3b5d45385b35e2bf9f8d848f2af24d5521861f",
        "light_gray": "413bf0e0234400226e61cbae77101bf731fa85fdb3f0863e421591607a05054d",
        "off_white": "cf69ba4a745fe50be4daf688ab02adae8e8887d7b6fc32ffa8064c6ca90d7825",
    }
    assert tooling["sources"]["realsense_d405"] == {
        "product_page": "https://www.realsenseai.com/products/stereo-depth-camera-d405/",
        "datasheet": (
            "https://realsenseai.com/wp-content/uploads/dlm_uploads/2025/08/"
            "Intel-RealSense-D400-Series-Datasheet-August-2025.pdf"
        ),
        "datasheet_revision": "017",
        "usage": "geometry_reference",
        "nominal_dimensions_m": {"width": 0.042, "height": 0.042, "depth": 0.023},
    }
    visual_proxies = _visual_proxy_contracts(tooling, appearance)
    assert set(visual_proxies) == {"molmoact2", "moonlake_office"}
    assert visual_proxies["molmoact2"]["output"] == "D405_proxy_molmoact2.usd"
    assert visual_proxies["molmoact2"]["materials"] == {"housing": "light_gray", "detail": "charcoal"}
    assert visual_proxies["moonlake_office"]["output"] == "D405_proxy_moonlake_office.usd"
    assert visual_proxies["moonlake_office"]["materials"] == {"housing": "charcoal", "detail": "charcoal"}
    assert all(proxy["physical"] is False for proxy in visual_proxies.values())
    assert "author_nonphysical_d405_visual_proxy" in tooling["asset"]["transformations"]
    assert _fixed_camera_frame_contract(tooling) == [
        {
            "name": "wrist_camera_mount",
            "parent": "gripper",
            "position": [0.0, 0.0, 0.0005],
            "orientation": [0.0, 0.7071067811865476, -0.7071067811865476, 0.0],
            "derivation_source": "historical_camera_mount_reference",
            "physical": False,
            "path": "gripper/wrist_camera_mount",
        }
    ]


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
    assert contract["collision_geometry_count"] == 13
    assert contract["visual_links"] == [
        "base",
        "gripper",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "tip_left",
        "tip_right",
    ]
    assert contract["links_without_visuals"] == ["root"]
    collisions = robot.findall(".//collision")
    assert len(collisions) == 13
    assert len(robot.findall(".//collision/geometry/mesh")) == 7
    assert len(robot.findall(".//collision/geometry/box")) == 6
    for link_name in ("tip_left", "tip_right"):
        link = next(link for link in robot.findall("link") if link.get("name") == link_name)
        assert [collision.get("name") for collision in link.findall("collision")] == ["pad_0", "pad_1", "pad_2"]
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
    assert contract["visual_proxies"]["molmoact2"]["output"] == "D405_proxy_molmoact2.usd"
    assert contract["visual_proxies"]["moonlake_office"]["output"] == "D405_proxy_moonlake_office.usd"


def test_yam_initial_state_and_control_seed_use_maximum_opening():
    config_source = ast.parse((ROOT / "src/robodojo/sim/environment/robot_manager/robot_config/yam.py").read_text())
    values = {}
    for node in ast.walk(config_source):
        if not isinstance(node, ast.Dict):
            continue
        try:
            candidate = ast.literal_eval(node)
        except (TypeError, ValueError):
            continue
        if isinstance(candidate, dict) and {f"dof_joint{index}" for index in range(1, 9)}.issubset(candidate):
            values = candidate
            break
    assert values == {
        "dof_joint1": 0.0,
        "dof_joint2": 0.0,
        "dof_joint3": 0.0,
        "dof_joint4": 0.0,
        "dof_joint5": 0.0,
        "dof_joint6": 0.0,
        "dof_joint7": -0.0475,
        "dof_joint8": -0.0475,
    }

    reference = yaml.safe_load((ROOT / "configs/reference/bimanual_yam.yml").read_text())
    assert reference["initial_state"] == {
        "name": "home_max_open",
        "per_arm_joint_position": [0.0] * 6,
        "per_finger_position_m": -0.0475,
    }

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


def test_sapien_camera_axes_are_converted_before_optical_roll():
    raw = [0.54167522, -0.45451948, 0.45451948, 0.54167522]
    converted = convert_mount_orientation(raw, "sapien_robotics")
    assert converted == pytest.approx([0.9961947, 0.08715574, 0.0, 0.0], abs=1e-8)

    from scipy.spatial.transform import Rotation

    raw_rotation = Rotation.from_quat(orientation_quaternion(raw)[[1, 2, 3, 0]])
    converted_rotation = Rotation.from_quat(converted[[1, 2, 3, 0]])
    assert converted_rotation.apply([0.0, 0.0, -1.0]) == pytest.approx(raw_rotation.apply([1.0, 0.0, 0.0]))
    assert converted_rotation.apply([0.0, 1.0, 0.0]) == pytest.approx(raw_rotation.apply([0.0, 0.0, 1.0]))
    assert mount_orientation(raw, "sapien_robotics", 90.0) == pytest.approx(apply_optical_roll(converted, 90.0))


def test_camera_pose_convention_default_and_validation_are_backward_compatible():
    config = yaml.safe_load((ROOT / "configs/camera/bimanual_yam_molmoact2.yml").read_text())
    mount = config["camera_rig"]["cameras"]["cam_head"]["mount"]
    mount.pop("pose_convention")
    normalized = normalize_camera_rig(config)
    assert normalized.cameras[0].runtime_camera()["mount_pose_convention"] == "isaac_usd"

    mount["pose_convention"] = "unknown_camera_axes"
    with pytest.raises(ValueError, match="invalid mount pose convention"):
        normalize_camera_rig(config)

    mount["pose_convention"] = "sapien_robotics"
    mount["orientation"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="require a scalar-first quaternion"):
        normalize_camera_rig(config)


def test_nested_yam_camera_mount_alias_resolves_to_generated_frame():
    tooling = yaml.safe_load((ROOT / "configs/tooling/yam.yml").read_text())
    nested_link = tooling["robot_config"]["camera_mount_links"]["wrist_camera_mount"]
    assert robot_link_prim_path(3, "robot0", nested_link) == ("/World/envs/env_3/robot0/gripper/wrist_camera_mount")
    with pytest.raises(ValueError, match="rebuild the embodiment asset"):
        require_camera_mount_prim("/World/envs/env_3/robot0/gripper/wrist_camera_mount", lambda _: False)
