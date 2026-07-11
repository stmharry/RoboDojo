import os

import numpy as np

from utils.pipeline_utils import get_embodiment_config_by_robot_type


class _OpenArm:
    side = ""

    def __init__(self, cfg: dict):
        self.robot_type = cfg.get("robot_type")
        self.robot_name = cfg.get("robot_name")
        self.is_coupled = True
        self.default_root_pos = cfg.get("default_root_pos")
        self.default_root_rot = cfg.get("default_root_rot")
        self.grasp_perfect_direction = cfg.get("grasp_perfect_direction")
        self.SceneCfg = None
        self.static_camera_list = None
        self.robot_cfg = get_embodiment_config_by_robot_type(self.robot_type, self.robot_name)
        self.robot_file = self.robot_cfg["robot_file"]
        args = self.robot_cfg["robot_config"]
        side_args = args[self.side]
        self.robot_args = side_args
        self.urdf_path = os.path.join(self.robot_file, args.get("urdf_path", "openarm.urdf"))
        self.srdf_path = ""
        self.curobo_yml_path = os.path.join(self.robot_file, "curobo.yml")
        self.ee_joint_name = side_args["ee_joints"]
        self.ee_link_name = side_args["ee_link"]
        self.arm_joints_name = side_args["arm_joints_name"]
        self.gripper_move = side_args["gripper_move"]
        self.gripper_joints_name = side_args["gripper_joints_name"]
        self.gripper_bias = side_args.get("gripper_bias", 0.0)
        self.gripper_scale = side_args["gripper_scale"]
        self.physical_gripper_interface = bool(side_args.get("physical_gripper_interface", False))
        self.base_link = args.get("base_link", "openarm_link0")
        self.delta_matrix = np.asarray(side_args.get("delta_matrix", np.eye(3)))
        self.inv_delta_matrix = np.linalg.inv(self.delta_matrix)
        self.global_trans_matrix = np.asarray(side_args.get("global_trans_matrix", np.eye(3)))
        self.grasp_camera_reference_axis = side_args.get("grasp_camera_reference_axis", [1, 0, 0])
        self.ee_type = "gripper"
        self.rotate_lim = [0, 0]
        self.entity_origin_pose = self.default_root_pos + self.default_root_rot
        self.camera = side_args.get("camera")
        self.camera_mount_links = side_args.get("camera_mount_links", {})
        self.mesh_dir = self.robot_file
        self.save_gripper_joints_name = side_args.get("save_gripper_joints_name", self.gripper_joints_name)


class LeftOpenArm(_OpenArm):
    side = "left"


class RightOpenArm(_OpenArm):
    side = "right"
