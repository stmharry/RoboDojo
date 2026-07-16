import os

import numpy as np

from robodojo.sim.utils.pipeline_utils import get_embodiment_config_by_robot_type


class PiPER:
    """AgileX PiPER arm with a physical half-jaw gripper interface."""

    def __init__(self, cfg: dict):
        self.robot_type = cfg.get("robot_type")
        self.robot_name = cfg.get("robot_name")
        self.is_coupled = False
        self.default_root_pos = cfg.get("default_root_pos")
        self.default_root_rot = cfg.get("default_root_rot")
        self.grasp_perfect_direction = cfg.get("grasp_perfect_direction")
        self.SceneCfg = None
        self.static_camera_list = None
        self.robot_cfg = get_embodiment_config_by_robot_type(self.robot_type, self.robot_name)
        self.robot_file = self.robot_cfg["robot_file"]
        self.robot_args = self.robot_cfg["robot_config"]
        self.urdf_path = os.path.join(self.robot_file, self.robot_args["urdf_path"])
        self.srdf_path = ""
        self.curobo_yml_path = os.path.join(self.robot_file, "curobo.yml")
        self.ee_joint_name = self.robot_args["ee_joints"]
        self.ee_link_name = self.robot_args["ee_link"]
        self.arm_joints_name = self.robot_args["arm_joints_name"]
        self.gripper_move = self.robot_args["gripper_move"]
        self.gripper_joints_name = self.robot_args["gripper_joints_name"]
        self.gripper_bias = self.robot_args.get("gripper_bias", 0.0)
        self.gripper_scale = self.robot_args["gripper_scale"]
        self.physical_gripper_interface = bool(self.robot_args.get("physical_gripper_interface", True))
        self.base_link = self.robot_args.get("base_link", "base_link")
        self.delta_matrix = np.asarray(self.robot_args.get("delta_matrix", np.eye(3)))
        self.inv_delta_matrix = np.linalg.inv(self.delta_matrix)
        self.global_trans_matrix = np.asarray(self.robot_args.get("global_trans_matrix", np.eye(3)))
        self.grasp_camera_reference_axis = self.robot_args.get("grasp_camera_reference_axis", [1, 0, 0])
        self.ee_type = self.robot_args.get("ee_type", "gripper")
        self.rotate_lim = self.robot_args.get("rotate_lim", [0, 0])
        self.entity_origin_pose = self.default_root_pos + self.default_root_rot
        self.camera = self.robot_args.get("camera")
        self.camera_mount_links = self.robot_args.get("camera_mount_links", {})
        self.mesh_dir = self.robot_args.get("mesh_dir", self.robot_file)
        self.save_gripper_joints_name = self.robot_args.get("save_gripper_joints_name", self.gripper_joints_name)
