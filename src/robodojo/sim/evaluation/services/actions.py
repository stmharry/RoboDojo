from __future__ import annotations

from copy import deepcopy
import logging

import numpy as np

logger = logging.getLogger(__name__)


class ActionsService:
    def get_action_type(self, action):
        action_type = []
        for robot in self.robot_manager.robot_list:
            if robot.type != "target":
                continue
            key_name = self.robot_manager.process_name(robot.arm_name)
            if key_name in action.keys() and "joint" not in action_type:
                action_type.append("joint")
            key_name = f"{robot.arm_name.split('_')[0]}_ee_pose"
            if key_name in action.keys() and "ee" not in action_type:
                action_type.append("ee")
        if len(action_type) == 0:
            raise ValueError(f"Cannot infer action type from action dict keys: {action.keys()}")
        if len(action_type) > 1:
            raise ValueError(f"Multiple action types found in action dict keys: {action.keys()}")
        return action_type[0]

    def _joint_action_to_control_info(self, action):
        """Convert a public joint action to the robot manager's target schema."""
        control_info = {}
        for robot in self.robot_manager.robot_list:
            if robot.type != "target":
                continue
            arm_key = self.robot_manager.process_name(robot.arm_name)
            control_info[arm_key] = {"position": action[arm_key]}

            ee_key = self.robot_manager.process_name(robot.gripper_name)
            if robot.ee_type == "gripper":
                val = deepcopy(action[ee_key][0])
                if getattr(robot, "physical_gripper_interface", False):
                    val = np.clip(val, robot.gripper_scale[0], robot.gripper_scale[1])
                else:
                    val = np.clip(val, 0, 1)
                    if robot.gripper_move["sign"] == 1:
                        val = val * (robot.gripper_scale[1] - robot.gripper_scale[0]) + robot.gripper_scale[0]
                    else:
                        val = (1 - val) * (robot.gripper_scale[1] - robot.gripper_scale[0]) + robot.gripper_scale[0]
                control_info[ee_key] = {
                    "position": [
                        val,
                        val * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2],
                    ]
                }
            elif robot.ee_type == "hand":
                control_info[ee_key] = {"position": action[ee_key]}
        return control_info

    def take_action(self, action):
        self.validate_action_dict(action)
        self.take_action_batch([action], env_idx_list=[0])

    def take_interpolated_action(self, actions, physics_ticks):
        """Execute one 30 Hz policy action as explicit higher-rate targets.

        LeRobot emits three linearly interpolated targets at 90 Hz. With
        240 Hz simulation physics, the targets are held for 3, 3, and 2
        ticks respectively. The policy step counter and reward advance
        once, after all eight physics ticks.
        """
        if len(actions) != len(physics_ticks) or not actions:
            raise ValueError("actions and physics_ticks must have the same non-zero length")
        if any(not isinstance(ticks, int) or ticks <= 0 for ticks in physics_ticks):
            raise ValueError("physics_ticks must contain positive integers")
        if sum(physics_ticks) != int(self.obs_manager.collect_interval):
            raise ValueError(
                "interpolated targets must span exactly one observation interval: "
                f"got {sum(physics_ticks)}, expected {self.obs_manager.collect_interval}"
            )
        if self.physx_monitor_enabled:
            self._check_physx_broken_envs()
        env_idx = 0
        if self.take_action_cnt[env_idx] == self.step_lim or self.end_flag[env_idx]:
            return

        control_seq = []
        for action, ticks in zip(actions, physics_ticks, strict=True):
            self.validate_action_dict(action)
            if self.get_action_type(action) != "joint":
                raise ValueError("take_interpolated_action supports joint actions only")
            control_info = self._joint_action_to_control_info(action)
            control_seq.extend(deepcopy(control_info) for _ in range(ticks))

        self.take_action_cnt[env_idx] += 1
        logger.debug("env%s step: %s / %s", env_idx, self.take_action_cnt[env_idx], self.step_lim)
        self.robot_manager.control_manager.push([env_idx], [control_seq])
        while not self.have_empty([env_idx]):
            self.step(env_idx_list=[env_idx])
            if self.physx_monitor_enabled:
                self._check_physx_broken_envs()
                self._check_endpose_finite([env_idx])
        self.reward_manager.step(env_idx_list=[env_idx])
        if getattr(self, "interact", False):
            if hasattr(self, "query_support_arm_traj"):
                self.query_support_arm_traj(env_idx=env_idx)
            if hasattr(self, "check_support_arm_stable"):
                self.check_support_arm_stable(env_idx=env_idx)
        self.is_episode_end()

    def take_action_batch(self, actions_list, env_idx_list=None):
        if self.physx_monitor_enabled:
            self._check_physx_broken_envs()
        control_info_list = []
        if env_idx_list is None:
            env_idx_list = list(range(self.num_envs))
        for idx, env_idx in enumerate(env_idx_list):
            action = actions_list[idx]
            self.validate_action_dict(action)
            action_type = self.get_action_type(action)
            if self.take_action_cnt[env_idx] == self.step_lim or self.end_flag[env_idx]:
                continue

            self.take_action_cnt[env_idx] += 1
            logger.debug("env%s step: %s / %s", env_idx, self.take_action_cnt[env_idx], self.step_lim)
            control_info = dict()
            if action_type == "joint":
                control_info = self._joint_action_to_control_info(action)
            elif action_type == "ee":
                for robot in self.robot_manager.robot_list:
                    if robot.type != "target":
                        continue
                    name = robot.arm_name.split("_")[0]
                    key_name = f"{name}_ee_pose"
                    obs_name = self.robot_manager.process_name(robot.arm_name)
                    target_pose = action[key_name]
                    ik_result = self.robot_manager.solve_ik(
                        target_pose=target_pose,
                        env_idx=env_idx,
                        robot=robot,
                    )
                    if ik_result["status"] == "Success":
                        control_info[obs_name] = {
                            "position": ik_result["joint_value"],
                        }

                    key_name = self.robot_manager.process_name(robot.gripper_name)
                    if robot.ee_type == "gripper":
                        val = deepcopy(action[key_name][0])
                        val = np.clip(val, 0, 1)
                        if robot.gripper_move["sign"] == 1:
                            val = val * (robot.gripper_scale[1] - robot.gripper_scale[0]) + robot.gripper_scale[0]
                        else:
                            val = (1 - val) * (robot.gripper_scale[1] - robot.gripper_scale[0]) + robot.gripper_scale[0]
                        vals = [
                            val,
                            val * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2],
                        ]
                        control_info[key_name] = {"position": vals}
                    elif robot.ee_type == "hand":
                        control_info[key_name] = {
                            "position": action[key_name],
                        }
                    else:
                        pass
            control_seq = self.process_control_info(control_info, env_idx)
            control_info_list.append(control_seq)

        if len(control_info_list) > 0:
            self.robot_manager.control_manager.push(env_idx_list, control_info_list)
            while not self.have_empty(env_idx_list):
                self.step(env_idx_list=env_idx_list)
                if self.physx_monitor_enabled:
                    self._check_physx_broken_envs()
                    self._check_endpose_finite(env_idx_list)

        self.reward_manager.step(env_idx_list=env_idx_list)
        if getattr(self, "interact", False):
            if hasattr(self, "query_support_arm_traj"):
                for env_idx in env_idx_list:
                    self.query_support_arm_traj(env_idx=env_idx)
            if hasattr(self, "check_support_arm_stable"):
                for env_idx in env_idx_list:
                    self.check_support_arm_stable(env_idx=env_idx)
        self.is_episode_end()

    def process_control_info(self, control_info, env_idx):
        """Expand one-step `control_info` into a per-step control sequence.

        This method returns a list with `interpolation_nums` control dicts
        (deep copies of the input), then overwrites arm/gripper positions
        frame by frame:

        - First 80% (floor) steps: linear interpolation from current state
            to target state.
        - Last 20% steps: hold the target state.

        Notes:
        - Arm joints are interpolated in joint space.
        - For grippers, interpolation is done on the primary scalar opening,
            then mapped to mimic joints.
        - Gripper values are clamped to `robot.gripper_scale` to avoid
            out-of-range commands.

        Args:
                control_info: One-step target control dict for a single env.
                env_idx: Environment index used to read current robot states.

        Returns:
                List[dict]: A control sequence of length `interpolation_nums`.
        """
        interpolation_nums = int(self.obs_manager.collect_interval)
        if interpolation_nums <= 0:
            return [deepcopy(control_info)]

        control_info_list = [deepcopy(control_info) for _ in range(interpolation_nums)]
        for robot in self.robot_manager.robot_list:
            if robot.type != "target":
                if hasattr(self, "support_arm_action") and len(self.support_arm_action[env_idx]) > 0:
                    for i in range(interpolation_nums):
                        if len(self.support_arm_action[env_idx]) == 0:
                            break
                        control_info_list[i].update(self.support_arm_action[env_idx][0])
                        self.support_arm_action[env_idx].pop(0)

            key_name = self.robot_manager.process_name(robot.arm_name)
            if key_name in control_info.keys():
                position = control_info[key_name]["position"]
                current_position = self.robot_manager.get_joint(robot, env_idx_list=[env_idx])[env_idx]
                if current_position is not None:
                    interp_count = int(np.floor(interpolation_nums * 0.8))

                    current_arr = np.array(current_position)
                    target_arr = np.array(position)

                    for i in range(interp_count):
                        alpha = (i + 1) / (interp_count + 1)
                        interp_pos = (1 - alpha) * current_arr + alpha * target_arr
                        control_info_list[i][key_name]["position"] = interp_pos.tolist()

                    for i in range(interp_count, interpolation_nums):
                        control_info_list[i][key_name]["position"] = target_arr.tolist()

            key_name = self.robot_manager.process_name(robot.gripper_name)
            if key_name in control_info.keys() and robot.ee_type == "gripper":
                position = control_info[key_name]["position"][0]
                current_position = self.robot_manager.get_end_effector_real_val(robot, env_idx_list=[env_idx])[env_idx][
                    0
                ]
                if current_position is not None:
                    interp_count = int(np.floor(interpolation_nums * 0.8))

                    scale = robot.gripper_scale
                    for i in range(interp_count):
                        alpha = (i + 1) / (interp_count + 1)
                        interp_pos = (1 - alpha) * current_position + alpha * position
                        interp_pos = np.clip(interp_pos, scale[0], scale[1])
                        vals = [
                            interp_pos,
                            interp_pos * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2],
                        ]
                        control_info_list[i][key_name]["position"] = vals

                    position = np.clip(position, scale[0], scale[1])
                    vals = [
                        position,
                        position * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2],
                    ]
                    for i in range(interp_count, interpolation_nums):
                        control_info_list[i][key_name]["position"] = vals

        return control_info_list

    def validate_action_dict(self, action_dict: dict) -> None:
        """Validate that a policy action dict uses the expected per-arm keys
        and per-key dimensions for this robot.

        Args:
            action_dict: action mapping returned by the policy. Single-arm
                robots use unprefixed keys (e.g. ``arm_joint_state``,
                ``ee_pose``); bi-manual robots use ``left_``/``right_`` prefixes.

        Raises:
            ValueError: unexpected keys, forbidden prefixes, or wrong dimensions.
            TypeError: a value is not array-like.
        """
        arm_dims = deepcopy(self.robot_action_dim_info["arm_dim"])
        ee_dims = deepcopy(self.robot_action_dim_info["ee_dim"])

        if len(arm_dims) != len(ee_dims):
            raise ValueError(
                f"robot_action_dim_info mismatch: len(arm_dim)={len(arm_dims)} != len(ee_dim)={len(ee_dims)}"
            )

        arm_count = len(arm_dims)

        if arm_count == 1:
            expected = {
                "arm_joint_state": arm_dims[0],
                "ee_joint_state": ee_dims[0],
                "ee_pose": 7,
                "tcp_pose": 7,
                "delta_ee_pose": 7,
            }
            forbidden_prefixes = ("left_", "right_")

        elif arm_count == 2:
            expected = {
                "left_arm_joint_state": arm_dims[0],
                "left_ee_joint_state": ee_dims[0],
                "left_ee_pose": 7,
                "left_tcp_pose": 7,
                "left_delta_ee_pose": 7,
                "right_arm_joint_state": arm_dims[1],
                "right_ee_joint_state": ee_dims[1],
                "right_ee_pose": 7,
                "right_tcp_pose": 7,
                "right_delta_ee_pose": 7,
            }
            forbidden_prefixes = ()
        else:
            raise ValueError(f"Unsupported arm count: {arm_count}")

        if forbidden_prefixes:
            bad_prefixed_keys = [k for k in action_dict.keys() if k.startswith(forbidden_prefixes)]
            if bad_prefixed_keys:
                raise ValueError(f"Single-arm robot should not contain prefixed keys, but got: {bad_prefixed_keys}")
        unexpected_keys = [k for k in action_dict if k not in expected]
        if unexpected_keys:
            raise ValueError(f"Unexpected state keys: {unexpected_keys}")

        for key, expected_dim in expected.items():
            if key not in action_dict.keys():
                continue
            value = action_dict[key]

            if not isinstance(value, (np.ndarray, list, tuple)):
                raise TypeError(f"action_dict['{key}'] must be array-like, got {type(value)}")

            arr = np.asarray(value)

            if arr.ndim != 1:
                raise ValueError(f"action_dict['{key}'] must be 1D, got shape {arr.shape}")

            if arr.shape[0] != expected_dim:
                raise ValueError(f"action_dict['{key}'] dim mismatch: expected {expected_dim}, got shape {arr.shape}")
