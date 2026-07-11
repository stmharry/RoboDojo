from isaaclab_assets.robots.openarm import OPENARM_BI_HIGH_PD_CFG
import numpy as np

from robodojo.client.environment.global_configs import ROBOTS_PATH


def get_robot_config():
    cfg = OPENARM_BI_HIGH_PD_CFG.copy()
    cfg.spawn.usd_path = f"{ROBOTS_PATH}/openarm/openarm_bimanual_cloth_folding.usd"
    right_deg = [-6.546, -2.065, 28.797, 20.819, -23.813, 20.294, -0.667]
    left_deg = [3.945, -0.973, -4.164, 2.983, 7.508, -7.049, -8.338]
    cfg.init_state.joint_pos = {
        **{f"openarm_right_joint{i + 1}": float(np.deg2rad(value)) for i, value in enumerate(right_deg)},
        **{f"openarm_left_joint{i + 1}": float(np.deg2rad(value)) for i, value in enumerate(left_deg)},
        "openarm_right_finger_joint.*": 0.00782,
        "openarm_left_finger_joint.*": 0.01798,
    }
    return cfg
