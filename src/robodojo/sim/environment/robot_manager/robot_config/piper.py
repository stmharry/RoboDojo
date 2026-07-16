from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

from robodojo.sim.environment.global_configs import ROBOTS_PATH

LEFT_MEDIAN = [
    -0.09788833134626188,
    0.8574327545542113,
    -1.1794956420880138,
    0.9722126642074953,
    0.1472871958280768,
    -1.5462758390893823,
]


def get_robot_config(usd_asset: str = "Piper.usd"):
    """Return the two-instance-compatible articulation config for one PiPER."""
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ROBOTS_PATH}/piper/{usd_asset}",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                **{f"joint{index + 1}": value for index, value in enumerate(LEFT_MEDIAN)},
                "joint7": 0.00773373982178254,
                "joint8": -0.00773373982178254,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                effort_limit_sim=100.0,
                stiffness=80.0,
                damping=4.0,
                armature=0.01,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["joint[7-8]"],
                effort_limit_sim=10.0,
                stiffness=1200.0,
                damping=30.0,
                armature=0.02,
            ),
        },
    )
