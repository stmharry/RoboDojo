from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

from robodojo.sim.environment.global_configs import ROBOTS_PATH


def get_robot_config(usd_asset: str = "YAM.usd"):
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ROBOTS_PATH}/yam/{usd_asset}",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "dof_joint1": 0.0,
                "dof_joint2": 0.0,
                "dof_joint3": 0.0,
                "dof_joint4": 0.0,
                "dof_joint5": 0.0,
                "dof_joint6": 0.0,
                "dof_joint7": -0.0475,
                "dof_joint8": -0.0475,
            },
        ),
        actuators={
            "arm_dm4340": ImplicitActuatorCfg(
                joint_names_expr=["dof_joint[1-3]"],
                effort_limit_sim=28.0,
                stiffness=40.0,
                damping=2.5,
                armature=0.032,
            ),
            "arm_joint4": ImplicitActuatorCfg(
                joint_names_expr=["dof_joint4"],
                effort_limit_sim=10.0,
                stiffness=20.0,
                damping=0.5,
                armature=0.0018,
            ),
            "arm_dm4310": ImplicitActuatorCfg(
                joint_names_expr=["dof_joint[5-6]"],
                effort_limit_sim=10.0,
                stiffness=10.0,
                damping=1.0,
                armature=0.0018,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["dof_joint7", "dof_joint8"],
                effort_limit_sim=40.0,
                stiffness=2000.0,
                damping=40.0,
                armature=0.1,
            ),
        },
    )
