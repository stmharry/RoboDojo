from robodojo.sim.environment.planner_manager.curobo_planner import (
    _interpolation_buffer_size,
    _lock_cspace_only_joints,
)


def test_lock_cspace_only_joints_uses_retract_positions():
    kinematics = {
        "cspace": {
            "joint_names": ["joint1", "joint2", "joint8", "joint7"],
            "default_joint_position": [0.0, 0.0, -0.021, -0.022],
        },
        "lock_joints": {"joint7": 0.044},
    }

    _lock_cspace_only_joints(kinematics, ["joint1", "joint2"])

    assert kinematics["lock_joints"] == {"joint8": -0.021, "joint7": 0.044}


def test_interpolation_buffer_preserves_default_time_horizon():
    assert _interpolation_buffer_size(0.025) == 1000
    assert _interpolation_buffer_size(1 / 240) == 6000
