import numpy as np
import pytest

from robodojo.sim.scene_export.contracts import (
    camera_axes,
    exact_simulation_steps,
    forward_ray_plane_intersection,
    geometric_cloth_support,
    project_points_to_camera,
    vector_drift,
)
from robodojo.sim.scene_export.visual_audit import _simulation_step_seconds


def test_visual_audit_duration_requires_exact_simulator_steps():
    assert exact_simulation_steps(2.0, 1.0 / 240.0) == 480
    assert exact_simulation_steps(2.0, 0.004) == 500
    with pytest.raises(ValueError, match="not an integral number"):
        exact_simulation_steps(2.0, 0.03)


def test_visual_audit_hold_uses_physics_dt_without_decimation_scaling():
    direct_env = type("DirectEnv", (), {"cfg": type("Cfg", (), {"decimation": 4})(), "physics_dt": 0.005})()
    env = type("Env", (), {"sim": type("Wrapper", (), {"unwrapped": direct_env})(), "dt": 0.01})()
    physics_dt, decimation, hold_step_seconds = _simulation_step_seconds(env)
    assert (physics_dt, decimation, hold_step_seconds) == (0.005, 4, 0.005)
    assert exact_simulation_steps(2.0, hold_step_seconds) == 400


def test_usd_camera_axes_and_pinhole_cloth_projection():
    camera_to_world = np.eye(4)
    assert camera_axes(camera_to_world) == {
        "right_world": [1.0, 0.0, 0.0],
        "up_world": [0.0, 1.0, 0.0],
        "forward_world": [-0.0, -0.0, -1.0],
    }
    projection = project_points_to_camera(
        [[0.0, 0.0, -1.0], [0.5, 0.0, -1.0], [0.0, 0.0, 1.0]],
        camera_to_world,
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        [100, 100],
    )
    assert projection["in_front_count"] == 2
    assert projection["visible_count"] == 1
    assert projection["visible_fraction"] == pytest.approx(1.0 / 3.0)
    assert projection["visible_pixel_bounds"] == {"min_xy": [50.0, 50.0], "max_xy": [50.0, 50.0]}
    assert projection["visible_normalized_bounds"] == {
        "min_xy": [0.5, 0.5],
        "max_xy": [0.5, 0.5],
    }
    assert forward_ray_plane_intersection(camera_to_world, -2.0) == {
        "plane_z_world_m": -2.0,
        "hit_in_front": True,
        "distance_m": 2.0,
        "point_world_m": [0.0, 0.0, -2.0],
    }


def test_geometric_support_is_explicitly_distinct_from_contact_measurement():
    supported = geometric_cloth_support(
        [[0.0, 0.0, 0.765], [0.1, 0.0, 0.775], [-0.1, 0.0, 0.77]],
        0.765,
    )
    assert supported["geometrically_supported"] is True
    assert supported["contact_measurement_available"] is False
    assert supported["contact_count"] is None

    unavailable = geometric_cloth_support(np.empty((0, 3)), None)
    assert unavailable["geometrically_supported"] is None
    assert unavailable["particle_fraction_near_surface"] is None


def test_vector_drift_reports_maximum_and_rms_without_mutating_inputs():
    before = np.array([0.0, -0.02, 1.0])
    after = np.array([0.01, -0.018, 0.99])
    result = vector_drift(before, after)
    assert result["max_abs"] == pytest.approx(0.01)
    assert result["rms"] == pytest.approx(np.sqrt((0.01**2 + 0.002**2 + 0.01**2) / 3.0))
    assert before.tolist() == [0.0, -0.02, 1.0]
