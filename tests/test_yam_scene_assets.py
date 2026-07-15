import numpy as np
import pytest

from robodojo.workflows.assets_yam_scene import (
    reshape_long_sleeves_for_yam_scene,
    update_garment_metadata,
)


def test_yam_scene_shirt_derivation_preserves_torso_and_topology():
    points = np.array(
        [
            [0.04, -0.15, 0.0],
            [-0.04, 0.04, 0.01],
            [0.19, 0.14, 0.0],
            [-0.19, 0.14, 0.0],
        ],
        dtype=np.float32,
    )
    result = reshape_long_sleeves_for_yam_scene(points)
    np.testing.assert_array_equal(result[:2], points[:2])
    assert result.shape == points.shape
    assert np.all(np.abs(result[2:, 0]) < np.abs(points[2:, 0]))
    assert np.all(np.abs(result[2:, 1] - 0.109) < np.abs(points[2:, 1] - 0.109))


def test_inherited_garment_metadata_keeps_source_landmarks_and_updates_geometry():
    source = {
        "geometry": {"vertices": 4, "faces": 2},
        "passive": {"functional": {"left_hem": {"id": [3]}}},
    }
    points = np.array(
        [[-0.2, -0.1, 0.0], [0.2, -0.1, 0.0], [-0.2, 0.1, 0.01], [0.2, 0.1, 0.01]]
    )
    metadata = update_garment_metadata(source, points, face_count=3)

    assert metadata["geometry"]["vertices"] == 4
    assert metadata["geometry"]["faces"] == 3
    assert metadata["geometry"]["aligned_bbox"]["extents"] == pytest.approx([0.4, 0.2, 0.01])
    assert metadata["passive"] == source["passive"]
