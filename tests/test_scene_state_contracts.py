import numpy as np
import pytest
import torch

from robodojo.sim.environment.scene_manager.appearance import normalize_rgb_color
from robodojo.sim.environment.scene_manager.state import make_root_pose_relative


@pytest.mark.parametrize(
    "origin",
    [
        [1.0, 2.0, 3.0],
        np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
    ],
)
def test_relative_root_pose_normalizes_origin_without_mutating_pose(origin):
    pose = torch.tensor([4.0, 6.0, 8.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    original = pose.clone()

    result = make_root_pose_relative(pose, origin)

    torch.testing.assert_close(result, torch.tensor([3.0, 4.0, 5.0, 1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(pose, original)
    assert result.dtype == pose.dtype
    assert result.device == pose.device


def test_relative_root_pose_preserves_short_origin_compatibility():
    pose = torch.tensor([4.0, 6.0, 8.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    result = make_root_pose_relative(pose, [1.0])
    torch.testing.assert_close(result[:3], torch.tensor([3.0, 6.0, 8.0], dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_relative_root_pose_preserves_cuda_device():
    pose = torch.tensor([4.0, 6.0, 8.0, 1.0, 0.0, 0.0, 0.0], device="cuda:0")
    result = make_root_pose_relative(pose, np.asarray([1.0, 2.0, 3.0]))
    assert result.device == pose.device
    torch.testing.assert_close(result[:3], torch.tensor([3.0, 4.0, 5.0], device="cuda:0"))


def test_rgb_color_contract_normalizes_valid_values_and_rejects_invalid_values():
    color = normalize_rgb_color([0.1, 0.2, 0.3], field="test color")
    assert color.dtype == np.float32
    np.testing.assert_allclose(color, [0.1, 0.2, 0.3])

    for invalid in ([0.1, 0.2], [0.0, 0.0, 1.1], [0.0, float("nan"), 0.0]):
        with pytest.raises(ValueError, match="test color"):
            normalize_rgb_color(invalid, field="test color")
