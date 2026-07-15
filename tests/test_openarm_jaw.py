import numpy as np
import pytest

from robodojo.workflows.openarm_jaw import JAW_CAD_TO_FINGER, register_enlarged_jaw, validate_jaw_mode


def box_vertices(bounds):
    lower, upper = np.asarray(bounds, dtype=np.float64)
    return np.asarray(
        [[x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])]
    )


def test_enlarged_jaw_registration_aligns_stock_base_without_reflection():
    stock = box_vertices(([-0.030499, -0.00576, -0.004401], [0.030501, 0.027506, 0.080501]))
    enlarged_cad = box_vertices(([-0.0178, -0.04359997, -0.05049474], [0.0813438, 0.01740003, -0.01731143]))

    result = register_enlarged_jaw(enlarged_cad, stock)

    assert np.linalg.det(JAW_CAD_TO_FINGER) == pytest.approx(1.0)
    assert result.transform[:3, 3] == pytest.approx([0.01310097, 0.044776085, 0.013399], abs=1e-8)
    assert result.base_plane_error_m <= 0.0005
    assert result.cross_section_center_error_m <= 0.0005
    assert result.width_error_m <= 0.001
    assert result.thickness_error_m <= 0.001
    assert result.enlarged_bounds[1, 2] > result.stock_bounds[1, 2]


@pytest.mark.parametrize("mode", ["stock", "enlarged"])
def test_jaw_mode_accepts_only_explicit_options(mode):
    assert validate_jaw_mode(mode) == mode


@pytest.mark.parametrize("mode", [None, True, "extended", "ENLARGED"])
def test_jaw_mode_rejects_ambiguous_options(mode):
    with pytest.raises(ValueError, match=r"stock\|enlarged"):
        validate_jaw_mode(mode)
