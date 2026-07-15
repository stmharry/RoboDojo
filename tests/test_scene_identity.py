import pytest

from robodojo.core.scene_identity import (
    SCENE_IDENTITY_FIELDS,
    require_matching_scene_identity,
    scene_identity,
)


def _identity():
    return {
        "scene_config": "molmo_yam",
        "scene_component": "molmo_yam",
        "scene_profile_hash": "a" * 64,
        "layout_config_name": "molmo_yam",
        "layout_source": "bundled",
        "layout_set_hash": "b" * 64,
        "scene_asset_hash": "c" * 64,
    }


def test_scene_identity_contains_every_result_and_resume_boundary_field():
    identity = _identity()
    assert scene_identity({**identity, "unrelated": True}) == identity
    assert tuple(identity) == SCENE_IDENTITY_FIELDS
    require_matching_scene_identity(identity, identity, context="resume manifest")


@pytest.mark.parametrize("field", ["scene_profile_hash", "layout_set_hash", "scene_asset_hash"])
def test_scene_identity_rejects_runtime_drift(field):
    expected = _identity()
    actual = {**expected, field: "d" * 64}
    with pytest.raises(ValueError, match=rf"resume manifest {field} mismatch"):
        require_matching_scene_identity(expected, actual, context="resume manifest")
