from omegaconf import OmegaConf
import pytest

from robodojo.core.scene_identity import (
    SCENE_IDENTITY_FIELDS,
    require_matching_scene_identity,
    scene_identity,
)


def _identity():
    return {
        "recipe_name": "pi05-bimanual_yam-molmo_yam-general_pickup",
        "contract_hash": "d" * 64,
        "protocol_name": "general_pickup",
        "task_name": "general_pickup",
        "layout_name": "general_pickup",
        "episode_horizon": 200,
        "native_eval_num": 50,
        "environment_profile_hash": "e" * 64,
        "policy_contract": "bimanual_yam",
        "scene_config": "molmo_yam",
        "scene_component": "molmo_yam",
        "scene_profile_hash": "a" * 64,
        "layout_config_name": "molmo_yam",
        "layout_source": "bundled",
        "layout_set_hash": "b" * 64,
        "scene_asset_hash": "c" * 64,
        "scene_asset_builds": [],
        "scene_asset_identities": [],
    }


def test_scene_identity_contains_every_result_and_resume_boundary_field():
    identity = _identity()
    assert scene_identity({**identity, "unrelated": True}) == identity
    assert tuple(identity) == SCENE_IDENTITY_FIELDS
    require_matching_scene_identity(identity, identity, context="resume manifest")


def test_scene_identity_detaches_omegaconf_asset_collections_for_json_artifacts():
    identity = _identity()
    identity["scene_asset_builds"] = OmegaConf.create(["moonlake_office"])
    identity["scene_asset_identities"] = OmegaConf.create(
        [{"destination": "Object/RoboDojo/Geometry/moonlake_office_fixture", "manifest_hash": "a" * 64}]
    )

    resolved = scene_identity(identity)

    assert type(resolved["scene_asset_builds"]) is list
    assert type(resolved["scene_asset_identities"]) is list
    assert type(resolved["scene_asset_identities"][0]) is dict


@pytest.mark.parametrize("field", ["scene_profile_hash", "layout_set_hash", "scene_asset_hash"])
def test_scene_identity_rejects_runtime_drift(field):
    expected = _identity()
    actual = {**expected, field: "d" * 64}
    with pytest.raises(ValueError, match=rf"resume manifest {field} mismatch"):
        require_matching_scene_identity(expected, actual, context="resume manifest")
