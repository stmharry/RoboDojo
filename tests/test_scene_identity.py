from omegaconf import OmegaConf
import pytest

from robodojo.core.scene_identity import (
    ARTIFACT_SCHEMA_VERSION,
    SCENE_IDENTITY_FIELDS,
    ArtifactSchemaError,
    require_current_artifact_schema,
    require_current_result_artifact,
    require_matching_scene_identity,
    scene_identity,
)


def _identity():
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "recipe_name": "pi05-bimanual_yam-molmo_yam-general_pickup",
        "contract_hash": "d" * 64,
        "protocol_name": "general_pickup",
        "task_name": "general_pickup",
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
    assert "layout_name" not in identity
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


@pytest.mark.parametrize(
    ("legacy", "message"),
    [
        ({}, "artifact_schema_version mismatch"),
        ({"artifact_schema_version": 1}, "artifact_schema_version mismatch"),
        (
            {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "layout_name": "general_pickup"},
            "removed layout_name selector",
        ),
    ],
)
def test_resume_artifacts_strictly_reject_legacy_schemas_and_layout_selectors(legacy, message):
    with pytest.raises(ArtifactSchemaError, match=message):
        require_current_artifact_schema(legacy, context="resume manifest")

    with pytest.raises(ArtifactSchemaError, match=message):
        require_matching_scene_identity(_identity(), legacy, context="resume manifest")


def test_current_result_artifact_requires_task_keyed_layout_file_and_hash():
    result = {
        **_identity(),
        "eval_time": 1,
        "details": {
            "0": {
                "layout_id": 0,
                "layout_file": "general_pickup_0.json",
                "layout_sha256": "f" * 64,
                "success": False,
                "score": 0.0,
            }
        },
    }

    require_current_result_artifact(result, context="evaluation result")

    for field, invalid, message in (
        ("layout_file", "alternate_0.json", "task-keyed layout identity"),
        ("layout_sha256", "short", "layout_sha256"),
    ):
        legacy = {**result, "details": {"0": {**result["details"]["0"], field: invalid}}}
        with pytest.raises(ArtifactSchemaError, match=message):
            require_current_result_artifact(legacy, context="evaluation result")
