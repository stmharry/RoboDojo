from pathlib import Path

import pytest

from robodojo.core.models import EnvironmentConfigDocument, SceneConfigDocument
from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles import load_environment_profile, load_scene_profile

ROOT = Path(__file__).resolve().parents[1]


def test_environment_profile_rejects_scene_owned_fields():
    with pytest.raises(ValueError, match="scene/task-owned fields"):
        EnvironmentConfigDocument.model_validate(
            {
                "config_name": "invalid",
                "config": {"sim": "sim", "robot": "robot", "camera": "camera"},
                "task_instruction_overrides": {"fold_clothes": []},
            }
        )


def test_scene_profile_rejects_opaque_asset_preparers_with_migration_message():
    with pytest.raises(ValueError, match="task_asset_preparers is no longer supported"):
        SceneConfigDocument.model_validate(
            {
                "config_name": "invalid",
                "component": "default",
                "layout_set": "arx_x5",
                "task_asset_preparers": {"fold_clothes": []},
            }
        )


def test_environment_profiles_resolve_policy_and_diagnostic_metadata():
    paths = RepositoryPaths.resolve(ROOT)
    openarm = load_environment_profile(paths, "openarm_lerobot")
    arx = load_environment_profile(paths, "arx_x5")

    assert openarm.path == ROOT / "configs/environment/openarm_lerobot.yml"
    assert openarm.component_paths["robot"] == ROOT / "configs/robot/dual_openarm_lerobot.yml"
    assert openarm.matched_replay_manifest == ROOT / "configs/reference/openarm_lerobot_wrist_calibration.yml"
    assert arx.matched_replay_manifest is None
    assert set(arx.component_paths) == {"sim", "robot", "camera"}

    scene = load_scene_profile(paths, "molmo_yam")
    assert scene.document.layout_set == "molmo_yam"
    assert scene.document.layout_source == "bundled"
    assert scene.component_path == ROOT / "configs/scene/components/molmo_yam.yml"
    [recipe] = scene.document.task_assets["fold_clothes"]
    assert recipe.kind == "garment_mesh_variant"
    assert recipe.transform == "yam_short_sleeve_v1"
    assert recipe.source.model_dump() == {"object_type": "Garment", "category": "Top_Long", "index": 9}
    assert recipe.destination.model_dump() == {"object_type": "Garment", "category": "Top_Long", "index": 12}

    moonlake = load_scene_profile(paths, "moonlake_office")
    assert moonlake.document.asset_builds == ["moonlake_office"]
    assert moonlake.document.task_asset_builds == {"pack_item_into_container": ["moonlake_packing"]}


def test_runtime_yaml_domains_use_the_canonical_config_root():
    paths = RepositoryPaths.resolve(ROOT)

    assert paths.environment_configs == ROOT / "configs"
    assert paths.environment_profiles == ROOT / "configs/environment"
    assert paths.scene_profiles == ROOT / "configs/scene/profiles"
    assert paths.scene_components == ROOT / "configs/scene/components"
    assert paths.task_configs == ROOT / "configs/task"
    assert (paths.task_configs / "_task.yml").is_file()
    assert not (ROOT / "configs/arx_x5.yml").exists()
    assert not (ROOT / "task/RoboDojo/config").exists()


def test_scene_profile_identity_changes_with_profile_or_component_content(tmp_path):
    profile_path = tmp_path / "configs/scene/profiles/test.yml"
    component_path = tmp_path / "configs/scene/components/test.yml"
    profile_path.parent.mkdir(parents=True)
    component_path.parent.mkdir(parents=True)
    profile_path.write_text(
        "config_name: test\ncomponent: test\nlayout_set: arx_x5\nlayout_source: assets\n",
        encoding="utf-8",
    )
    component_path.write_text("Room: {}\n", encoding="utf-8")
    paths = RepositoryPaths(root=tmp_path)

    original = load_scene_profile(paths, "test").identity_hash
    component_path.write_text("Room:\n  visual_color: [0.1, 0.2, 0.3]\n", encoding="utf-8")
    component_changed = load_scene_profile(paths, "test").identity_hash
    profile_path.write_text(
        "config_name: test\ncomponent: test\nlayout_set: other\nlayout_source: assets\n",
        encoding="utf-8",
    )
    profile_changed = load_scene_profile(paths, "test").identity_hash

    assert len({original, component_changed, profile_changed}) == 3


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"layout_source": "fallback"}, "layout_source"),
        (
            {
                "task_assets": {
                    "fold_clothes": [
                        {
                            "kind": "garment_mesh_variant",
                            "transform": "unknown_v1",
                            "source": {"object_type": "Garment", "category": "Top_Long", "index": 9},
                            "destination": {"object_type": "Garment", "category": "Top_Long", "index": 12},
                        }
                    ]
                }
            },
            "transform",
        ),
        (
            {
                "task_assets": {
                    "fold_clothes": [
                        {
                            "kind": "garment_mesh_variant",
                            "transform": "yam_short_sleeve_v1",
                            "source": {"object_type": "Garment", "category": "../Top_Long", "index": 9},
                            "destination": {"object_type": "Garment", "category": "Top_Long", "index": 12},
                        }
                    ]
                }
            },
            "safe catalog path segment",
        ),
        (
            {
                "task_assets": {
                    "fold_clothes": [
                        {
                            "kind": "garment_mesh_variant",
                            "transform": "yam_short_sleeve_v1",
                            "source": {"object_type": "Garment", "category": "Top_Long", "index": 9},
                            "destination": {"object_type": "Garment", "category": "..\\Top_Long", "index": 12},
                        }
                    ]
                }
            },
            "safe catalog path segment",
        ),
    ],
)
def test_scene_profile_rejects_invalid_typed_asset_contracts(override, message):
    payload = {"config_name": "invalid", "component": "default", "layout_set": "arx_x5", **override}
    with pytest.raises(ValueError, match=message):
        SceneConfigDocument.model_validate(payload)


@pytest.mark.parametrize("name", ["openarm_wowrobo_v1_1", "openarm_anvil_v2"])
def test_pending_hardware_profiles_can_be_inspected_but_not_launched(name):
    paths = RepositoryPaths.resolve(ROOT)
    profile = load_environment_profile(paths, name, validate_calibration=False)

    assert profile.document.hardware_calibration == name
    with pytest.raises(ValueError, match="not release-ready"):
        load_environment_profile(paths, name)
