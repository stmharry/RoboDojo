import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from robodojo.core.experiments.catalogs import load_protocol_catalog
from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles.scene import load_scene_profile
from robodojo.core.workspace import task_placement_rules
from robodojo.sim.environment.reward_manager.reward_manager import RewardManager
from robodojo.workflows.asset_builders.moonlake_packing.publication import _publish_paths, build
from robodojo.workflows.assets import generated_fixture_error, required_fixture_builds

ROOT = Path(__file__).resolve().parents[1]
PATHS = RepositoryPaths.resolve(ROOT)
LAYOUT_ROOT = ROOT / "configs/layout/moonlake_office/0"
VARIANTS = (
    ("moonlake_measuring_spoon", "Rigid"),
    ("moonlake_phone15_dummy", "Rigid"),
    ("moonlake_fake_apple", "Rigid"),
    ("moonlake_phillips_screwdriver", "Rigid"),
    ("moonlake_anker_cable", "Articulation"),
    ("moonlake_abc_block", "Rigid"),
)
RUN_ISAAC_USD_TESTS = os.environ.get("ROBODOJO_RUN_ISAAC_USD_TESTS") == "1"


@pytest.fixture(scope="module")
def isaac_app():
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")
    from isaacsim import SimulationApp

    return SimulationApp({"headless": True})


def test_moonlake_packing_profile_declares_generated_scene_asset_build():
    scene = load_scene_profile(PATHS, "moonlake_office")
    assert scene.document.asset_builds == ["moonlake_office"]
    assert scene.document.task_asset_builds == {
        "pack_item_into_container": ["moonlake_packing"],
    }
    assert "pack_item_into_container" not in scene.document.task_assets
    assert required_fixture_builds(scene, "general_pickup") == ("moonlake_office",)
    assert required_fixture_builds(scene, "pack_item_into_container") == (
        "moonlake_office",
        "moonlake_packing",
    )


def test_container_behavior_remains_owned_by_pack_item_into_container():
    task_config = yaml.safe_load((ROOT / "configs/task/pack_item_into_container.yml").read_text(encoding="utf-8"))
    protocol = load_protocol_catalog(PATHS).protocols["pack_item_into_container"]
    reward_manager = RewardManager(1)

    assert set(task_placement_rules(task_config)) == {"item", "container"}
    assert protocol.task == "pack_item_into_container"
    assert protocol.episode_horizon == 1300
    assert protocol.evaluation_episodes == 6
    assert reward_manager.is_object_in_functional_volume("item", "container", "packing_cavity") == (
        "is_object_in_functional_volume",
        {"label_A": "item", "label_B": "container", "B_volume_tag": "packing_cavity", "margin": 0.0},
    )


def test_asset_publication_restores_previous_directory_when_replace_fails(tmp_path, monkeypatch):
    source = tmp_path / "staged"
    destination = tmp_path / "published"
    source.mkdir()
    destination.mkdir()
    (source / "state.txt").write_text("new", encoding="utf-8")
    (destination / "state.txt").write_text("old", encoding="utf-8")
    real_replace = os.replace
    failed = False

    def fail_new_publication(src, dst):
        nonlocal failed
        if Path(src) == source and Path(dst) == destination and not failed:
            failed = True
            raise OSError("injected publication failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_new_publication)
    with pytest.raises(OSError, match="injected"):
        _publish_paths([(source, destination)])

    assert (destination / "state.txt").read_text(encoding="utf-8") == "old"
    assert (source / "state.txt").read_text(encoding="utf-8") == "new"


def test_asset_publication_rolls_back_directories_when_manifest_replace_fails(tmp_path, monkeypatch):
    staged_asset = tmp_path / "staged_asset"
    published_asset = tmp_path / "published_asset"
    staged_manifest = tmp_path / "staged_manifest.json"
    published_manifest = tmp_path / "published_manifest.json"
    staged_asset.mkdir()
    published_asset.mkdir()
    (staged_asset / "state.txt").write_text("new", encoding="utf-8")
    (published_asset / "state.txt").write_text("old", encoding="utf-8")
    staged_manifest.write_text('{"generation": "new"}\n', encoding="utf-8")
    published_manifest.write_text('{"generation": "old"}\n', encoding="utf-8")
    real_replace = os.replace

    def fail_manifest_publication(src, dst):
        if Path(src) == staged_manifest and Path(dst) == published_manifest:
            raise OSError("injected manifest publication failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_manifest_publication)
    with pytest.raises(OSError, match="manifest publication"):
        _publish_paths([(staged_asset, published_asset), (staged_manifest, published_manifest)])

    assert (published_asset / "state.txt").read_text(encoding="utf-8") == "old"
    assert published_manifest.read_text(encoding="utf-8") == '{"generation": "old"}\n'


def test_six_packing_layouts_are_deterministic_labelled_and_reachable():
    for layout_id, (item_category, item_type) in enumerate(VARIANTS):
        path = LAYOUT_ROOT / f"pack_item_into_container_{layout_id}.json"
        layout = json.loads(path.read_text(encoding="utf-8"))
        assert layout["Table"]["scale"] == [1.2, 0.7, 0.05]
        assert layout["Table"]["default_pos"] == [0, 0, 0.725]
        assert list(layout["Geometry"]) == ["moonlake_office_fixture"]
        assert len(layout["Geometry"]["moonlake_office_fixture"]) == 1

        [container] = layout["Articulation"]["moonlake_magnetic_gift_box"]
        assert container["label"] == "container"
        assert container["default_pos"] == [-0.12, 0.08, 0.75]
        [item] = layout[item_type][item_category]
        assert item["label"] == "item"
        assert item["category_idx"] == 0
        assert item["default_pos"][2] == 0.75
        assert -0.6 < item["default_pos"][0] < 0.6
        assert -0.35 < item["default_pos"][1] < 0.35
        assert item["default_ori"] == [1, 0, 0, 0]
        assert item["xlim"] == [item["default_pos"][0], item["default_pos"][0]]
        assert item["ylim"] == [item["default_pos"][1], item["default_pos"][1]]
        assert item["rotate_rand"] is False


@pytest.mark.skipif(not RUN_ISAAC_USD_TESTS, reason="run in an isolated Isaac USD test process")
def test_builder_authors_valid_internal_assets_and_provenance(tmp_path, isaac_app, monkeypatch):
    result = build(tmp_path, PATHS.moonlake_packing_manifest)
    manifest = yaml.safe_load(PATHS.moonlake_packing_manifest.read_text(encoding="utf-8"))

    assert result["distribution"]["classification"] == "internal_reference_only"
    assert set(result["assets"]) == set(manifest["assets"])
    for key, record in result["assets"].items():
        root = tmp_path / record["asset"]
        assert {path.name for path in root.iterdir()} == {
            "object.usd",
            "metadata.json",
            "description.json",
            "provenance.json",
        }
        provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
        assert provenance["distribution"]["publish"] == "prohibited"
        assert provenance["reference"]["media_embedded"] is False
        assert record["validation"]["cameras"] == 0
        assert record["validation"]["collisions"] > 0

    container = result["assets"]["container"]
    assert container["validation"] == {
        "articulations": 1,
        "rigid_bodies": 2,
        "joints": 1,
        "collisions": 7,
        "cameras": 0,
    }
    cavity = container["metadata"]["passive"]["volumes"]["packing_cavity"]
    assert cavity["base_link"] == "base"
    assert cavity["minimum"] == pytest.approx([-0.1494, -0.0732, 0.003])
    assert cavity["maximum"] == pytest.approx([0.1494, 0.0732, 0.0986])

    cable = result["assets"]["cable"]
    assert cable["validation"]["rigid_bodies"] == 32
    assert cable["validation"]["joints"] == 31
    assert cable["metadata"]["physics"]["joint_model"] == "alternating_revolute"
    assert cable["metadata"]["physics"]["self_collision"] is False
    assert len(cable["metadata"]["geometry"]["link_bboxes"]) == 32

    monkeypatch.setattr("robodojo.workflows.assets.assets_root", lambda: tmp_path)
    assert generated_fixture_error(PATHS, "moonlake_packing") is None
    cable_object = tmp_path / cable["asset"] / "object.usd"
    cable_object.write_bytes(cable_object.read_bytes() + b"tampered")
    assert "checksum mismatch" in generated_fixture_error(PATHS, "moonlake_packing")


def test_articulation_link_pose_uses_live_physics_transform_and_scalar_first_quaternion():
    import torch

    from robodojo.sim.utils.physics_pose import articulation_link_pose_wxyz

    physics_view = SimpleNamespace(
        get_link_transforms=lambda: torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9]]]
        )
    )
    articulation_view = SimpleNamespace(
        is_physics_handle_valid=lambda: True,
        get_body_index=lambda name: {"target_link": 1}[name],
        _physics_view=physics_view,
    )
    pose = articulation_link_pose_wxyz(articulation_view, "target_link", "/World/test")

    assert pose.tolist() == pytest.approx([1.0, 2.0, 3.0, 0.9, 0.1, 0.2, 0.3])
