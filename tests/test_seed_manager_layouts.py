import json

import pytest

from robodojo.core.layouts import resolve_layout_set
from robodojo.sim.environment.seed_manager import seed_manager as seed_manager_module
from robodojo.sim.environment.seed_manager.seed_manager import SeedManager


def _manager(layout_source="bundled", expected_hash=None):
    return SeedManager(
        {
            "num_envs": 1,
            "task_name": "general_pickup",
            "layout_name": "general_pickup",
            "config_name": "bimanual_yam",
            "layout_config_name": "molmo_yam",
            "layout_source": layout_source,
            "layout_set_hash": expected_hash,
            "seed": 0,
        }
    )


def test_seed_manager_uses_explicit_bundled_layout(monkeypatch, tmp_path):
    monkeypatch.setattr(seed_manager_module, "ASSETS_PATH", str(tmp_path / "assets"))
    monkeypatch.setattr(seed_manager_module, "ENV_CONFIG_PATH", str(tmp_path / "configs"))
    bundled = tmp_path / "configs/layout/molmo_yam/0/general_pickup_0.json"
    bundled.parent.mkdir(parents=True)
    bundled.write_text(json.dumps({"source": "bundled"}), encoding="utf-8")

    manager = _manager()
    manager.init_eval()

    assert manager.seed_list == [0]
    assert manager.get_seed_scene_info(0) == {"source": "bundled"}


def test_bundled_layout_ignores_runtime_layout(monkeypatch, tmp_path):
    monkeypatch.setattr(seed_manager_module, "ASSETS_PATH", str(tmp_path / "assets"))
    monkeypatch.setattr(seed_manager_module, "ENV_CONFIG_PATH", str(tmp_path / "configs"))
    bundled = tmp_path / "configs/layout/molmo_yam/0/general_pickup_0.json"
    runtime = tmp_path / "assets/Eval_Layout/RoboDojo/molmo_yam/0/general_pickup_0.json"
    for path, source in ((bundled, "bundled"), (runtime, "runtime")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"source": source}), encoding="utf-8")

    manager = _manager()
    manager.init_eval()

    assert manager.get_seed_scene_info(0) == {"source": "bundled"}


def test_asset_layout_ignores_bundled_layout(monkeypatch, tmp_path):
    monkeypatch.setattr(seed_manager_module, "ASSETS_PATH", str(tmp_path / "assets"))
    monkeypatch.setattr(seed_manager_module, "ENV_CONFIG_PATH", str(tmp_path / "configs"))
    bundled = tmp_path / "configs/layout/molmo_yam/0/general_pickup_0.json"
    runtime = tmp_path / "assets/Eval_Layout/RoboDojo/molmo_yam/0/general_pickup_0.json"
    for path, source in ((bundled, "bundled"), (runtime, "runtime")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"source": source}), encoding="utf-8")

    manager = _manager("assets")
    manager.init_eval()

    assert manager.get_seed_scene_info(0) == {"source": "runtime"}


def test_layout_resolver_sorts_numeric_ids_and_hashes_content(tmp_path):
    directory = tmp_path / "configs/layout/molmo_yam/0"
    directory.mkdir(parents=True)
    (directory / "general_pickup_10.json").write_text('{"value": 10}', encoding="utf-8")
    first = directory / "general_pickup_2.json"
    first.write_text('{"value": 2}', encoding="utf-8")

    resolved = resolve_layout_set(
        config_root=tmp_path / "configs",
        assets_root=tmp_path / "assets",
        benchmark="RoboDojo",
        layout_set="molmo_yam",
        layout_source="bundled",
        task="general_pickup",
        seed=0,
    )
    assert [layout.layout_id for layout in resolved.layouts] == [2, 10]
    original_hash = resolved.identity_hash

    first.write_text('{"value": "changed"}', encoding="utf-8")
    changed = resolve_layout_set(
        config_root=tmp_path / "configs",
        assets_root=tmp_path / "assets",
        benchmark="RoboDojo",
        layout_set="molmo_yam",
        layout_source="bundled",
        task="general_pickup",
        seed=0,
    )
    assert changed.identity_hash != original_hash


def test_layout_resolver_rejects_missing_and_duplicate_layouts(tmp_path):
    kwargs = {
        "config_root": tmp_path / "configs",
        "assets_root": tmp_path / "assets",
        "benchmark": "RoboDojo",
        "layout_set": "molmo_yam",
        "layout_source": "bundled",
        "task": "general_pickup",
        "seed": 0,
    }
    with pytest.raises(ValueError, match="no bundled layouts"):
        resolve_layout_set(**kwargs)

    directory = tmp_path / "configs/layout/molmo_yam/0"
    directory.mkdir(parents=True)
    (directory / "general_pickup_1.json").write_text("{}", encoding="utf-8")
    (directory / "general_pickup_01.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate layout ids"):
        resolve_layout_set(**kwargs)


def test_seed_manager_rejects_precomputed_layout_hash_drift(monkeypatch, tmp_path):
    monkeypatch.setattr(seed_manager_module, "ASSETS_PATH", str(tmp_path / "assets"))
    monkeypatch.setattr(seed_manager_module, "ENV_CONFIG_PATH", str(tmp_path / "configs"))
    layout = tmp_path / "configs/layout/molmo_yam/0/general_pickup_0.json"
    layout.parent.mkdir(parents=True)
    layout.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="resolved layout hash mismatch"):
        _manager(expected_hash="0" * 64).init_eval()
