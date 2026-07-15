import json

from robodojo.sim.environment.seed_manager import seed_manager as seed_manager_module
from robodojo.sim.environment.seed_manager.seed_manager import SeedManager


def _manager():
    return SeedManager(
        {
            "num_envs": 1,
            "task_name": "general_pickup",
            "config_name": "bimanual_yam_molmoact2",
            "layout_config_name": "bimanual_yam_molmoact2",
            "seed": 0,
        }
    )


def test_seed_manager_falls_back_to_bundled_layout(monkeypatch, tmp_path):
    monkeypatch.setattr(seed_manager_module, "ASSETS_PATH", str(tmp_path / "assets"))
    monkeypatch.setattr(seed_manager_module, "ENV_CONFIG_PATH", str(tmp_path / "configs"))
    bundled = tmp_path / "configs/layout/bimanual_yam_molmoact2/0/general_pickup_0.json"
    bundled.parent.mkdir(parents=True)
    bundled.write_text(json.dumps({"source": "bundled"}), encoding="utf-8")

    manager = _manager()
    manager.init_eval()

    assert manager.seed_list == [0]
    assert manager.get_seed_scene_info(0) == {"source": "bundled"}


def test_runtime_layout_takes_precedence_over_bundled(monkeypatch, tmp_path):
    monkeypatch.setattr(seed_manager_module, "ASSETS_PATH", str(tmp_path / "assets"))
    monkeypatch.setattr(seed_manager_module, "ENV_CONFIG_PATH", str(tmp_path / "configs"))
    bundled = tmp_path / "configs/layout/bimanual_yam_molmoact2/0/general_pickup_0.json"
    runtime = tmp_path / "assets/Eval_Layout/RoboDojo/bimanual_yam_molmoact2/0/general_pickup_0.json"
    for path, source in ((bundled, "bundled"), (runtime, "runtime")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"source": source}), encoding="utf-8")

    manager = _manager()
    manager.init_eval()

    assert manager.get_seed_scene_info(0) == {"source": "runtime"}
