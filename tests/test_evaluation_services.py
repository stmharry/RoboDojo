import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robodojo.core.artifacts.results import ARTIFACT_SCHEMA_VERSION
from robodojo.sim.evaluation import restart, resume
from robodojo.sim.evaluation.completion import IncompleteEvaluationError, ensure_evaluation_complete
from robodojo.sim.evaluation.configuration import build_deploy_config, build_evaluation_config
from robodojo.sim.evaluation.services.episodes import EpisodesService
from robodojo.sim.evaluation.services.health import HealthService
from robodojo.sim.evaluation.services.video import VideoService


def _identity(version=ARTIFACT_SCHEMA_VERSION):
    values = {
        "artifact_schema_version": version,
        "recipe": "demo",
        "experiment_hash": "a" * 64,
        "task_protocol": "general_pickup",
        "task": "general_pickup",
        "episode_horizon": 200,
        "evaluation_episodes": 50,
        "environment": "bimanual_yam_molmoact2",
        "embodiment": "bimanual_yam",
        "scene": "molmo_yam",
        "layout_set": "molmo_yam",
        "policy_profile": "manual",
    }
    if version == 3:
        for current, old in {
            "recipe": "recipe_name",
            "experiment_hash": "contract_hash",
            "task_protocol": "protocol_name",
            "task": "task_name",
            "evaluation_episodes": "native_eval_num",
            "environment": "environment_profile",
            "embodiment": "policy_contract",
            "scene": "scene_config",
            "layout_set": "layout_config_name",
        }.items():
            values[old] = values.pop(current)
    return values


def test_evaluation_configuration_uses_domain_vocabulary_and_ws_boundary(tmp_path):
    environment = SimpleNamespace(
        payload={"config_name": "base"},
        name="bimanual_yam_molmoact2",
        identity_hash="e" * 64,
        embodiment="bimanual_yam",
        document=SimpleNamespace(variant=None, asset_builds=[]),
    )
    scene = SimpleNamespace(
        name="molmo_yam",
        identity_hash="s" * 64,
        document=SimpleNamespace(
            component="molmo_yam",
            layout_set="molmo_yam",
            layout_source="bundled",
            asset_builds=[],
            task_asset_builds={},
        ),
    )
    assets = SimpleNamespace(identity_hash="f" * 64, artifacts=())
    config = build_evaluation_config(
        environment=environment,
        scene=scene,
        layout_set=SimpleNamespace(identity_hash="l" * 64),
        environment_assets=assets,
        scene_assets=assets,
        runtime_experiment=None,
        task="general_pickup",
        task_protocol="general_pickup",
        episode_horizon=200,
        evaluation_episodes=50,
        recipe="demo",
        experiment_hash="a" * 64,
        policy_name="Pi_05",
        policy_profile="manual",
        seed=0,
        additional_info="test",
        device_id=0,
        num_envs=1,
        physx_monitor_enabled=False,
        robodojo_revision="r" * 40,
        xpolicylab_revision="x" * 40,
        assets_root=tmp_path,
    )
    assert config["task"] == config["task_protocol"] == "general_pickup"
    assert config["evaluation_episodes"] == 50
    assert config["environment"] == "bimanual_yam_molmoact2"
    assert config["embodiment"] == "bimanual_yam"
    assert config["scene"] == config["layout_set"] == "molmo_yam"
    assert "contract_hash" not in config and "protocol_name" not in config

    deploy = build_deploy_config(
        policy_name="Pi_05",
        host="127.0.0.1",
        port=19000,
        transport="ws",
        server_url="",
        run_id="run",
        task_protocol="general_pickup",
    )
    assert deploy["protocol"] == "ws"
    assert "transport" not in deploy


def test_evaluation_completion_rejects_layout_shortfall():
    ensure_evaluation_complete(eval_time=1, requested=1)
    ensure_evaluation_complete(eval_time=20, requested=20)

    with pytest.raises(IncompleteEvaluationError, match="completed=1, requested=20"):
        ensure_evaluation_complete(eval_time=1, requested=20)


def test_episode_completion_records_final_frames_for_success_and_horizon_failure():
    frames = []

    class Environment(EpisodesService):
        num_envs = 2
        end_flag = [False, False]
        take_action_cnt = [1, 2]
        step_lim = 2
        success = [True, True]
        reward_manager = SimpleNamespace(get_reward=lambda **_kwargs: [1.0, 0.0])

        def get_obs_batch(self, *, env_idx_list, last_frame):
            frames.append((env_idx_list, last_frame))

    environment = Environment()
    assert environment.is_episode_end() is True
    assert environment.end_flag == [True, True]
    assert environment.success == [True, False]
    assert frames == [([0, 1], True)]


def test_resume_reader_normalizes_v3_manifest_without_rewriting(monkeypatch, tmp_path):
    monkeypatch.setattr(resume, "eval_work_root", lambda: tmp_path)
    expected = {**_identity(), "seed": 0, "additional_info": "test"}
    path = Path(resume.resume_manifest_path(expected, "run"))
    path.parent.mkdir(parents=True)
    legacy = {**_identity(3), "seed": 0, "additional_info": "test", "success_nums": 1}
    path.write_text(json.dumps(legacy), encoding="utf-8")

    loaded = resume.load_resume_manifest(expected, "run")

    assert loaded["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert loaded["experiment_hash"] == "a" * 64
    assert loaded["task_protocol"] == "general_pickup"
    assert json.loads(path.read_text(encoding="utf-8"))["artifact_schema_version"] == 3


def test_video_stream_lifecycle_finalizes_camera_artifact(monkeypatch, tmp_path):
    from robodojo.sim.evaluation.services import video

    class Writer:
        def __init__(self, out_path, height, width, channels, fps):
            self.out_path = out_path
            self.height = height
            self.width = width
            self.channels = channels
            self.fps = fps
            self.n_frames = 0
            Path(out_path).write_bytes(b"video")

        def append(self, _frame):
            self.n_frames += 1

        def close(self, *, announce):
            assert announce is False

        def abort(self):
            Path(self.out_path).unlink(missing_ok=True)

    monkeypatch.setattr(video, "VideoStreamWriter", Writer)
    environment = VideoService()
    environment.video_writers = {}
    environment.obs_manager = SimpleNamespace(collect_freq=30)
    environment._stream_dir = str(tmp_path)
    environment._stream_vision(0, {"vision": {"cam_head": {"color": np.zeros((2, 3, 3), dtype=np.uint8)}}})
    environment.save_video(0, str(tmp_path / "episode.mp4"), "success")

    assert (tmp_path / "episode_cam_head_success.mp4").read_bytes() == b"video"
    assert environment.video_writers == {}


def test_physx_health_and_restart_policy_surface_recoverable_and_fatal_states(monkeypatch):
    class Broken(Exception):
        pass

    class Fatal(Exception):
        pass

    monitor = SimpleNamespace(
        is_fatal=lambda: False,
        get_fatal_message=lambda: "kernel failure",
        get_broken_envs=lambda: {1},
    )
    environment = HealthService()
    environment.physx_monitor_enabled = True
    environment._physx_get_monitor = lambda: monitor
    environment._PhysXBrokenError = Broken
    environment._PhysXFatalError = Fatal
    environment.num_envs = 2
    environment.end_flag = [False, False]
    with pytest.raises(Broken, match="1"):
        environment._check_physx_broken_envs()

    monitor.is_fatal = lambda: True
    with pytest.raises(Fatal, match="kernel failure"):
        environment._check_physx_broken_envs()

    persisted = []
    closed = []
    recovery_env = SimpleNamespace(
        persist_resume_manifest=lambda **kwargs: persisted.append(kwargs),
    )
    monkeypatch.delenv("ROBODOJO_FATAL_RESTART_COUNT", raising=False)
    monkeypatch.setattr(restart.os, "execv", lambda *_args: (_ for _ in ()).throw(RuntimeError("exec")))
    with pytest.raises(RuntimeError, match="exec"):
        restart.restart_or_exit(recovery_env, SimpleNamespace(close=lambda: closed.append(True)), "fatal")
    assert persisted == [{"restart_count": 1}]
    assert closed == [True]
    assert restart.os.environ["ROBODOJO_FATAL_RESTART_COUNT"] == "1"

    monkeypatch.setenv("ROBODOJO_FATAL_RESTART_COUNT", str(restart.MAX_INPROC_RESTARTS))
    with pytest.raises(SystemExit) as exc:
        restart.restart_or_exit(recovery_env, SimpleNamespace(close=lambda: None), "fatal")
    assert exc.value.code == 99
