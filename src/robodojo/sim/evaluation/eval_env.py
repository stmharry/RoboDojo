from copy import deepcopy
from datetime import datetime
import inspect
import logging
import os

from client_server.ws.model_client import WsModelClient
import websockets

from robodojo.core.artifacts.results import require_matching_scene_identity, scene_identity
from robodojo.core.storage import eval_work_root
from robodojo.sim import task_discovery
from robodojo.sim.environment.observation_manager.obs_manager import ObsManager
from robodojo.sim.environment.seed_manager.seed_manager import SeedManager
from robodojo.sim.evaluation.services.actions import ActionsService
from robodojo.sim.evaluation.services.episodes import EpisodesService
from robodojo.sim.evaluation.services.health import HealthService
from robodojo.sim.evaluation.services.persistence import PersistenceService
from robodojo.sim.evaluation.services.video import VideoService
from robodojo.sim.utils.cluttered_generator import UnStableError
from robodojo.sim.utils.pipeline_utils import get_robot_action_dim_info
from robodojo.sim.utils.save_file import VideoStreamWriter

logger = logging.getLogger(__name__)


def _patch_websockets_proxy_compat():
    # Isaac Sim may load an older bundled websockets package that does not
    # accept the proxy kwarg used by XPolicyLab's websocket client.
    connect = websockets.connect
    if getattr(connect, "_robodojo_proxy_compat", False):
        return

    try:
        if "proxy" in inspect.signature(connect).parameters:
            return
    except (TypeError, ValueError):
        pass

    def connect_without_proxy(*args, proxy=None, **kwargs):
        _ = proxy
        return connect(*args, **kwargs)

    connect_without_proxy._robodojo_proxy_compat = True
    websockets.connect = connect_without_proxy


def create_eval_env(config, app, resume_state=None, **kwargs):
    task_name = config.eval_cfg.get("task", None)
    if task_name is None:
        raise ValueError("Task name must be specified in eval_cfg!")

    task_name, task_class = task_discovery.load_task_class(task_name)
    config.eval_cfg["task"] = task_name

    class EvalEnv(ActionsService, EpisodesService, PersistenceService, VideoService, HealthService, task_class):
        def __init__(self, config, app, resume_state=None, **kwargs):
            self.policy_enabled = bool(kwargs.pop("policy_enabled", True))
            self.record_video_enabled = bool(kwargs.pop("record_video_enabled", True))
            super().__init__(config, app, **kwargs)
            self.eval_cfg = config.eval_cfg
            self.environment = self.eval_cfg["environment"]
            self.environment_profile_hash = self.eval_cfg.get("environment_profile_hash")
            self.embodiment = self.eval_cfg.get("embodiment")
            self.scene = self.eval_cfg.get("scene")
            self.scene_component = self.eval_cfg.get("scene_component")
            self.scene_profile_hash = self.eval_cfg.get("scene_profile_hash")
            self.layout_set = self.eval_cfg.get("layout_set")
            self.layout_source = self.eval_cfg.get("layout_source")
            self.layout_set_hash = self.eval_cfg.get("layout_set_hash")
            self.scene_asset_hash = self.eval_cfg.get("scene_asset_hash")
            self.task_name = self.eval_cfg.get("task", None)
            self.task_protocol = self.eval_cfg.get("task_protocol", self.task_name)
            self.recipe = self.eval_cfg.get("recipe")
            self.experiment_hash = self.eval_cfg.get("experiment_hash")
            self.step_lim = int(self.eval_cfg["episode_horizon"])
            self.eval_batch = self.eval_cfg.get("eval_batch", False)
            self.eval_num = int(self.eval_cfg.get("eval_num", 50))
            self.policy_name = self.eval_cfg.get("policy_name", None)
            self.policy_profile = self.eval_cfg.get("policy_profile", self.policy_name)
            self.additional_info = self.eval_cfg.get("additional_info", "")
            self.eval_seed = self.eval_cfg.get("seed", 0)
            self.physx_monitor_enabled = bool(self.eval_cfg.get("physx_monitor_enabled", False))
            if self.physx_monitor_enabled:
                from robodojo.sim.evaluation.physx_warning_monitor import (
                    PhysXBrokenError,
                    PhysXFatalError,
                    get_monitor,
                )

                self._physx_get_monitor = get_monitor
                self._PhysXBrokenError = PhysXBrokenError
                self._PhysXFatalError = PhysXFatalError

            run_id = os.environ.get("ROBODOJO_RUN_ID")
            if not run_id:
                run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                os.environ["ROBODOJO_RUN_ID"] = run_id
            self.run_id = run_id
            self.save_dir = os.path.join(
                str(eval_work_root()),
                self.task_protocol,
                self.policy_profile,
                self.environment,
                str(self.eval_seed) + "_" + self.additional_info,
                run_id,
            )

            if resume_state is not None:
                resumed_save_dir = resume_state.get("save_dir")
                if resumed_save_dir:
                    self.save_dir = resumed_save_dir

            # Temp dir for in-progress streaming videos. Orphans left behind by
            # a hard kill (SIGKILL/crash) can't be cleaned via the in-memory
            # writer dict, so sweep them from disk on startup: any *.tmp.mp4
            # here belongs to an unfinished episode that will be re-run.
            self._stream_dir = os.path.join(self.save_dir, "_stream")
            self._sweep_stream_dir()

            self.obs_config = deepcopy(self.eval_cfg.get("observation", {}))
            self.obs_config["save_dir"] = self.save_dir
            self.description_cfg = self.eval_cfg.get("description", dict())
            self.obs_manager = ObsManager(
                obs_config=self.obs_config,
                num_envs=self.num_envs,
                dt=self.dt,
                task_name=self.task_name,
                description_cfg=self.description_cfg,
                seeds_per_env=self.env_seed_list,
            )

            self.success = [True] * self.num_envs
            self.end_flag = [False] * self.num_envs
            self.take_action_cnt = [0] * self.num_envs
            # Per-env streaming video writers: {env_idx: {camera_key: writer}}.
            # Replaces the old full-episode frame cache; only vision frames are
            # streamed to disk as they arrive instead of buffered in RAM.
            self.video_writers: dict[int, dict[str, VideoStreamWriter]] = {}
            self.episode_nums = self.num_envs
            self.unstable_nums = 0
            self.unstable_envs: set[int] = set()
            self.success_nums = 0
            self.fail_nums = 0
            self.total_score = 0

            self.abandoned_seeds: set[int] = set()
            self.current_env_seed_map: dict[int, int] = {}
            self.eval_result = {
                "success_rate": 0.0,
                "eval_time": 0,
                "score": 0.0,
                **scene_identity(self.eval_cfg),
                "details": {},
            }

            self.scene_manager.layout_manager.replay = True
            self.seed_manager = SeedManager(config.eval_cfg)

            completed_layout_ids: list[int] = []
            abandoned_layout_ids: list[int] = []
            if resume_state is not None:
                require_matching_scene_identity(self.eval_cfg, resume_state, context="resume manifest")
                self.success_nums = int(resume_state.get("success_nums", 0))
                self.fail_nums = int(resume_state.get("fail_nums", 0))
                self.total_score = float(resume_state.get("total_score", 0.0))

                resumed_details = resume_state.get("details") or {}

                normalised_details = {}
                for k, v in resumed_details.items():
                    try:
                        normalised_details[int(k)] = v
                    except (TypeError, ValueError):
                        normalised_details[k] = v
                self.eval_result["details"] = normalised_details
                self.abandoned_seeds = set(int(s) for s in resume_state.get("abandoned_layout_ids", []))
                completed_layout_ids = [
                    int(v["layout_id"]) for v in normalised_details.values() if isinstance(v, dict) and "layout_id" in v
                ]
                abandoned_layout_ids = list(self.abandoned_seeds)
                eval_time = self.success_nums + self.fail_nums
                if eval_time > 0:
                    self.eval_result["success_rate"] = self.success_nums / eval_time
                    self.eval_result["score"] = self.total_score / eval_time * 100
                self.eval_result["eval_time"] = eval_time
                logger.info(
                    "[EvalEnv][resume] save_dir=%s success=%s fail=%s completed=%s abandoned=%s",
                    self.save_dir,
                    self.success_nums,
                    self.fail_nums,
                    len(completed_layout_ids),
                    len(abandoned_layout_ids),
                )
            self.seed_manager.init_eval(
                completed_layout_ids=completed_layout_ids,
                abandoned_layout_ids=abandoned_layout_ids,
            )

            self.deploy_cfg = config.deploy_cfg
            self.port = self.deploy_cfg.get("port", None)
            if self.policy_enabled and self.port is None:
                raise ValueError("Port must be specified in deploy_cfg for the policy server!")
            self.host = self.deploy_cfg.get("host", "localhost")
            self.model_client = None
            if self.policy_enabled:
                _patch_websockets_proxy_compat()
                policy_server_url = self.deploy_cfg.get("policy_server_url") or f"ws://{self.host}:{self.port}"
                evaluation_id = self.deploy_cfg.get("evaluation_id", self.run_id)
                trial_id = self.deploy_cfg.get("trial_id", f"{self.task_name}-{self.run_id}")
                action_case_id = self.deploy_cfg.get("action_case_id", f"{self.task_name}_case")
                self.model_client = WsModelClient(
                    url=policy_server_url,
                    evaluation_id=evaluation_id,
                    trial_id=trial_id,
                    action_case_id=action_case_id,
                    repeat_index=self.deploy_cfg.get("repeat_index"),
                )
            self.robot_action_dim_info = get_robot_action_dim_info(env_cfg=self.eval_cfg)

        def close(self):
            self._abort_video_writers()
            self.obs_manager.reset()
            super().close()

        def _post_setup_scene(self, sim):
            super()._post_setup_scene(sim)
            self.obs_manager.initialize(self)

        def reset(self, seed=None, options=None):
            seed = list(seed)
            if len(seed) < self.num_envs:
                seed = seed + [None] * (self.num_envs - len(seed))

            real_indices = [i for i, s in enumerate(seed) if s is not None]
            safe_seed = seed[real_indices[0]] if real_indices else 0
            # Fill None positions with safe_seed so scene_manager can still load
            self.env_seeds = [s if s is not None else safe_seed for s in seed]

            self.success = [True] * self.num_envs
            self.end_flag = [False] * self.num_envs
            self.take_action_cnt = [0] * self.num_envs
            # Discard any writers left open by a previous (e.g. crashed or
            # unstable) batch before starting a fresh one.
            self._abort_video_writers()
            self.episode_nums = len(real_indices)
            self.unstable_envs = set()

            self.current_env_seed_map = {}
            for idx in range(self.num_envs):
                self.scene_manager.layout_manager.set_saved_layout(
                    idx, self.seed_manager.get_seed_scene_info(self.env_seeds[idx])
                )
                if seed[idx] is None:
                    self.success[idx] = False
                    self.end_flag[idx] = True
                else:
                    self.current_env_seed_map[idx] = seed[idx]

            super().reset(seed=self.env_seeds, options=options)
            self.obs_manager.reset()  # Reset observation manager for the next episode
            self.setup_scene()
            self.robot_manager.set_origin_endpose()
            self.robot_manager.set_robot_init_state()
            self.reward_manager.init_state()

            if self.model_client is not None:
                self.model_client.call(func_name="reset")

        def setup_scene(self):
            self.scene_manager.apply_saved_poses(env_idx_list=list(range(self.num_envs)))
            self._align_layout_success()
            success, unstable_envs = self.scene_manager.layout_manager.check_layout_stability(self)
            unstable_envs = [idx for idx in set(unstable_envs) if idx < self.num_envs]
            self.unstable_nums += len(unstable_envs)
            self.episode_nums -= len(unstable_envs)
            if not success or self.episode_nums <= 0:
                raise UnStableError("All scene Unstable Error!")
            for _ in range(10):
                self.render()
            for idx in range(200):
                self.sim_step()
                if idx % 5 == 0:
                    self.render()
                    self.obs_manager.get_obs()
            if self.physx_monitor_enabled:
                self._check_physx_broken_envs()

        def get_obs(self):
            return self.get_obs_batch(env_idx_list=[0])[0]

        def get_obs_batch(self, env_idx_list=None, last_frame=False):
            if self.physx_monitor_enabled:
                self._check_physx_broken_envs()
            self.render()
            if env_idx_list is None:
                env_idx_list = list(range(self.num_envs))
            if self.physx_monitor_enabled:
                self._check_endpose_finite(env_idx_list)
            data = self.obs_manager.get_obs(env_idx_list=env_idx_list)
            data_list = []
            for env_idx in env_idx_list:
                if self.record_video_enabled and (not self.end_flag[env_idx] or last_frame):
                    self._stream_vision(env_idx, data[env_idx])
                env_data = deepcopy(data[env_idx])
                env_data["env_idx"] = env_idx
                data_list.append(env_data)
            return data_list

        def step(self, env_idx_list, decimation=1):
            meta_control_list = self.robot_manager.control_manager.pop(env_idx_list)
            for _ in range(decimation):
                super().step(meta_control_list=meta_control_list)
                self.sim_step(render=False)

        def _align_layout_success(self):
            for env_idx in range(self.num_envs):
                if self.end_flag[env_idx]:
                    continue
                if not self.scene_manager.layout_manager.layout_valid[env_idx]:
                    self.success[env_idx] = False
                    self.end_flag[env_idx] = True
                    self.episode_nums -= 1

    return EvalEnv(config, app, resume_state=resume_state, **kwargs)
