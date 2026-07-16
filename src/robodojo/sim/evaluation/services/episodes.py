from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import logging
import os
from pathlib import Path

from robodojo.sim.utils.save_file import save_json

logger = logging.getLogger(__name__)


class EpisodesService:
    def eval_one_episode(self):
        policy_name = self.deploy_cfg["policy_name"]
        try:
            eval_module = __import__(
                f"XPolicyLab.policy.{policy_name}.deploy",
                fromlist=["eval_one_episode"],
            )
        except ImportError as e:
            logger.error(
                "[TestEnv] failed to import policy module XPolicyLab.policy.%s.deploy: %s",
                policy_name,
                e,
            )
            raise e

        if not hasattr(eval_module, "eval_one_episode"):
            logger.error("[TestEnv] module '.%s.deploy' does not have an eval_one_episode function", policy_name)
            raise AttributeError("Missing eval_one_episode in policy module")

        eval_module.eval_one_episode(TASK_ENV=self, model_client=self.model_client)

    def eval_one_episode_batch(self):
        policy_name = self.deploy_cfg["policy_name"]
        try:
            eval_module = __import__(
                f"XPolicyLab.policy.{policy_name}.deploy",
                fromlist=["eval_one_episode_batch"],
            )
        except ImportError as e:
            logger.error(
                "[TestEnv] failed to import policy module XPolicyLab.policy.%s.deploy: %s",
                policy_name,
                e,
            )
            raise e

        if not hasattr(eval_module, "eval_one_episode_batch"):
            logger.error("[TestEnv] module '.%s.deploy' does not have an eval_one_episode_batch function", policy_name)
            raise AttributeError("Missing eval_one_episode_batch in policy module")

        eval_module.eval_one_episode_batch(TASK_ENV=self, model_client=self.model_client)

    def mark_env_unstable(self, env_idx):
        """Flag an env as unstable so run_eval drops it from the eval set.

        Unstable envs produce no fail video and are not counted toward the
        eval total (they are accounted under ``unstable_nums`` instead).
        """
        self.unstable_envs.add(env_idx)

    def run_eval(self):
        self.run_reward()
        if hasattr(self, "get_score"):
            self.get_score()
        exist_envs = self.get_running_env_idx_list()
        if getattr(self, "interact", False):
            if hasattr(self, "query_support_arm_traj"):
                for env_idx in exist_envs:
                    self.query_support_arm_traj(env_idx=env_idx)
        if self.eval_batch:
            self.eval_one_episode_batch()
        else:
            self.eval_one_episode()
        success = 0
        process_scores = self.reward_manager.get_score() if hasattr(self, "get_score") else None
        # Envs flagged unstable during the episode (e.g. make_kong's
        # support-arm discard failed to knock the target tile down) are not
        # valid eval samples: skip their videos and exclude them from the
        # eval total, accounting them under unstable_nums instead.
        unstable_in_batch = [e for e in exist_envs if e in self.unstable_envs]
        if unstable_in_batch:
            self.unstable_nums += len(unstable_in_batch)
            self.episode_nums -= len(unstable_in_batch)
        eval_envs = [e for e in exist_envs if e not in self.unstable_envs]
        for idx, env_idx in enumerate(eval_envs):
            index = idx + self.success_nums + self.fail_nums
            episode_score = 0.0
            tag = "fail"
            if self.success[env_idx]:
                self.total_score += 1.0
                episode_score = 1.0
                success += 1
                tag = "success"
            elif process_scores is not None:
                episode_score = process_scores[env_idx] / 100.0
                self.total_score += episode_score

            # seed_list was filtered by completed/abandoned ids on resume,
            # so seed_list.index(seed) no longer yields the original
            # layout id. init_eval populates seed_list with numeric ids
            # parsed from filenames, so env_seeds[env_idx] is stable.
            layout_id = int(self.env_seeds[env_idx])
            layout_path = Path(self.seed_manager.seed_info[layout_id]["scene_layout"])
            detail = {
                "layout_id": layout_id,
                "layout_file": layout_path.name,
                "layout_sha256": hashlib.sha256(layout_path.read_bytes()).hexdigest(),
                "success": bool(self.success[env_idx]),
                "score": episode_score,
            }
            metadata_hook = getattr(self, "get_episode_metadata", None)
            if callable(metadata_hook):
                task_metadata = metadata_hook(env_idx)
                if task_metadata:
                    json.dumps(task_metadata)
                    detail["task_metadata"] = deepcopy(task_metadata)
            self.eval_result["details"][index] = detail
            video_path = os.path.join(self.save_dir, f"episode_{index:07d}.mp4")
            self.save_video(env_idx, video_path, tag)

        # Drop streams for envs not saved this batch (e.g. unstable ones).
        self._abort_video_writers()

        fail = self.episode_nums - success
        self.success_nums += success
        self.fail_nums += fail
        eval_time = self.success_nums + self.fail_nums
        if eval_time > 0:
            self.eval_result["success_rate"] = self.success_nums / eval_time
            self.eval_result["score"] = self.total_score / eval_time * 100
        self.eval_result["eval_time"] = eval_time
        save_json(self.eval_result, os.path.join(self.save_dir, "_result.json"))
        # Refresh the resume manifest at the end of every batch so that a
        # downstream SIGABRT (which beats the in-process PhysXFatalError
        # handler) still recovers everything up to the previous batch.
        try:
            self.persist_resume_manifest()
        except Exception as e:
            logger.warning("[EvalEnv] persist_resume_manifest after run_eval failed: %s", e)

    def is_episode_end(self):
        pre_end_flag = deepcopy(self.end_flag)
        final_check = False
        for env_idx in range(self.num_envs):
            if self.take_action_cnt[env_idx] >= self.step_lim or (
                not self.success[env_idx] and not self.end_flag[env_idx]
            ):
                final_check = True
                break
        reward_list = self.reward_manager.get_reward(final_check=final_check)
        for env_idx in range(self.num_envs):
            if self.end_flag[env_idx]:
                continue
            if reward_list[env_idx] > 1 - 1e-3:
                self.end_flag[env_idx] = True
                self.success[env_idx] = True
                continue
            if self.take_action_cnt[env_idx] >= self.step_lim or not self.success[env_idx]:
                self.end_flag[env_idx] = True
                self.success[env_idx] = False

        end_flag_changed_list = [
            env_idx for env_idx in range(self.num_envs) if self.end_flag[env_idx] != pre_end_flag[env_idx]
        ]
        if len(end_flag_changed_list) > 0:
            self.get_obs_batch(env_idx_list=end_flag_changed_list, last_frame=True)
        return all(self.end_flag)

    def get_running_env_idx_list(self):
        return [idx for idx in range(self.num_envs) if not self.end_flag[idx]]

    def have_empty(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = list(range(self.num_envs))
        return len(self.get_control_empty(env_idx_list=env_idx_list)) != 0

    def get_control_empty(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = list(range(self.num_envs))
        return self.robot_manager.control_manager.get_empty(env_idx_list=env_idx_list)
