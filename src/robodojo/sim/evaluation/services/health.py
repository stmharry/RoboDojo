from __future__ import annotations

import logging

import numpy as np
import transforms3d as t3d

logger = logging.getLogger(__name__)


class HealthService:
    def _check_physx_broken_envs(self):
        if not self.physx_monitor_enabled:
            return
        # Fatal kernel-failure short-circuit: the GPU solver has died and
        # nothing inside this process can recover. Surface immediately
        # so main.py can persist progress and re-exec.
        monitor = self._physx_get_monitor()
        if monitor.is_fatal():
            raise self._PhysXFatalError(monitor.get_fatal_message())
        bad = monitor.get_broken_envs()
        new_bad = {i for i in bad if i < self.num_envs and not self.end_flag[i]}
        if new_bad:
            raise self._PhysXBrokenError(new_bad)

    def _check_endpose_finite(self, env_idx_list):
        """NaN backstop run right before obs_manager.get_obs.

        Carb's PhysX warnings are captured by PhysXWarningMonitor via fd
        interception. Kit/Carb rebinds its logger fd across
        env.close()/recreate cycles, so the monitor occasionally misses
        warnings. When a miss happens, the next obs_manager.get_obs
        feeds a NaN-laden 3x3 into mat2quat, which throws LinAlgError.
        main.py's generic except then sees an empty broken-env set,
        falls through to seed_manager.eval_step(), and silently consumes
        the whole batch of seeds.

        This method recomputes the exact 3x3 that get_delta_endpose is
        about to hand to mat2quat and verifies np.isfinite on it. Any env
        that fails the check is treated as PhysX-broken and we raise
        PhysXBrokenError so main.py's existing recovery path (abandon
        seed -> refill from queue -> retry round) fires.
        """
        if not self.physx_monitor_enabled:
            return
        bad = set()
        for robot in self.robot_manager.robot_list:
            poses = self.robot_manager.get_real_endpose(robot, env_idx_list=env_idx_list, is_relative=True)
            for env_idx in env_idx_list:
                if env_idx >= self.num_envs or self.end_flag[env_idx]:
                    continue
                ee_pose = poses.get(env_idx)
                if ee_pose is None:
                    continue
                if not np.isfinite(ee_pose).all():
                    bad.add(env_idx)
                    continue
                try:
                    rot_3x3 = t3d.quaternions.quat2mat(ee_pose[-4:]) @ robot.delta_matrix
                except Exception:
                    bad.add(env_idx)
                    continue
                if not np.isfinite(rot_3x3).all():
                    bad.add(env_idx)
        if bad:
            self._physx_get_monitor().add_broken_envs(bad)
            raise self._PhysXBrokenError(bad)

    def get_seeds_for_envs(self, env_idxs) -> set:
        return {self.current_env_seed_map[i] for i in env_idxs if i in self.current_env_seed_map}
