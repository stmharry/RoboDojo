from collections.abc import Mapping as ABCMapping, Sequence
from copy import deepcopy
import logging
import os
from typing import List

import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from robodojo.sim.environment.global_configs import OBJECTS_PATH
from robodojo.sim.environment.scene_manager.appearance import merge_fixture_appearance
from robodojo.sim.environment.seeding import seed_everywhere
from robodojo.sim.utils.load_file import load_desc_info, load_object_metadata
from robodojo.sim.utils.path import (
    resolve_path,
)
from robodojo.sim.utils.transformer import cal_quat_dis

logger = logging.getLogger(__name__)


def _as_host_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class LayoutStateService:
    def get_layout_records(self, env_idx: int, object_type: str):
        return self.object_records_by_type[object_type].layout_records_by_env[env_idx]

    def set_saved_layout(self, env_idx: int, layout):
        self.saved_layouts[env_idx] = layout

    def clear_object_records(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = list(range(self.num_envs))
        for env_idx in env_idx_list:
            for records in self.object_records_by_type.values():
                records.clear_env(env_idx)

    def clear_fixture_records(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = list(range(self.num_envs))
        for env_idx in env_idx_list:
            self.rooms[env_idx] = []
            self.tables[env_idx] = []
            self.grounds[env_idx] = []
            self.lights[env_idx] = []
            self.table_info[env_idx] = {}
            self.ground_info[env_idx] = {}

    def clear_layout_state(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = list(range(self.num_envs))
        self.clear_object_records(env_idx_list)
        self.clear_fixture_records(env_idx_list)
        for env_idx in env_idx_list:
            self.instance_type_by_env[env_idx] = {}
            self.layout_valid[env_idx] = True
            for generator in self.cluttered_generator.values():
                generator[env_idx].reset()

    def load_saved_layout(self, env_idx):
        self._set_env_seed(env_idx)
        env_config = deepcopy(self.saved_layouts[env_idx])
        if env_config is None:
            return None
        # Bundled layouts may keep immutable fixture definitions in the scene
        # component and record only per-layout poses/objects. Merge saved
        # overrides onto those defaults so a compact layout cannot accidentally
        # drop the selected scene's material, lighting, or physics contract.
        for fixture in ("Room", "Table", "Ground", "Light"):
            defaults = self.scene_config.get(fixture)
            if defaults is not None:
                merged = OmegaConf.merge(deepcopy(defaults), deepcopy(env_config.get(fixture) or {}))
                env_config[fixture] = OmegaConf.to_container(merged, resolve=True)
        self.clear_layout_state([env_idx])
        for key, value in env_config.items():
            if key in ["Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid"]:
                for cat, inst_list in value.items():
                    for inst in inst_list:
                        cat_idx = inst.get("category_idx", None)
                        prim_path, inst_name = self._generate_object_paths(env_idx, cat, cat_idx, type=key.lower())
                        if key in ["Rigid", "Dynamic", "Garment", "Articulation", "Geometry", "Fluid"]:
                            if "type" in inst and inst["type"] == "cluttered":
                                usd_path = f"{OBJECTS_PATH}/Clutter/{cat}/{cat_idx:05d}/object.usdz"
                            else:
                                usd_path = f"{OBJECTS_PATH}/{key}/{cat}/{cat_idx:05d}/object.usdz"
                            if not os.path.exists(usd_path):
                                usd_path = f"{OBJECTS_PATH}/{key}/{cat}/{cat_idx:05d}/object.usd"
                        if "visual" in inst:
                            visual_cfg = inst.get("visual")
                            if isinstance(visual_cfg, DictConfig):
                                visual_cfg = OmegaConf.to_container(visual_cfg, resolve=True)
                            if isinstance(visual_cfg, dict):
                                visual_usd_path = visual_cfg.get("visual_usd_path", None)
                                if isinstance(visual_usd_path, str) and "$" in visual_usd_path:
                                    visual_cfg["visual_usd_path"] = resolve_path(visual_usd_path)
                                visual_mdl_path = visual_cfg.get("visual_mdl_path", None)
                                if isinstance(visual_mdl_path, str) and "$" in visual_mdl_path:
                                    visual_cfg["visual_mdl_path"] = resolve_path(visual_mdl_path)
                                inst["visual"] = visual_cfg
                        inst["prim_path"] = prim_path
                        inst["usd_path"] = usd_path
                        inst["inst_name"] = inst_name
                        if "type" in inst and inst["type"] == "cluttered":
                            modeldir = f"{OBJECTS_PATH}/Clutter/{cat}"
                        else:
                            modeldir = f"{OBJECTS_PATH}/{key}/{cat}"
                        data = load_object_metadata(modeldir, cat_idx)
                        data["model_name"] = cat
                        data["model_id"] = cat_idx
                        info = load_desc_info(modeldir, cat_idx, key)
                        if info is not None:
                            data.update(info)
                        self.object_records_by_type[key].add_instance(env_idx, inst, data)
                        self.instance_type_by_env[env_idx][inst_name] = key.lower()
            elif key == "Room":
                value = merge_fixture_appearance(value, self.scene_config.Room)
                value = self.select_room(env_idx, room_cfg=value)
                env_config[key] = value
            elif key == "Table":
                value = merge_fixture_appearance(value, self.scene_config.Table)
                value = self.select_table(env_idx, table_cfg=value)
                env_config[key] = value
            elif key == "Ground":
                value = self.select_ground(env_idx, ground_cfg=value)
            elif key == "Light":
                value = self.select_light(env_idx, light_cfg=value)
        self.cluttered_generator_init(env_idx)
        return env_config

    def check_layout_stability(self, env, render=False):
        first_pose = []
        stable = []
        unstable = []
        history_pose = []
        window_success_actor = []
        for env_idx in range(self.num_envs):
            actors = []
            for object_type in ["Rigid", "Articulation"]:
                for obj in self.object_records_by_type[object_type].layout_records_by_env[env_idx]:
                    check = True
                    relative_plane = obj.get("relative_plane", None)
                    relative_plane = relative_plane.split("/")[0]
                    if not obj.get("need_check_stable", True):
                        check = False
                    if relative_plane is not None and relative_plane not in ["table", "ground"]:
                        inst_name = self.get_instance_name(env_idx, relative_plane)
                        physics_type = self.instance_type_by_env[env_idx].get(inst_name, None)
                        if physics_type == "dynamic":
                            check = False
                    if check:
                        actors.append(obj["inst_name"])
            first_pose.append({})
            stable.append(set(actors))
            unstable.append([])
            history_pose.append({})
            history_pose[env_idx] = {actor: [] for actor in set(actors)}
            window_success_actor.append([])

        def rot_diff(r1, r2):
            return np.abs(cal_quat_dis(r1, r2) * 180 / np.pi)

        def get_pose(actor, env_idx):
            pos, rot = self.get_instance_pose(env_idx=env_idx, inst_name=actor)
            if isinstance(pos, torch.Tensor):
                pos = pos.cpu().numpy().flatten()
            if isinstance(rot, torch.Tensor):
                rot = rot.cpu().numpy().flatten()
            return (pos, rot)

        def is_stable(prev_pos, prev_rot, cur_pos, cur_rot, angle=5, eps=0.04):
            if rot_diff(prev_rot, cur_rot) > angle:
                return False
            delta = np.abs(cur_pos - prev_pos)
            if any(delta > eps):
                return False
            if cur_pos[2] < self.table_info[env_idx].get("height", 0.765) - 0.05:
                return False
            return True

        maxstep = 300
        window = 100
        unstable_envs = []
        for step in range(maxstep):
            env.sim_step(render=render)
            for env_idx in range(self.num_envs):
                if not env.success[env_idx]:
                    continue
                if unstable[env_idx] is not None and len(unstable[env_idx]) > 0:
                    env.success[env_idx] = False
                    env.end_flag[env_idx] = True
                    unstable_envs.append(env_idx)
                    continue
                for actor in list(stable[env_idx]):
                    pos, rot = get_pose(actor, env_idx)
                    if step == 0:
                        first_pose[env_idx][actor] = (pos, rot)
                        continue
                    if not is_stable(first_pose[env_idx][actor][0], first_pose[env_idx][actor][1], pos, rot, angle=30):
                        stable[env_idx].remove(actor)
                        unstable[env_idx].append(actor)
                    history_pose[env_idx][actor].append((pos, rot))
            if step >= maxstep - window:
                for env_idx in range(self.num_envs):
                    if env.success[env_idx] is False:
                        continue
                    for actor in list(stable[env_idx]):
                        prev_pos, prev_rot = history_pose[env_idx][actor][maxstep - window - 1]
                        cur_pos, cur_rot = history_pose[env_idx][actor][-1]
                        if not is_stable(prev_pos, prev_rot, cur_pos, cur_rot, angle=10):
                            stable[env_idx].remove(actor)
                            unstable[env_idx].append(actor)
                            unstable_envs.append(env_idx)
                            env.success[env_idx] = False
                            env.end_flag[env_idx] = True
        num_fail = 0
        for env_idx in range(self.num_envs):
            if env.success[env_idx] is False:
                num_fail += 1
                env.end_flag[env_idx] = True
            elif len(unstable[env_idx]) > 0:
                num_fail += 1
                env.success[env_idx] = False
                env.end_flag[env_idx] = True
                unstable_envs.append(env_idx)
        if len(unstable_envs) > 0:
            logger.warning("Unstable envs: %s", unstable_envs)
        return (num_fail != self.num_envs, unstable_envs)

    @staticmethod
    def required_keys(data: dict, keys: str | List[str], sep: str = "/") -> bool:
        def _is_missing(value) -> bool:
            return value is None or value == {}

        def _get_by_path(obj, path: str):
            if path is None:
                return None
            if not isinstance(path, str):
                return None
            path = path.strip(sep)
            if path == "":
                return None
            cur = obj
            for part in path.split(sep):
                if part == "":
                    continue
                if isinstance(cur, ABCMapping):
                    cur = cur.get(part, None)
                elif isinstance(cur, list):
                    try:
                        idx = int(part)
                    except Exception:
                        return None
                    if idx < 0 or idx >= len(cur):
                        return None
                    cur = cur[idx]
                else:
                    return None
                if _is_missing(cur):
                    return None
            return cur

        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            val = _get_by_path(data, key)
            if _is_missing(val):
                return False
        return True

    def update_env_seeds(self, seeds: Sequence[int] | None):
        """Update per-environment seed list."""
        if seeds is None:
            self._seeds_per_env = None
            return
        seed_list = [int(s) for s in seeds]
        if len(seed_list) != self.num_envs:
            raise ValueError(f"seed list length {len(seed_list)} does not match num_envs {self.num_envs}.")
        self._seeds_per_env = seed_list

    def _set_env_seed(self, env_id: int):
        if self._seeds_per_env is None:
            return
        if env_id >= len(self._seeds_per_env):
            raise IndexError(f"Requested env_id {env_id} exceeds configured seeds (len={len(self._seeds_per_env)}).")
        seed_everywhere(self._seeds_per_env[env_id])

    def _add(self, env_info, inst_name, inst_info, type="Rigid"):
        model_name, _ = self.decode_inst_name(inst_name)
        if type not in env_info:
            env_info[type] = {}
        if model_name not in env_info[type]:
            env_info[type][model_name] = []
        env_info[type][model_name].append(inst_info)
        return env_info

    def get_instance_metadata(self, env_idx: int, inst_name: str = None, label: str = None):
        if inst_name is None:
            inst_name = self.get_instance_name(env_idx=env_idx, label=label)
        if inst_name is None:
            return None
        instance_type = self.instance_type_by_env[env_idx].get(inst_name)
        if instance_type is None:
            return None
        object_type = instance_type.capitalize()
        if object_type not in self.object_records_by_type:
            return None
        return self.object_records_by_type[object_type].metadata_by_env[env_idx].get(inst_name, None)
