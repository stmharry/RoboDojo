from copy import deepcopy
import logging
import os
from pathlib import Path
import random

import numpy as np
from omegaconf import DictConfig, OmegaConf
from shapely.geometry import box
import torch

from robodojo.sim.environment.global_configs import ASSETS_PATH
from robodojo.sim.utils.path import deep_resolve_paths, get_mdl_paths_from_folder, get_usd_paths_from_folder

logger = logging.getLogger(__name__)


def _as_host_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class FixtureSelectionService:
    def cluttered_generator_init(self, env_idx):
        for key in self.cluttered_generator.keys():
            if key in ["Table", "Ground"]:
                info = getattr(self, f"{key.lower()}_info")[env_idx]
                self.cluttered_generator[key][env_idx].reset(
                    box(
                        info.get("size", [0, 0, 0, 0])[0] + 0.05 + info.get("pos", [0, 0])[0],
                        info.get("size", [0, 0, 0, 0])[1] + 0.05 + info.get("pos", [0, 0])[1],
                        info.get("size", [0, 0, 0, 0])[2] - 0.05 + info.get("pos", [0, 0])[0],
                        info.get("size", [0, 0, 0, 0])[3] - 0.05 + info.get("pos", [0, 0])[1],
                    ),
                    frame=np.array([0.0, 0.0, info.get("height", 0), 1.0, 0.0, 0.0, 0.0]),
                )

    def select_background(self, background_cfg=None):
        if self.replay:
            for env_idx in range(self.num_envs):
                if self.saved_layouts[env_idx] is not None and "Background" in self.saved_layouts[env_idx]:
                    self.background = deepcopy(self.saved_layouts[env_idx]["Background"])
                    base_dir = f"{ASSETS_PATH}/Background"
                    category_name = self.background.get("category_name", "brown_photostudio_02_16k.hdr")
                    texture_path = f"{base_dir}/{category_name}"
                    self.background["texture_path"] = texture_path
                    return self.background
            return None
        if background_cfg is None:
            background_cfg = deepcopy(self.scene_config.Background)
            background_cfg = OmegaConf.to_container(background_cfg, resolve=True)
        base_dir = f"{ASSETS_PATH}/Background"
        is_randomized = background_cfg.get("random", False)
        if is_randomized:
            base_path = Path(base_dir)
            hdr_files = [str(p) for p in base_path.rglob("*") if p.is_file() and p.suffix.lower() == ".hdr"]
            if not hdr_files:
                texture_path = f"{base_dir}/{background_cfg.get('default', 'brown_photostudio_02_16k.hdr')}"
                category_name = background_cfg.get("default", "brown_photostudio_02_16k.hdr")
            else:
                texture_path = random.choice(hdr_files)
                category_name = Path(texture_path).name
        else:
            texture_path = f"{base_dir}/{background_cfg.get('default', 'brown_photostudio_02_16k.hdr')}"
            category_name = background_cfg.get("default", "brown_photostudio_02_16k.hdr")
        if "random" in background_cfg:
            background_cfg.pop("random")
        if "default" in background_cfg:
            background_cfg.pop("default")
        background_cfg["texture_path"] = texture_path
        background_cfg["category_name"] = category_name
        self.background = background_cfg
        return background_cfg

    def select_room(self, env_idx, room_cfg=None):
        if room_cfg is None:
            room_cfg = deepcopy(self.scene_config.Room)
            room_cfg = OmegaConf.to_container(room_cfg, resolve=True)
        room_dirs = os.listdir(f"{ASSETS_PATH}/Room")
        is_randomized = room_cfg.get("random", False)
        if is_randomized:
            cat_name = random.choice(room_dirs)
        else:
            cat_name = room_cfg.get("default", "base0")
        room_dir = os.path.join(ASSETS_PATH, "Room", cat_name)
        usd_files = [f for f in os.listdir(room_dir) if f.endswith(".usd")]
        if len(usd_files) == 0:
            raise ValueError(f"No USD files found in {room_dir} for room category {cat_name}.")
        usd_path = os.path.join(room_dir, usd_files[0])
        if not usd_path:
            raise ValueError(f"env_id {env_idx} room category {cat_name} missing 'usd_path' field.")
        prim_path, inst_name = self._generate_object_paths(env_idx, cat_name, len(self.rooms[env_idx]), type="Rooms")
        self.rooms[env_idx].append(inst_name)
        self.instance_type_by_env[env_idx][inst_name] = "room"
        if "random" in room_cfg:
            room_cfg.pop("random")
        room_cfg["usd_path"] = usd_path
        room_cfg["prim_path"] = prim_path
        room_cfg["inst_name"] = inst_name
        return room_cfg

    def select_light(self, env_idx, light_cfg=None):
        if light_cfg is None:
            light_cfg = deepcopy(self.scene_config.Light)
            light_cfg = OmegaConf.to_container(light_cfg, resolve=True)
        for light_config_key in light_cfg.keys():
            if isinstance(light_cfg[light_config_key]["types"], list):
                light_type = np.random.choice(light_cfg[light_config_key]["types"])
            else:
                light_type = light_cfg[light_config_key]["types"]
            light_cfg[light_config_key]["types"] = str(light_type)
            all_light_type = ["Rect", "Sphere", "Cylinder", "Dome", "Disk"]
            for lt in all_light_type:
                if lt != light_type and lt in light_cfg[light_config_key]:
                    light_cfg[light_config_key].pop(lt)
            prim_path, inst_name = self._generate_object_paths(
                env_idx, light_type, len(self.lights[env_idx]), type="Light"
            )
            light_cfg[light_config_key]["prim_path"] = prim_path
            light_cfg[light_config_key]["inst_name"] = inst_name
            self.lights[env_idx].append(inst_name)
            self.instance_type_by_env[env_idx][inst_name] = "light"
        return light_cfg

    def select_ground(self, env_idx, ground_cfg=None):
        if ground_cfg is None:
            ground_cfg = deepcopy(self.scene_config.Ground)
            ground_cfg = OmegaConf.to_container(ground_cfg, resolve=True)
        self.ground_info[env_idx] = {
            "pos": ground_cfg.get("default_pos", [0, 0, 0]),
            "size": [-self.env_spacing / 2, -self.env_spacing / 2, self.env_spacing / 2, self.env_spacing / 2],
            "height": ground_cfg.get("thickness", 0.1) / 2,
        }
        prim_path, inst_name = self._generate_object_paths(env_idx, "Ground", len(self.grounds[env_idx]), type="Ground")
        ground_cfg["prim_path"] = prim_path
        ground_cfg["inst_name"] = inst_name
        self.grounds[env_idx].append(inst_name)
        self.instance_type_by_env[env_idx][inst_name] = "ground"
        return ground_cfg

    def select_table(self, env_idx, table_cfg=None):
        if table_cfg is None:
            table_cfg = deepcopy(self.scene_config.Table)
            table_cfg = OmegaConf.to_container(table_cfg, resolve=True)
        default_pos = table_cfg.get("default_pos")
        scale = table_cfg.get("scale")
        self.table_info[env_idx] = {
            "pos": default_pos,
            "size": [-scale[0] / 2, -scale[1] / 2, scale[0] / 2, scale[1] / 2],
            "height": default_pos[2] + scale[2] / 2,
        }
        table_dirs = sorted(d for d in os.listdir(f"{ASSETS_PATH}/Material") if d.lower().startswith("material"))
        is_randomized = table_cfg.get("random", False)
        if is_randomized:
            cat_name = random.choice(table_dirs)
        else:
            cat_name = table_cfg.get("replay_material_override", table_cfg.get("default", "material_0001"))
        mdl_dir = os.path.join(ASSETS_PATH, "Material", cat_name)
        mdl_files = [f for f in os.listdir(mdl_dir) if f.endswith(".mdl")]
        table_cfg["mdl_path"] = os.path.join(mdl_dir, mdl_files[0])
        table_cfg["mdl_name"] = os.path.splitext(mdl_files[0])[0]
        prim_path, inst_name = self._generate_object_paths(env_idx, cat_name, len(self.tables[env_idx]), type="Table")
        table_cfg["prim_path"] = prim_path
        table_cfg["inst_name"] = inst_name
        self.tables[env_idx].append(inst_name)
        self.instance_type_by_env[env_idx][inst_name] = "table"
        return table_cfg

    def process_visual(self, visual_cfg):
        if visual_cfg is None:
            return None
        visual_cfg = deepcopy(visual_cfg)
        if not isinstance(visual_cfg, DictConfig):
            visual_cfg = OmegaConf.create(visual_cfg)
            deep_resolve_paths(visual_cfg)
            visual_cfg = OmegaConf.to_container(visual_cfg, resolve=True)
        color_list = visual_cfg.get("color", None)
        if color_list is not None and isinstance(color_list[0], (int, float)):
            color_list = [color_list]
        visual_material_usd_folder = None
        visual_material_mdl_folder = None
        if color_list:
            _current_color = np.random.choice(color_list)
            visual_cfg["color"] = _current_color
        elif visual_cfg.get("material_usd_folder", None) is not None:
            visual_material_usd_folder = visual_cfg.get("material_usd_folder", None)
            if visual_material_usd_folder is not None:
                visual_usd_paths = get_usd_paths_from_folder(
                    folder_path=visual_material_usd_folder, skip_keywords=[".thumbs"]
                )
                if visual_usd_paths:
                    selected_indices = torch.randint(low=0, high=len(visual_usd_paths), size=(1,)).tolist()
                    visual_cfg["visual_usd_path"] = visual_usd_paths[selected_indices[0]]
        else:
            visual_material_mdl_folder = visual_cfg.get("material_mdl_folder", None)
            if visual_material_mdl_folder is not None:
                visual_mdl_paths = get_mdl_paths_from_folder(folder_path=visual_material_mdl_folder)
                if visual_mdl_paths:
                    selected_indices = torch.randint(low=0, high=len(visual_mdl_paths), size=(1,)).tolist()
                    visual_cfg["visual_mdl_path"] = visual_mdl_paths[selected_indices[0]]
        if "material_usd_folder" in visual_cfg:
            visual_cfg.pop("material_usd_folder")
        if "material_mdl_folder" in visual_cfg:
            visual_cfg.pop("material_mdl_folder")
        return visual_cfg
