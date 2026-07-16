import json
import logging
import os
import re
from typing import List

from isaacsim.core.utils.prims import is_prim_path_valid
import numpy as np
import torch

from robodojo.sim.environment.global_configs import OBJECTS_PATH

logger = logging.getLogger(__name__)


def _as_host_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class LayoutObjectsService:
    @staticmethod
    def format_label_map_key(model_name: str, model_idx: int) -> str:
        return f"{model_name}/{int(model_idx):05d}"

    @staticmethod
    def _load_category_label_map(objects_path: str, category_name: str) -> dict:
        map_path = os.path.join(objects_path, category_name, "map.json")
        if not os.path.isfile(map_path):
            raise FileNotFoundError(f"Label map file does not exist: {map_path}")
        with open(map_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid label map format in {map_path}, expected a JSON object.")
        return data

    @staticmethod
    def decode_inst_name(object_name: str):
        pattern = "^(.+)_(\\d+)_(\\d+)$"
        match = re.match(pattern, object_name)
        if match:
            model_name = match.group(1)
            model_id = match.group(2)
            return (model_name, int(model_id))
        else:
            raise ValueError(
                f"Object name '{object_name}' does not match the expected pattern 'modelname_id_randomint'"
            )

    @staticmethod
    def decode_object_name(object_name: str):
        pattern = "^(.+)_(\\d+)"
        match = re.match(pattern, object_name)
        if match:
            model_name = match.group(1)
            model_id = match.group(2)
            return (model_name, int(model_id))
        else:
            raise ValueError(f"Object name '{object_name}' does not match the expected pattern 'modelname_id'")

    def _get_category_indices(self, cat_name: List[str], type: str) -> List[List[int]]:
        cat_idx = []
        for name in cat_name:
            modeldir = os.path.join(OBJECTS_PATH, type, name)
            if not os.path.isdir(modeldir):
                cat_idx.append([])
                continue
            model_indices = os.listdir(modeldir)
            indices = []
            for model_idx in model_indices:
                try:
                    idx = int(model_idx)
                    indices.append(idx)
                except ValueError:
                    continue
            cat_idx.append(indices)
        return cat_idx

    @staticmethod
    def _get_object_usd_path(model_info: dict) -> str:
        object_dir = os.path.join(model_info["root"], f"{model_info['model_id']:05d}")
        usd_path = os.path.join(object_dir, "object.usdz")
        if not os.path.exists(usd_path):
            usd_path = os.path.join(object_dir, "object.usd")
        return usd_path

    def _generate_object_paths(self, env_idx: int, cat_name: str, model_idx: int, type: str) -> tuple[str, str]:
        """Generate sequential prim path and instance name for a category.

        Args:
            env_idx: Environment ID
            cat_name: Category name
            model_idx: Model index
            type: Object type

        Returns:
            Tuple of (prim_path, inst_name)
        """
        env_root = self.env_roots[env_idx]
        obj_id = self._get_next_object_id(env_idx, cat_name, model_idx, type)
        inst_name = f"{cat_name}_{model_idx}_{obj_id}"
        prim_path = f"{env_root}/{type}/{cat_name}/{inst_name}"
        return (prim_path, inst_name)

    def _get_next_object_id(self, env_idx: int, cat_name: str, model_idx: int, type: str) -> int:
        """Get the next available ID for a category in the specified environment.

        Args:
            env_idx: Environment ID
            cat_name: Category name
            model_idx: Model index
            type: Object type

        Returns:
            Next available integer ID for this category
        """
        while is_prim_path_valid(
            f"{self.env_roots[env_idx]}/{type}/{cat_name}/{cat_name}_{model_idx}_{self._object_id_counters}"
        ):
            self._object_id_counters += 1
        self._object_id_counters += 1
        return self._object_id_counters
