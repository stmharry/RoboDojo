from dataclasses import dataclass
import logging
from typing import Any, List

import numpy as np
from omegaconf import DictConfig, open_dict
import torch

from robodojo.sim.environment.scene_manager.services.fixture_selection import FixtureSelectionService
from robodojo.sim.environment.scene_manager.services.layout_objects import LayoutObjectsService
from robodojo.sim.environment.scene_manager.services.layout_state import LayoutStateService
from robodojo.sim.environment.scene_manager.services.spatial_queries import SpatialQueriesService
from robodojo.sim.utils.cluttered_generator import ClutteredGenerator

logger = logging.getLogger(__name__)

OBJECT_CONFIG_TYPES = ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid")


def _as_host_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass
class ObjectTypeRecords:
    layout_records_by_env: list[list[dict[str, Any]]]
    metadata_by_env: list[dict[str, dict[str, Any]]]

    @classmethod
    def create(cls, num_envs: int):
        return cls(
            layout_records_by_env=[[] for _ in range(num_envs)],
            metadata_by_env=[{} for _ in range(num_envs)],
        )

    def clear_env(self, env_idx: int):
        self.layout_records_by_env[env_idx] = []
        self.metadata_by_env[env_idx] = {}

    def add_instance(self, env_idx: int, layout_record: dict[str, Any], metadata: dict[str, Any]):
        inst_name = layout_record["inst_name"]
        self.layout_records_by_env[env_idx].append(layout_record)
        self.metadata_by_env[env_idx][inst_name] = metadata


class LayoutManager(LayoutStateService, FixtureSelectionService, SpatialQueriesService, LayoutObjectsService):
    _FAR_CENTER = (100000.0, 100000.0, 100000)
    _FAR_STEP = 1000.0
    _FAR_JITTER = 0.5

    def __init__(
        self,
        num_envs,
        env_spacing,
        seeds_per_env: List[int] = None,
        scene_config: DictConfig = None,
        task_config: DictConfig = None,
    ):
        self.scene_manager = None
        self._far_idx = 0
        self.num_envs = num_envs
        self.env_spacing = env_spacing
        self.layout_valid = [True] * self.num_envs
        self.env_roots = [f"/World/envs/env_{env_idx}" for env_idx in range(self.num_envs)]
        self.caption_info = dict()
        self._object_id_counters = 0
        self.scene_config = scene_config
        self.task_config = task_config
        self.process_config()
        self.cluttered_generator = {
            "Table": [ClutteredGenerator() for _ in range(self.num_envs)],
            "Ground": [ClutteredGenerator() for _ in range(self.num_envs)],
        }
        self.object_records_by_type = {
            object_type: ObjectTypeRecords.create(self.num_envs) for object_type in OBJECT_CONFIG_TYPES
        }
        self.rooms = [[] for _ in range(self.num_envs)]
        self.tables = [[] for _ in range(self.num_envs)]
        self.grounds = [[] for _ in range(self.num_envs)]
        self.lights = [[] for _ in range(self.num_envs)]
        self.table_info = [{} for _ in range(self.num_envs)]
        self.ground_info = [{} for _ in range(self.num_envs)]
        self.background = None
        self.instance_type_by_env = [{} for _ in range(self.num_envs)]
        self.saved_layouts = [None for _ in range(self.num_envs)]
        self.replay = True

    def process_config(self):
        keys = ["Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid"]
        with open_dict(self.scene_config), open_dict(self.task_config):
            for key in keys:
                if key in self.scene_config:
                    config = self.scene_config[key]
                    del self.scene_config[key]
                    if key not in self.task_config:
                        self.task_config[key] = config
                    else:
                        self.task_config[key].extend(config)

    def close(self):
        self._far_idx = 0
        self.layout_valid = [True] * self.num_envs
        self.saved_layouts = [None for _ in range(self.num_envs)]
        self.background = None
        self.clear_layout_state()
        self.cluttered_generator = {
            "Table": [ClutteredGenerator() for _ in range(self.num_envs)],
            "Ground": [ClutteredGenerator() for _ in range(self.num_envs)],
        }
