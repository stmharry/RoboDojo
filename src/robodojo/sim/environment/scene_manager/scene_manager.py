from collections.abc import Sequence
import logging
from typing import Any, Dict, List

from isaacsim.core.utils.stage import get_current_stage
from omegaconf import DictConfig
import torch

from robodojo.sim.environment.environment.isaac.isaac_rl_env import IsaacRLEnv
from robodojo.sim.environment.scene_manager.layout_manager import LayoutManager
from robodojo.sim.environment.scene_manager.objects.articulation import ArticulationObject
from robodojo.sim.environment.scene_manager.objects.background import Background
from robodojo.sim.environment.scene_manager.objects.fluid import FluidObject
from robodojo.sim.environment.scene_manager.objects.garment import GarmentObject
from robodojo.sim.environment.scene_manager.objects.geometry import GeometryObject
from robodojo.sim.environment.scene_manager.objects.ground import Ground
from robodojo.sim.environment.scene_manager.objects.light import Light
from robodojo.sim.environment.scene_manager.objects.room import Room
from robodojo.sim.environment.scene_manager.services.object_factory import ObjectFactoryService
from robodojo.sim.environment.scene_manager.services.scene_lifecycle import SceneLifecycleService
from robodojo.sim.environment.scene_manager.services.scene_queries import SceneQueriesService

logger = logging.getLogger(__name__)


class SceneManager(ObjectFactoryService, SceneLifecycleService, SceneQueriesService):
    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        env_spacing: float,
        scene_config: DictConfig,
        task_config: DictConfig,
        use_fabric: bool = False,
        seeds_per_env: Sequence[int] | None = None,
    ):
        """Initialize the SceneManager for managing multiple parallel environments and their objects.

        Args:
            num_envs: Number of parallel environments to manage
            config: Configuration dictionary containing environment and object parameters
            device: PyTorch device (CPU/GPU) for computations
            layout_manager: LayoutManager instance for position management
            use_fabric: Whether to use fabric physics engine
        """
        self.config = DictConfig({})
        self.background_config = DictConfig({})
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.env_spacing = env_spacing
        self.use_fabric = use_fabric
        self.stage = get_current_stage()
        self.sim: IsaacRLEnv | None = None
        self.layout_manager = LayoutManager(
            num_envs=num_envs,
            env_spacing=env_spacing,
            seeds_per_env=seeds_per_env,
            scene_config=scene_config,
            task_config=task_config,
        )
        self.layout_manager.scene_manager = self  # Provide reference to SceneManager for LayoutManager
        if seeds_per_env is not None:
            self.update_env_seeds(seeds_per_env)
        else:
            self._seeds_per_env = None

        # Environment roots and origins
        self.env_roots = [f"/World/envs/env_{env_id}" for env_id in range(num_envs)]

        # Object storage structure: [env_id][obj_key] = object
        self._rigid_and_dynamic_objects: List[Dict[str, Any]] = [{} for _ in range(num_envs)]
        self._articulation_objects: List[Dict[str, ArticulationObject]] = [{} for _ in range(num_envs)]
        self._garment_objects: List[Dict[str, GarmentObject]] = [{} for _ in range(num_envs)]
        self._geometry_objects: List[Dict[str, GeometryObject]] = [{} for _ in range(num_envs)]
        self._fluid_objects: List[Dict[str, FluidObject]] = [{} for _ in range(num_envs)]

        # Rooms and lights remain environment-specific
        self._rooms: List[Room | None] = [None] * num_envs
        self._tables: List[Any | None] = [None] * num_envs
        self._grounds: List[List[Ground]] = [[] for _ in range(num_envs)]
        self._lights: List[List[Light]] = [[] for _ in range(num_envs)]

        # Category counters managed by environment and category
        self._category_counters: Dict[int, Dict[str, int]] = {env_id: {} for env_id in range(num_envs)}
        # Background (shared across all environments)
        self._background: Background | None = None
        self.background_prim_path = "/World/background"
        self.pending_initialization = []
        self.spawnable_object_types = [
            "rigid",
            "articulation",
            "garment",
            "geometry",
            "fluid",
        ]
