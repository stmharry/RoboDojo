from collections.abc import Sequence
from copy import deepcopy
import logging
from typing import Any, List

from isaacsim.core.utils.prims import delete_prim, is_prim_path_valid
import torch

from robodojo.sim.environment.environment.isaac.isaac_rl_env import IsaacRLEnv
from robodojo.sim.environment.scene_manager.active_pose import validate_active_pose
from robodojo.sim.environment.scene_manager.objects.articulation import ArticulationObject
from robodojo.sim.environment.scene_manager.objects.background import Background
from robodojo.sim.environment.scene_manager.objects.fluid import FluidObject
from robodojo.sim.environment.scene_manager.objects.garment import GarmentObject
from robodojo.sim.environment.scene_manager.objects.geometry import GeometryObject
from robodojo.sim.environment.scene_manager.objects.ground import Ground
from robodojo.sim.environment.scene_manager.objects.light import Light
from robodojo.sim.environment.scene_manager.objects.rigid import RigidObject
from robodojo.sim.environment.scene_manager.objects.room import Room
from robodojo.sim.environment.scene_manager.objects.table import Table
from robodojo.sim.environment.scene_manager.pose_restore import restore_saved_poses
from robodojo.sim.environment.seeding import seed_everywhere

logger = logging.getLogger(__name__)


class SceneLifecycleService:
    def update_env_seeds(self, seeds: Sequence[int] | None):
        """Update per-environment seed list."""
        if seeds is None:
            self._seeds_per_env = None
            return
        seed_list = [int(s) for s in seeds]
        if len(seed_list) != self.num_envs:
            raise ValueError(f"seed list length {len(seed_list)} does not match num_envs {self.num_envs}.")
        self._seeds_per_env = seed_list
        if self.layout_manager is not None:
            self.layout_manager.update_env_seeds(seeds)

    def initialize(self, sim: IsaacRLEnv):
        """Initialize the scene manager."""
        self.sim = sim
        self.env_origins = self.sim.scene.env_origins

        self._setup_background()
        for env_idx in range(self.num_envs):
            self.config[f"env{env_idx}"] = self.layout_manager.load_saved_layout(env_idx)

        for env_idx in range(self.num_envs):
            self._set_env_seed(env_idx)
            if self.device.type == "cuda":
                # Static geometry must exist before cameras are attached to any
                # fixture-published mount frames. Dynamic and articulation
                # objects also need to be present before the simulation starts.
                self.spawn_scene_objects(
                    env_id=env_idx,
                    exclude_types=list(set(self.spawnable_object_types) - {"rigid", "articulation", "geometry"}),
                )
            else:
                self.spawn_scene_objects(
                    env_id=env_idx,
                    exclude_types=list(set(self.spawnable_object_types) - {"articulation"}),
                )

        self.setup_scene = True

    def post_init(self):
        """Post-initialization logic.
        Initialize objects created in the initialize() method here.
        For CUDA devices, initialize rigid and articulation objects here.
        """
        for env_id in range(self.num_envs):
            # Initialize rigid and articulation objects
            for obj_dict in [
                self._rigid_and_dynamic_objects[env_id],
                self._articulation_objects[env_id],
            ]:
                for obj in obj_dict.values():
                    obj.initialize()

    def _set_env_seed(self, env_id: int):
        if self._seeds_per_env is None:
            return
        if env_id >= len(self._seeds_per_env):
            raise IndexError(f"Requested env_id {env_id} exceeds configured seeds (len={len(self._seeds_per_env)}).")
        seed_everywhere(self._seeds_per_env[env_id])

    def reload_scene(self):
        """Reload all environments from saved eval layouts."""
        logger.debug("Resetting all environments")
        self.post_init()
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reload_envs(env_ids)
        self.setup_scene = False

    def reload_envs(self, env_ids: Sequence[int]):
        """Reload specified environments from saved eval layouts.

        Args:
            env_ids: Sequence of environment IDs to reset
        """
        if self._background and not self.setup_scene:
            self.reload_background()

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.int32)

        for env_id in env_ids:
            env_id = int(env_id)
            self._set_env_seed(env_id)
            self.reload_env_scene(env_id)

        for obj in self.pending_initialization:
            obj.initialize()
        self.pending_initialization.clear()

        self.sim.sim_step()
        for env_idx in range(self.num_envs):
            self.relocate_stale_objects(env_idx)
            if self._tables[env_idx] is not None:
                self._tables[env_idx].relocate_offscreen()

    def apply_saved_poses(self, env_idx_list: Sequence[int]):
        """Restore selected environments to their saved evaluation poses.

        Tables settle first so support surfaces are in place before the
        remaining scene objects are restored as one batch.
        """
        object_groups = []
        for env_idx in range(self.num_envs):
            active_names = self.layout_manager.get_active_instance_names(env_idx)
            object_groups.append(
                [
                    {name: obj for name, obj in group.items() if name in active_names}
                    for group in [
                        self._rigid_and_dynamic_objects[env_idx],
                        self._articulation_objects[env_idx],
                        self._garment_objects[env_idx],
                        self._geometry_objects[env_idx],
                        self._fluid_objects[env_idx],
                    ]
                ]
            )
        restore_saved_poses(env_idx_list, self._tables, object_groups, self.sim)
        for env_idx in env_idx_list:
            self.validate_active_object_poses(env_idx)

    def validate_active_object_poses(self, env_idx: int):
        """Reject missing, non-finite, or offscreen active reward identities."""
        for object_type in ["Rigid", "Dynamic", "Geometry", "Articulation", "Garment"]:
            for record in self.layout_manager.get_layout_records(env_idx, object_type):
                inst_name = record["inst_name"]
                pos, _ = self.layout_manager.get_instance_pose(env_idx=env_idx, inst_name=inst_name)
                validate_active_pose(inst_name, pos)

    def restore_active_articulations(self, env_idx_list: Sequence[int]):
        """Reassert saved roots and joint values at the rollout boundary."""
        for env_idx in env_idx_list:
            active_names = self.layout_manager.get_active_instance_names(env_idx)
            for inst_name, obj in self._articulation_objects[env_idx].items():
                if inst_name in active_names:
                    obj.apply_saved_pose()
            self.validate_active_object_poses(env_idx)

    def reload_env_scene(self, env_id: int, object_types: List[str] = None):
        """Reload scene objects for the specified environment.
        Resets lights, room, and recreates objects.

        Args:
            env_id: ID of the environment to reset
        """
        if not self.setup_scene:
            self.config[f"env{env_id}"] = self.layout_manager.load_saved_layout(env_id)
        self.reload_lights(env_id)
        self.reload_ground(env_id)

        self.reload_room(env_id)
        self.reload_table(env_id)

        if self.device.type == "cuda":
            self.relocate_stale_objects(env_id, ["rigid", "articulation"])
            preserved_types = ["rigid", "articulation"]
            if self.setup_scene:
                preserved_types.append("geometry")
            # Clear existing objects (based on exclude_types)
            self.clear_scene_objects(env_id, exclude_types=preserved_types)
            self.reconcile_preserved_scene_objects(env_id, preserved_types)
            self.spawn_scene_objects(env_id, exclude_types=preserved_types)
        else:
            self.relocate_stale_objects(env_id, ["articulation"])
            preserved_types = ["articulation"]
            if self.setup_scene:
                preserved_types.append("geometry")
            # Clear existing objects (based on exclude_types)
            self.clear_scene_objects(env_id, exclude_types=preserved_types)
            self.reconcile_preserved_scene_objects(env_id, preserved_types)
            self.spawn_scene_objects(env_id, exclude_types=preserved_types)

    def _setup_background(self):
        """Set up background for all environments."""
        # Setup shared background
        self.background_config = self.layout_manager.select_background()
        if self.background_config is not None:
            self._background = Background(self.background_prim_path, self.background_config)

    def reload_background(self):
        """Reset background properties for all environments."""
        self.background_config = self.layout_manager.select_background()
        if self.background_config is not None:
            self._background.inst_config = deepcopy(self.background_config)
            self._background.reset()

    def reload_lights(self, env_id: int):
        """Reset lights for the specified environment.

        Args:
            env_id: Environment ID
        """
        # Check if light configuration exists
        env_config = self.config.get(f"env{env_id}")
        if env_config is None or env_config.get("Light") is None:
            return
        light_cfg = self.config[f"env{env_id}"].get("Light")
        if light_cfg is None:
            return

        self.delete_fixture_prims(env_id, "Light")

        # Clear light references and recreate
        self._lights[env_id].clear()
        for light in light_cfg.values():
            light_type = light["types"]
            light_prim_path = light["prim_path"]
            self._lights[env_id].append(Light(prim_path=light_prim_path, light_type=light_type, config=light))

    def reload_ground(self, env_id: int):
        """Reset ground for the specified environment.

        Args:
            env_id: Environment ID
        """
        # Check if ground configuration exists
        env_config = self.config.get(f"env{env_id}")
        if env_config is None or env_config.get("Ground") is None:
            return
        ground_cfg = self.config[f"env{env_id}"].get("Ground")
        if ground_cfg is None:
            return

        self.delete_fixture_prims(env_id, "Ground")

        self._grounds[env_id].clear()
        self._grounds[env_id].append(
            Ground(
                prim_path=ground_cfg["prim_path"],
                config=ground_cfg,
                env_spacing=self.env_spacing,
            )
        )

    def reload_room(self, env_id: int):
        """Reset room for the specified environment.

        Args:
            env_id: Environment ID
        """
        env_config = self.config.get(f"env{env_id}")
        if env_config is None or env_config.get("Room") is None:
            return
        room_cfg = self.config[f"env{env_id}"].get("Room")
        if room_cfg is None:
            return

        self.delete_fixture_prims(env_id, "Room")

        self._rooms[env_id] = Room(
            prim_path=room_cfg["prim_path"],
            usd_path=room_cfg["usd_path"],
            instance_config=room_cfg,
        )

    def reload_table(self, env_id: int):
        """Reset table for the specified environment.

        Args:
            env_id: Environment ID
        """
        env_config = self.config.get(f"env{env_id}")
        if env_config is None or env_config.get("Table") is None:
            return
        table_cfg = self.config[f"env{env_id}"].get("Table")
        if table_cfg is None:
            return

        self.delete_fixture_prims(env_id, "Table")

        self._tables[env_id] = Table(
            prim_path=table_cfg["prim_path"],
            mdl_file_path=table_cfg["mdl_path"],
            instance_config=table_cfg,
            mdl_name=table_cfg["mdl_name"],
        )

    def delete_fixture_prims(self, env_id: int, types: str):
        env_root = self.env_roots[env_id]
        if types not in ["Table", "Room", "Light", "Ground"]:
            return
        prim_path = f"{env_root}/{types}"
        if is_prim_path_valid(prim_path):
            delete_prim(prim_path)

    def close(self):
        self.layout_manager.close()
        for env_id in range(self.num_envs):
            self.clear_scene_objects(env_id)

            self._lights[env_id].clear()
            self._rooms[env_id] = None
            self._tables[env_id] = None
            self._grounds[env_id].clear()

        self._background = None

    def clear_scene_objects(self, env_id: int, exclude_types: List[str] | None = None):
        """Clear all existing objects in the specified environment.

        Args:
            env_id: Environment ID
            exclude_types: Object types to exclude from clearing
        """
        exclude_types = exclude_types or []

        # Determine which object collections to clear based on exclude_types
        # Relocate articulations before deleting rigid prims on CUDA. Deleting
        # any prim invalidates the shared PhysX tensor view; articulations need
        # that view to restore joint state while moving offscreen.
        all_collections = [
            (self._articulation_objects[env_id], "articulation"),
            (self._rigid_and_dynamic_objects[env_id], "rigid"),
            (self._garment_objects[env_id], "garment"),
            (self._geometry_objects[env_id], "geometry"),
            (self._fluid_objects[env_id], "fluid"),
        ]

        collections_to_clear = [obj_dict for obj_dict, obj_type in all_collections if obj_type not in exclude_types]

        for obj_dict in collections_to_clear:
            for obj_key, obj in list(obj_dict.items()):
                self.delete_scene_object(obj)
                del obj_dict[obj_key]

    def relocate_stale_objects(self, env_id: int, obj_types: List[str] = None):
        """Relocate stale scene objects offscreen and clear their velocities without deleting prims.

        Args:
            env_id: Environment ID
            obj_types: Object types to relocate offscreen
        """
        type_mapping = {
            "rigid": self._rigid_and_dynamic_objects,
            "articulation": self._articulation_objects,
            "geometry": self._geometry_objects,
        }
        if obj_types is None:
            obj_types = type_mapping.keys()
        for obj_type, obj_collection in type_mapping.items():
            if obj_type in obj_types and env_id < len(obj_collection):
                for obj in obj_collection[env_id].values():
                    obj.relocate_offscreen()

    def delete_scene_object(self, obj: Any):
        """Delete object based on its type using appropriate method.

        Args:
            obj: Object instance to delete
        """
        if self.device.type == "cuda":
            if isinstance(obj, GarmentObject):
                if hasattr(obj, "usd_prim_path") and is_prim_path_valid(obj.usd_prim_path):
                    delete_prim(obj.usd_prim_path)
            elif isinstance(obj, FluidObject):
                if hasattr(obj, "usd_prim_path") and is_prim_path_valid(obj.usd_prim_path):
                    delete_prim(obj.usd_prim_path)
                # Only delete inline Fluid containers; keep config-imported containers
                if (
                    getattr(obj, "container_owned", False)
                    and hasattr(obj, "container_prim_path")
                    and obj.container_prim_path
                    and is_prim_path_valid(obj.container_prim_path)
                ):
                    delete_prim(obj.container_prim_path)
            elif isinstance(obj, ArticulationObject):
                obj.relocate_offscreen()
            else:
                if hasattr(obj, "prim_path") and is_prim_path_valid(obj.prim_path):
                    delete_prim(obj.prim_path)
        else:
            if isinstance(obj, RigidObject) or isinstance(obj, GeometryObject):
                obj.destroy()
            elif isinstance(obj, GarmentObject):
                if hasattr(obj, "usd_prim_path") and is_prim_path_valid(obj.usd_prim_path):
                    delete_prim(obj.usd_prim_path)
            elif isinstance(obj, FluidObject):
                if hasattr(obj, "usd_prim_path") and is_prim_path_valid(obj.usd_prim_path):
                    delete_prim(obj.usd_prim_path)
                if (
                    getattr(obj, "container_owned", False)
                    and hasattr(obj, "container_prim_path")
                    and obj.container_prim_path
                    and is_prim_path_valid(obj.container_prim_path)
                ):
                    delete_prim(obj.container_prim_path)
            elif isinstance(obj, ArticulationObject):
                obj.relocate_offscreen()
            else:
                if hasattr(obj, "prim_path") and is_prim_path_valid(obj.prim_path):
                    delete_prim(obj.prim_path)
