import logging
from typing import Any, Dict, List

from pxr import Gf

from robodojo.sim.environment.scene_manager.objects.articulation import ArticulationObject
from robodojo.sim.environment.scene_manager.objects.dynamic import DynamicObject
from robodojo.sim.environment.scene_manager.objects.fluid import FluidObject
from robodojo.sim.environment.scene_manager.objects.garment import GarmentObject
from robodojo.sim.environment.scene_manager.objects.geometry import GeometryObject
from robodojo.sim.environment.scene_manager.objects.primitives import PRIMITIVE_MAP
from robodojo.sim.environment.scene_manager.objects.rigid import RigidObject
from robodojo.sim.utils.path import deep_resolve_paths

logger = logging.getLogger(__name__)


class ObjectFactoryService:
    def spawn_category_objects(
        self,
        env_id: int,
        cat_name: str,
        cat_cfg: List[Dict[str, Any]],
        exclude_types: List[str] | None = None,
    ):
        """Create objects for a specific category in an environment.

        Args:
            env_id: Environment ID
            cat_name: Category name
            cat_spec: Category configuration
            exclude_types: Object types to exclude from creation
        """
        exclude_types = exclude_types or []
        num_per_env = len(cat_cfg)
        for i in range(num_per_env):
            inst_cfg = cat_cfg[i]
            deep_resolve_paths(inst_cfg)
            asset_to_spawn = inst_cfg.get("usd_path", None)
            inst_name = inst_cfg.get("inst_name", None)
            prim_path = inst_cfg.get("prim_path", None)
            obj_type = inst_cfg.get("physics", {}).get("type", "rigid")

            if obj_type in exclude_types:
                continue

            default_pos = inst_cfg.get("default_pos", (0.0, 0.0, 0.0))
            default_ori = inst_cfg.get("default_ori", (1.0, 0.0, 0.0, 0.0))
            scale = inst_cfg.get("scale", (1, 1, 1))

            obj = self.create_scene_object(
                env_id,
                asset_to_spawn,
                prim_path,
                inst_cfg,
                default_pos,
                default_ori,
                scale,
            )
            if obj is not None:
                self.register_scene_object(inst_name, env_id, obj)
                self.pending_initialization.append(obj)

    def create_scene_object(
        self,
        env_id: int,
        asset_to_spawn: str,
        prim_path: str,
        inst_cfg: Dict,
        default_pos: tuple,
        default_ori: tuple,
        scale: tuple,
    ) -> Any:
        """Create a single object instance.

        Args:
            env_id: Environment ID
            asset_to_spawn: Asset to spawn (USD path or primitive name)
            prim_path: Prim path for the object
            inst_cfg: Instance configuration
            default_pos: Default position for the object
            default_ori: Default orientation for the object

        Returns:
            Created object instance or None
        """
        # Get object type first to check if it should be disabled
        obj_type = inst_cfg.get("physics", {}).get("type", "rigid")
        # Check if garment is disabled due to cpu+use_fabric combination
        # Do this BEFORE creating any primitive shapes to avoid creating orphaned primitives
        if self.device.type == "cpu" and self.use_fabric:
            if obj_type in ["garment", "articulation"]:
                logger.warning(
                    "%s objects are disabled when device='cpu' and use_fabric=True. Skipping object creation at '%s'.",
                    obj_type.capitalize(),
                    prim_path,
                )
                return None
        if self.device.type.startswith("cuda"):
            if obj_type in ["fluid"]:
                logger.warning(
                    "%s objects are disabled when using CUDA device. Skipping object creation at '%s'.",
                    obj_type.capitalize(),
                    prim_path,
                )
                return None

        usd_path = None
        # Handle primitive shapes
        if asset_to_spawn in PRIMITIVE_MAP:
            self.create_primitive_shape(asset_to_spawn, prim_path, inst_cfg, default_pos, default_ori)
        else:
            usd_path = asset_to_spawn

        primitive_type = asset_to_spawn if asset_to_spawn in PRIMITIVE_MAP else None

        return self.instantiate_object_wrapper(
            obj_type,
            prim_path,
            usd_path,
            primitive_type,
            env_id,
            default_pos,
            default_ori,
            scale,
            inst_cfg,
        )

    def register_scene_object(self, obj_key: str, env_id: int, obj: Any):
        """Assign object to appropriate collection.

        Args:
            obj_key: Object key (typically instance name)
            env_id: Environment ID
            obj: Object instance to assign
        """
        # Store in hierarchy: environment -> object key -> object
        if isinstance(obj, RigidObject):
            self._rigid_and_dynamic_objects[env_id][obj_key] = obj
        elif isinstance(obj, DynamicObject):
            self._rigid_and_dynamic_objects[env_id][obj_key] = obj
        elif isinstance(obj, ArticulationObject):
            self._articulation_objects[env_id][obj_key] = obj
        elif isinstance(obj, GarmentObject):
            self._garment_objects[env_id][obj_key] = obj
        elif isinstance(obj, GeometryObject):
            self._geometry_objects[env_id][obj_key] = obj
        elif isinstance(obj, FluidObject):
            self._fluid_objects[env_id][obj_key] = obj

    def create_primitive_shape(
        self,
        primitive_name: str,
        prim_path: str,
        inst_cfg: Dict,
        default_pos: tuple,
        default_ori: tuple,
    ):
        """Create a primitive shape on the stage.

        Args:
            primitive_name: Name of the primitive shape
            prim_path: Prim path for the primitive
            inst_cfg: Instance configuration
            default_pos: Default position for the primitive
            default_ori: Default orientation for the primitive
        """
        primitive_class = PRIMITIVE_MAP[primitive_name]
        primitive_params_cfg = inst_cfg.get(primitive_name, {})
        params = {k: v for k, v in primitive_params_cfg.items()}

        # Use position from LayoutManager
        translation = default_pos
        params["position"] = Gf.Vec3f(float(translation[0]), float(translation[1]), float(translation[2]))
        primitive_class(prim_path=prim_path, **params)

    def instantiate_object_wrapper(
        self,
        obj_type: str,
        prim_path: str,
        usd_path: str | None,
        primitive_type: str | None,
        env_id: int,
        default_pos: tuple,
        default_ori: tuple,
        scale: tuple,
        inst_cfg: Dict,
    ) -> Any:
        """Create object based on its type.

        Args:
            obj_type: Object type (rigid, dynamic, geometry, garment, fluid, articulation)
            prim_path: Prim path for the object
            usd_path: USD file path (if applicable)
            primitive_type: Primitive type name (if applicable)
            env_id: Environment ID
            default_pos: Default position for the object
            default_ori: Default orientation for the object

        Returns:
            Created object instance
        """
        base_args = {
            "prim_path": prim_path,
            "usd_path": usd_path,
            "primitive_type": primitive_type,
            "default_pos": default_pos,
            "default_ori": default_ori,
            "scale": scale,
            "inst_config": inst_cfg,
        }
        if obj_type == "rigid":
            return RigidObject(
                env_origin=self.sim.scene.env_origins[env_id],
                **{k: v for k, v in base_args.items() if k != "primitive_type"},  # RigidObject does not use inst_config
            )
        elif obj_type == "dynamic":
            return DynamicObject(
                env_origin=self.sim.scene.env_origins[env_id],
                **{k: v for k, v in base_args.items() if k != "primitive_type"},
            )
        elif obj_type == "geometry":
            return GeometryObject(
                env_origin=self.sim.scene.env_origins[env_id],
                **{k: v for k, v in base_args.items()},
            )
        elif obj_type == "garment":
            return GarmentObject(
                env_origin=self.sim.scene.env_origins[env_id],
                **{k: v for k, v in base_args.items()},
            )
        elif obj_type == "fluid":
            return FluidObject(
                env_origin=self.sim.scene.env_origins[env_id],
                **{k: v for k, v in base_args.items() if k != "primitive_type"},
            )
        elif obj_type == "articulation":
            return ArticulationObject(
                env_origin=self.sim.scene.env_origins[env_id],
                **{k: v for k, v in base_args.items() if k != "primitive_type"},
            )
        else:
            raise ValueError(f"Invalid object type: {obj_type}")

    def spawn_scene_objects(self, env_id: int, exclude_types: List[str] | None = None):
        """Reset objects for the specified environment.

        Args:
            env_id: Environment ID
            exclude_types: Object types to exclude from reset
        """
        if self.config == {} or self.config is None:
            return
        objects_cfg = self.config.get(f"env{env_id}", None)
        if objects_cfg is None:
            return
        exclude_types = exclude_types or []
        # Create new objects for this environment
        for physics_type in objects_cfg:
            if physics_type in ["Light", "Room", "Ground", "Background", "Table"]:
                continue
            for cat_name in objects_cfg[physics_type]:
                if self.is_global_config_key(cat_name):
                    continue
                self.spawn_category_objects(env_id, cat_name, objects_cfg[physics_type][cat_name], exclude_types)
