import logging
import re
from typing import Any, Dict, List

from isaacsim.core.utils.prims import is_prim_path_valid

logger = logging.getLogger(__name__)


class SceneQueriesService:
    def resolve_camera_fixture_mount(self, env_id: int, fixture_label: str, frame: str | None = None) -> str:
        """Resolve a scene-published fixture label to its current prim path."""
        matches = []
        records_by_type = self.layout_manager.object_records_by_type
        for object_type in ("Rigid", "Dynamic", "Geometry", "Articulation"):
            records = records_by_type[object_type].layout_records_by_env[env_id]
            matches.extend(record for record in records if record.get("label") == fixture_label)
        if len(matches) != 1:
            raise ValueError(f"camera fixture label {fixture_label!r} resolved to {len(matches)} instances")
        prim_path = matches[0].get("prim_path")
        if not prim_path:
            raise ValueError(f"camera fixture {fixture_label!r} has no prim path")
        if frame:
            parts = frame.split("/")
            if frame.startswith("/") or any(
                not part or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part) for part in parts
            ):
                raise ValueError(f"invalid camera fixture frame: {frame!r}")
            prim_path = f"{prim_path}/{frame}"
            if not is_prim_path_valid(prim_path):
                raise ValueError(f"camera fixture frame {frame!r} does not exist below fixture {fixture_label!r}")
        return prim_path

    def is_global_config_key(self, key: str) -> bool:
        """Check if the key is a global configuration key.

        Args:
            key: Configuration key to check

        Returns:
            True if it's a global config key, False otherwise
        """
        global_keys = [
            "common",
            "deformable_material",
            "visual_material",
            "deformable_config",
            "particle_system",
            "particle_material",
            "garment_config",
            "fem_config",
        ]
        return key in global_keys

    def get_objects(
        self,
        env_ids: List[int] | None = None,
        object_name: str | None = None,
        object_type: str | None = None,
    ) -> Dict[str, Any]:
        """Get object instances by environment IDs, object name, and object type.

        Args:
            env_ids: List of environment IDs to query. If None, query all environments
            object_name: Object key to retrieve. If None, return all objects
            object_type: Type of objects to retrieve (rigid, articulation, etc.). If None, return all types

        Returns:
            Dictionary with object collections organized by composite keys (env_{id}_{type}_{object_key})
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        result = {}
        obj_collections = self._get_object_collections_by_type(object_type)
        for collection_type, env_dict_list in obj_collections.items():
            for env_id in env_ids:
                if env_id >= len(env_dict_list):
                    continue

                env_dict = env_dict_list[env_id]
                for obj_key, obj in env_dict.items():
                    if object_name is None or obj_key == object_name:
                        key = f"env{env_id}_{collection_type}_{obj_key}"
                        result[key] = obj

        return result

    def _get_object_collections_by_type(self, object_type: str | None) -> Dict[str, List[Dict[str, Any]]]:
        """Get object collections filtered by type.

        Args:
            object_type: Type filter for objects

        Returns:
            Dictionary of object collections
        """
        all_collections = {
            "rigid": self._rigid_and_dynamic_objects,
            "dynamic": self._rigid_and_dynamic_objects,
            "articulation": self._articulation_objects,
            "garment": self._garment_objects,
            "geometry": self._geometry_objects,
            "fluid": self._fluid_objects,
        }

        if object_type is None:
            return all_collections
        elif object_type in all_collections:
            return {object_type: all_collections[object_type]}
        else:
            return {}
