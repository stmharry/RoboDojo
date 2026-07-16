from copy import deepcopy
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import transforms3d as t3d

from robodojo.sim.environment.description_manager.desc_manager import descriptions_from_metadata
from robodojo.sim.utils.transformer import _get_link_matrix_from_usd, pose_to_matrix

logger = logging.getLogger(__name__)


def _as_host_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class SpatialQueriesService:
    def get_instance_name(self, env_idx, label):
        types = ["Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid"]
        for t in types:
            for obj_cfg in self.object_records_by_type[t].layout_records_by_env[env_idx]:
                if obj_cfg.get("label", None) == label:
                    return obj_cfg["inst_name"]
        return None

    def get_scene_object(self, env_idx, inst_name):
        instance_type = self.instance_type_by_env[env_idx].get(inst_name, None)
        if instance_type is None:
            logger.warning("Instance name %s not found in env %s.", inst_name, env_idx)
            return None
        key = f"env{env_idx}_{instance_type}_{inst_name}"
        result = self.scene_manager.get_objects(env_ids=[env_idx], object_name=inst_name, object_type=instance_type)
        if len(result) == 0 or result.get(key) is None:
            logger.warning("Object %s of type %s not found in env %s.", inst_name, instance_type, env_idx)
            return None
        return result[key]

    def get_labels_by_prefix(self, env_idx, prefix):
        types = ["Rigid", "Geometry", "Articulation", "Garment", "Fluid"]
        label_list = []
        for t in types:
            for obj_cfg in self.object_records_by_type[t].layout_records_by_env[env_idx]:
                label = obj_cfg.get("label", None)
                if label is not None and label.startswith(prefix):
                    label_list.append(label)
        return label_list

    def get_instance_pose(self, env_idx: int, label: str = None, inst_name: str = None, relative: bool = True):
        """Return position and quaternion as host NumPy arrays.

        Official RoboDojo currently returns device tensors for rigid and
        articulated objects but host arrays for garments and geometry. Keep
        this fork's NumPy-based reward boundary consistent across object types.
        """
        if inst_name is None:
            inst_name = self.get_instance_name(env_idx, label)
        if inst_name is None:
            logger.warning("No instance found with label %s or inst_name %s in env %s.", label, inst_name, env_idx)
            return (None, None)
        instance_type = self.instance_type_by_env[env_idx].get(inst_name, None)
        obj = self.get_scene_object(env_idx, inst_name)
        if obj is None:
            return (None, None)
        if instance_type in ["rigid", "dynamic", "articulation"]:
            pos, rot, device = obj._get_object_transform()
            if not relative:
                env_pos = deepcopy(self.scene_manager.env_origins[env_idx]).to(device)
                pos = pos + env_pos
            return (_as_host_numpy(pos), _as_host_numpy(rot))
        elif instance_type in ["garment", "geometry"]:
            state = obj.get_state(is_relative=True)
            root_pose = _as_host_numpy(state["root_pose"])
            pos = root_pose[:3]
            rot = root_pose[3:]
            return (pos, rot)

    def get_label_descriptions(self, env_idx, label=None, inst_name=None):
        metadata = self.get_instance_metadata(env_idx=env_idx, label=label, inst_name=inst_name)
        if metadata is None:
            logger.warning(
                "No instance found with label %s or inst_name %s in env %s for getting description.",
                label,
                inst_name,
                env_idx,
            )
            return []
        return descriptions_from_metadata(metadata)

    def get_instance_bbox_vertices(self, inst_name, env_idx):
        obj = self.get_scene_object(env_idx, inst_name)
        if obj is None:
            logger.warning("Instance %s not found in env %s for getting bbox.", inst_name, env_idx)
            return None
        physics_type = self.instance_type_by_env[env_idx].get(inst_name, None)
        if physics_type:
            _data = deepcopy(
                self.object_records_by_type[physics_type.capitalize()].metadata_by_env[env_idx].get(inst_name, None)
            )
            if (
                _data is not None
                and "geometry" in _data
                and ("oriented_bbox" in _data["geometry"])
                and ("vertices" in _data["geometry"]["oriented_bbox"])
            ):
                return np.asarray(_data["geometry"]["oriented_bbox"]["vertices"], dtype=float).reshape(-1, 3)
        if getattr(obj, "get_bbox", None) is not None:
            return obj.get_bbox()
        return None

    def to_transformation_matrix(self, pos: torch.Tensor, rot: torch.Tensor) -> np.ndarray:
        """
        Convert position and quaternion to 4x4 transformation matrix (NumPy array).
        pos: (1,3)
        rot: (1,4) in (w,x,y,z)
        """
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().reshape(-1).numpy()
        if isinstance(rot, torch.Tensor):
            rot = rot.cpu().reshape(-1).numpy()
        w, x, y, z = rot
        R = np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
            ],
            dtype=np.float32,
        )
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = pos
        return T

    def get_functional_points(
        self,
        tag: str,
        type: Literal["active", "passive"],
        config: dict = None,
        ret: Literal["matrix", "list", "pose"] = "list",
        obj_name: str = None,
        env_idx=None,
    ) -> np.ndarray | list:
        """Get the transformation matrix of given functional point of the actor."""
        data = config.get(f"{type}", {})
        result = []
        for key, item in data.get("functional", {}).items():
            if key == tag:
                base_link = item.get("base_link", None)
                pose_list = item.get("frame", None)
                if base_link is not None:
                    inst = self.get_scene_object(env_idx=env_idx, inst_name=obj_name)
                    env_origin = deepcopy(self.scene_manager.env_origins[env_idx])
                    if isinstance(env_origin, torch.Tensor):
                        env_origin = env_origin.cpu().numpy()
                    link_pose = np.asarray(inst.get_link_pose(base_link), dtype=float).reshape(-1)
                    link_pose[:3] -= env_origin
                    actor_matrix = pose_to_matrix(link_pose)
                else:
                    pos, rot = self.get_instance_pose(inst_name=obj_name, env_idx=env_idx)
                    actor_matrix = self.to_transformation_matrix(pos, rot)
                if pose_list is not None:
                    for pose in pose_list:
                        local_matrix = pose_to_matrix(pose)
                        world_matrix = actor_matrix @ local_matrix
                        if ret == "matrix":
                            result.append(world_matrix)
                        elif ret == "list":
                            result.append(
                                list(world_matrix[:3, 3].tolist())
                                + list(t3d.quaternions.mat2quat(world_matrix[:3, :3]).tolist())
                            )
        return result

    def get_support_points(
        self,
        tag: str,
        type: Literal["active", "passive"],
        config: dict = None,
        ret: Literal["matrix", "list", "pose"] = "list",
        obj_name: str = None,
        env_idx=None,
        default_pos=None,
        default_ori=None,
        usd_path=None,
    ) -> np.ndarray | list:
        data = config.get(f"{type}", {})
        result_list = []
        radius = []
        for key, item in data.get("support", {}).items():
            if key == tag:
                base_link = item.get("base_link", None)
                center = item.get("center", [])
                radius = item.get("radius", [])
                if len(center) != len(radius):
                    return None
                if default_pos is None or default_ori is None:
                    default_pos, default_ori = self.get_instance_pose(inst_name=obj_name, env_idx=env_idx)
                if base_link is not None and Path(usd_path).exists():
                    root_matrix = self.to_transformation_matrix(default_pos, default_ori)
                    link_local_matrix = _get_link_matrix_from_usd(usd_path, base_link)
                    ref_matrix = root_matrix @ link_local_matrix
                else:
                    ref_matrix = self.to_transformation_matrix(default_pos, default_ori)
                for c in center:
                    world_matrix = ref_matrix @ pose_to_matrix(c)
                    if ret == "matrix":
                        result_list.append(world_matrix)
                    elif ret == "list":
                        result_list.append(
                            list(world_matrix[:3, 3].tolist())
                            + list(t3d.quaternions.mat2quat(world_matrix[:3, :3]).tolist())
                        )
        return (result_list, radius)
