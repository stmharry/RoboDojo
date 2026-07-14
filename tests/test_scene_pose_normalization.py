import importlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from robodojo.sim.environment.reward_manager.func_parser import Func_Parser


def _import_layout_manager_without_starting_kit():
    module_names = ("isaacsim", "isaacsim.core", "isaacsim.core.utils", "isaacsim.core.utils.prims")
    prior_modules = {name: sys.modules.get(name) for name in module_names}
    isaacsim = ModuleType("isaacsim")
    isaacsim.__path__ = []
    core = ModuleType("isaacsim.core")
    core.__path__ = []
    utils = ModuleType("isaacsim.core.utils")
    utils.__path__ = []
    prims = ModuleType("isaacsim.core.utils.prims")
    prims.is_prim_path_valid = lambda _: True
    sys.modules.update(
        {
            "isaacsim": isaacsim,
            "isaacsim.core": core,
            "isaacsim.core.utils": utils,
            "isaacsim.core.utils.prims": prims,
        }
    )
    try:
        return importlib.import_module("robodojo.sim.environment.scene_manager.layout_manager").LayoutManager
    finally:
        for name, module in prior_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


LayoutManager = _import_layout_manager_without_starting_kit()


class _TensorPoseObject:
    def __init__(self, pos: torch.Tensor, rot: torch.Tensor):
        self.pos = pos
        self.rot = rot

    def _get_object_transform(self):
        return self.pos, self.rot, self.pos.device

    def get_all_joints_info(self):
        return {}


class _StatePoseObject:
    def __init__(self, root_pose: torch.Tensor):
        self.root_pose = root_pose

    def get_state(self, is_relative=True):
        assert is_relative is True
        return {"root_pose": self.root_pose}


def _manager(instance_type, scene_object, *, env_origin=None):
    manager = LayoutManager.__new__(LayoutManager)
    manager.instance_type_by_env = [{"target": instance_type}]
    manager.get_scene_object = lambda env_idx, inst_name: scene_object
    if env_origin is not None:
        manager.scene_manager = SimpleNamespace(env_origins=[env_origin])
    return manager


@pytest.mark.parametrize("instance_type", ["rigid", "articulation"])
def test_get_instance_pose_normalizes_tensor_objects_to_host_numpy(instance_type):
    pos = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    rot = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
    manager = _manager(instance_type, _TensorPoseObject(pos, rot))

    actual_pos, actual_rot = manager.get_instance_pose(env_idx=0, inst_name="target")

    np.testing.assert_array_equal(actual_pos, pos.detach().numpy())
    np.testing.assert_array_equal(actual_rot, rot.detach().numpy())
    assert isinstance(actual_pos, np.ndarray)
    assert isinstance(actual_rot, np.ndarray)


def test_get_instance_pose_applies_world_origin_before_host_conversion():
    pos = torch.tensor([1.0, 2.0, 3.0])
    rot = torch.tensor([1.0, 0.0, 0.0, 0.0])
    env_origin = torch.tensor([10.0, 20.0, 30.0])
    manager = _manager("rigid", _TensorPoseObject(pos, rot), env_origin=env_origin)

    actual_pos, actual_rot = manager.get_instance_pose(env_idx=0, inst_name="target", relative=False)

    np.testing.assert_array_equal(actual_pos, np.array([11.0, 22.0, 33.0]))
    np.testing.assert_array_equal(actual_rot, np.array([1.0, 0.0, 0.0, 0.0]))


@pytest.mark.parametrize("instance_type", ["garment", "geometry"])
def test_get_instance_pose_preserves_state_object_pose_semantics(instance_type):
    root_pose = torch.tensor([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    manager = _manager(instance_type, _StatePoseObject(root_pose))

    actual_pos, actual_rot = manager.get_instance_pose(env_idx=0, inst_name="target")

    np.testing.assert_array_equal(actual_pos, root_pose[:3].numpy())
    np.testing.assert_array_equal(actual_rot, root_pose[3:].numpy())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the regression case")
@pytest.mark.parametrize("instance_type", ["rigid", "articulation"])
def test_reward_state_initialization_accepts_cuda_scene_poses(instance_type):
    pos = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda")
    manager = _manager(instance_type, _TensorPoseObject(pos, rot))
    manager.get_layout_records = lambda env_idx, object_type: (
        [{"inst_name": "target"}] if object_type.lower() == instance_type else []
    )
    env = SimpleNamespace(
        success=[True],
        scene_manager=SimpleNamespace(layout_manager=manager),
        robot_manager=SimpleNamespace(robot_list=[]),
    )
    parser = Func_Parser(num_envs=1)
    parser.initialize(env)

    parser.init_state()

    pose = parser.pre_state[0]["target"]["pose"]
    np.testing.assert_array_equal(pose, np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]))
    assert isinstance(pose, np.ndarray)
