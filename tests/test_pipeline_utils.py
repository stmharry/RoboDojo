from omegaconf import OmegaConf
import pytest

from robodojo.sim.utils.pipeline_utils import configure_task_physics_device


@pytest.mark.parametrize("section", ["Garment", "Fluid"])
def test_cpu_only_task_objects_disable_cuda_and_fabric(section):
    config = OmegaConf.create(
        {
            "sim": {"device": "cuda:0", "use_fabric": True},
            "task_env": {section: [{"category": []}]},
        }
    )

    processed = configure_task_physics_device(config)

    assert processed.sim.device == "cpu"
    assert processed.sim.use_fabric is False


def test_gpu_compatible_task_objects_keep_selected_device():
    config = OmegaConf.create(
        {
            "sim": {"device": "cuda:0", "use_fabric": True},
            "task_env": {"Rigid": [{"category": []}]},
        }
    )

    processed = configure_task_physics_device(config)

    assert processed.sim.device == "cuda:0"
    assert processed.sim.use_fabric is True
