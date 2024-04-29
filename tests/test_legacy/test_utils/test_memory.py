import pytest

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.legacy.utils.memory import colo_device_memory_capacity, colo_set_process_memory_fraction
from colossalai.testing import spawn


def _run_colo_set_process_memory_fraction_and_colo_device_memory_capacity():
    frac1 = colo_device_memory_capacity(get_accelerator().get_current_device())
    colo_set_process_memory_fraction(0.5)
    frac2 = colo_device_memory_capacity(get_accelerator().get_current_device())
    assert frac2 * 2 == frac1


def run_dist(rank, world_size, port):
    colossalai.legacy.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    _run_colo_set_process_memory_fraction_and_colo_device_memory_capacity()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [3, 4])
def test_memory_utils(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_memory_utils(world_size=2)
