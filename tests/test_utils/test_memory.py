import pytest

import colossalai
from colossalai.utils.cuda import get_current_device
from colossalai.utils.memory import colo_set_process_memory_fraction, colo_device_memory_capacity
from colossalai.utils import free_port

from functools import partial
import torch.multiprocessing as mp


def _run_colo_set_process_memory_fraction_and_colo_device_memory_capacity():
    frac1 = colo_device_memory_capacity(get_current_device())
    colo_set_process_memory_fraction(0.5)
    frac2 = colo_device_memory_capacity(get_current_device())
    assert frac2 * 2 == frac1


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    _run_colo_set_process_memory_fraction_and_colo_device_memory_capacity()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [3, 4])
def test_memory_utils(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_memory_utils(world_size=2)
