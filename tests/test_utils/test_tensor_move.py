import pytest

from colossalai.utils.cuda import get_current_device
from colossalai.utils.memory_utils.utils import colo_model_data_tensor_move, colo_model_data_tensor_move_inline
from colossalai.utils import free_port
from colossalai.zero.sharded_param import ShardedTensor
import colossalai

import torch

from functools import partial
import torch.multiprocessing as mp


def _run_colo_model_data_tensor_move_inline():
    for t in [torch.randn(2, 3), ShardedTensor(torch.randn(2, 3))]:
        colo_model_data_tensor_move_inline(t, torch.device(f"cuda:{get_current_device()}"))
        assert t.device == torch.device(f"cuda:{get_current_device()}")


def _run_colo_model_data_tensor_move():
    for t in [(torch.ones(2, 3), torch.zeros(2, 3).cuda(get_current_device())),
              (ShardedTensor(torch.ones(2, 3)), ShardedTensor(torch.zeros(2, 3).cuda(get_current_device())))]:
        cpu_t, cuda_t = t
        colo_model_data_tensor_move(cpu_t, cuda_t)


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    _run_colo_model_data_tensor_move_inline()
    _run_colo_model_data_tensor_move()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 4])
def test_tensor_move(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_tensor_move(4)
