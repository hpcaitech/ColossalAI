import pytest

from colossalai.utils.cuda import get_current_device
from colossalai.utils.memory_utils.utils import colo_tensor_mem_usage, colo_model_data_tensor_move, colo_model_data_tensor_move_inline, colo_set_process_memory_fraction, colo_cuda_memory_capacity
from colossalai.utils import free_port
from colossalai.zero.sharded_param.tensorful_state import StatefulTensor
import colossalai

import torch

from functools import partial
import torch.multiprocessing as mp

def _run_colo_tensor_mem_usage():
    for t in [StatefulTensor(torch.randn(2, 3)), torch.randn(2, 3)]:
        cuda_use, cpu_use = colo_tensor_mem_usage(t)
        print("cuda_use")
        print(cuda_use)
        print("cput_use")
        print(cpu_use)

def _run_colo_set_process_memory_fraction():
    colo_set_process_memory_fraction(0.5)

def _run_colo_cuda_memory_capacity():
    frac = colo_cuda_memory_capacity
    print("cuda memory capacity of the current cuda")
    print(frac)


def _run_colo_model_data_tensor_move_inline():
    for t in [StatefulTensor(torch.randn(2,3)), torch.randn(2,3)]:
        colo_model_data_tensor_move_inline(t, torch.device("cuda:3"))
        assert t.device == torch.device("cuda:3")


def _run_colo_model_data_tensor_move():
    for t in [(StatefulTensor(torch.ones(2, 3)), StatefulTensor(torch.zeros(2, 3).cuda(get_current_device()))),
        (torch.ones(2, 3), torch.zeros(2, 3).cuda(get_current_device()))]:
        cpu_t, cuda_t = t
        colo_model_data_tensor_move(cpu_t, cuda_t)
        assert cuda_t.device == torch.device(f"cuda:{get_current_device()}")


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    _run_colo_set_process_memory_fraction()
    _run_colo_cuda_memory_capacity()
    _run_colo_model_data_tensor_move_inline()
    _run_colo_model_data_tensor_move()
    _run_colo_tensor_mem_usage()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 4])
def test_tensor_move(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_tensor_move(4)
