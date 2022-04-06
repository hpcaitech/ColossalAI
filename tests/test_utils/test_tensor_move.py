import pytest

from colossalai.utils.cuda import get_current_device
from colossalai.zero.shard_utils.tensor_utils import colo_tensor_mem_usage, colo_model_data_tensor_move, colo_model_data_tensor_move_inline, colo_model_data_move_to_cpu, colo_model_tensor_clone
from colossalai.utils.memory_utils.utils import colo_set_process_memory_fraction, colo_cuda_memory_capacity
from colossalai.utils import free_port
from colossalai.zero.sharded_param.tensorful_state import StatefulTensor
import colossalai

import torch

from functools import partial
import torch.multiprocessing as mp

def _run_colo_tensor_mem_usage():
    for i in range(1):
        if i == 1:
            t1 = StatefulTensor(torch.randn(2,2))
            t2 = StatefulTensor(torch.randn(4,4))
            c1 , g1 = colo_tensor_mem_usage(t1)
            c2 , g2 = colo_tensor_mem_usage(t2)
            assert c1*4 == c2
            assert g1*4 == g2
        else:
            t1 = torch.randn(2,2)
            t2 = torch.randn(4,4)
            c1 , g1 = colo_tensor_mem_usage(t1)
            c2 , g2 = colo_tensor_mem_usage(t2)
            assert c1*4 == c2
            assert g1*4 == g2

def _run_colo_set_process_memory_fraction_and_colo_cuda_memory_capacity():
    frac1 = colo_cuda_memory_capacity()
    colo_set_process_memory_fraction(0.5)
    frac2 = colo_cuda_memory_capacity()
    assert frac2*2 == frac1

def _run_colo_model_data_tensor_move_inline():
    for t in [StatefulTensor(torch.randn(2,3)), torch.randn(2,3)]:
        colo_model_data_tensor_move_inline(t, torch.device(f"cuda:{get_current_device()}"))
        assert t.device == torch.device(f"cuda:{get_current_device()}")

def _run_colo_model_data_tensor_move():
    for t in [(StatefulTensor(torch.ones(2, 3)), StatefulTensor(torch.zeros(2, 3).cuda(get_current_device()))),
        (torch.ones(2, 3), torch.zeros(2, 3).cuda(get_current_device()))]:
        cpu_t, cuda_t = t
        colo_model_data_tensor_move(cpu_t, cuda_t)
        assert cuda_t.device == torch.device(f"cuda:{get_current_device()}")

def _run_colo_model_data_move_to_cpu():
    for t in [StatefulTensor(torch.randn(2,2)), torch.randn(4,4)]:
        colo_model_data_move_to_cpu(t)
        assert t.device == torch.device("cpu")

def _run_colo_model_tensor_clone():
    for t in [StatefulTensor(torch.randn(2,2).cuda(torch.cuda.current_device())), torch.randn(4,4).cuda(torch.cuda.current_device())]:
        if issubclass(type(t), StatefulTensor):
            assert t.payload.device == torch.device(f"cuda:{get_current_device()}")
        else:
            assert t.device == torch.device(f"cuda:{get_current_device()}")
        p = colo_model_tensor_clone(t, torch.device(f"cuda:{get_current_device()}"))
        assert p.device == torch.device(f"cuda:{get_current_device()}")
        for i in range(2):
            for j in range(2):
                if issubclass(type(t), StatefulTensor):
                    assert t.payload.device == p.device
                    assert t.payload[i][j] == p[i][j]
                else:
                    assert t.device == p.device
                    assert t[i][j] == p[i][j]



def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    _run_colo_set_process_memory_fraction_and_colo_cuda_memory_capacity()
    _run_colo_model_data_tensor_move_inline()
    _run_colo_model_data_tensor_move()
    _run_colo_tensor_mem_usage()
    _run_colo_model_data_move_to_cpu()
    _run_colo_model_tensor_clone()

@pytest.mark.dist
@pytest.mark.parametrize("world_size", [4, 5])
def test_tensor_move(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)

if __name__ == '__main__':
    test_tensor_move(4)
