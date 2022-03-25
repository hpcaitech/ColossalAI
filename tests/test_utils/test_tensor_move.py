import pytest

from colossalai.utils.cuda import get_current_device
from colossalai.utils.memory_tracer.model_data_memtracer import GLOBAL_MODEL_DATA_TRACER
from colossalai.utils.memory_utils.utils import colo_model_data_tensor_move, colo_model_data_tensor_move_inline
from colossalai.zero.sharded_param import ShardedTensor

import colossalai

import torch

from functools import partial
import torch.multiprocessing as mp
from colossalai.utils import free_port


def _run_colo_model_data_tensor_move_inline():
    assert (GLOBAL_MODEL_DATA_TRACER.cuda_usage == 0)
    GLOBAL_MODEL_DATA_TRACER.start()

    for t in [torch.randn(2, 3), ShardedTensor(torch.randn(2, 3))]:
        GLOBAL_MODEL_DATA_TRACER.add_tensor(t)
        assert GLOBAL_MODEL_DATA_TRACER.cpu_usage == 2 * 3 * 4
        assert GLOBAL_MODEL_DATA_TRACER.cuda_usage == 0
        colo_model_data_tensor_move_inline(t, torch.device(f"cuda:{get_current_device()}"))
        assert t.device == torch.device(f"cuda:{get_current_device()}")
        assert GLOBAL_MODEL_DATA_TRACER.cpu_usage == 0
        assert GLOBAL_MODEL_DATA_TRACER.cuda_usage == 2 * 3 * 4
        GLOBAL_MODEL_DATA_TRACER.clear()

    GLOBAL_MODEL_DATA_TRACER.close()


def _run_colo_model_data_tensor_move():
    assert (GLOBAL_MODEL_DATA_TRACER.cuda_usage == 0)
    GLOBAL_MODEL_DATA_TRACER.start()

    for t in [(torch.ones(2, 3), torch.zeros(2, 3).cuda(get_current_device())),
              (ShardedTensor(torch.ones(2, 3)), ShardedTensor(torch.zeros(2, 3).cuda(get_current_device())))]:
        cpu_t, cuda_t = t
        GLOBAL_MODEL_DATA_TRACER.add_tensor(cpu_t)
        assert GLOBAL_MODEL_DATA_TRACER.cpu_usage == 2 * 3 * 4
        assert GLOBAL_MODEL_DATA_TRACER.cuda_usage == 0
        colo_model_data_tensor_move(cpu_t, cuda_t)
        assert GLOBAL_MODEL_DATA_TRACER.cpu_usage == 0
        assert GLOBAL_MODEL_DATA_TRACER.cuda_usage == 2 * 3 * 4
        GLOBAL_MODEL_DATA_TRACER.clear()

    GLOBAL_MODEL_DATA_TRACER.close()


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
