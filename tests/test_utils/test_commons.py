from colossalai.utils.cuda import get_current_device
from colossalai.utils.memory_tracer.model_data_memtracer import GLOBAL_MODEL_DATA_TRACER
from colossalai.utils.memory_utils.utils import colo_model_data_tensor_move, colo_model_data_tensor_move_inline
from colossalai.utils import free_port

from colossalai.zero.sharded_param import ShardedTensor
import colossalai

import torch

from functools import partial
import torch.multiprocessing as mp
import pytest


def run_colo_model_data_tensor_move_inline(rank, world_size):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=free_port(), backend='nccl')
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


def run_colo_model_data_tensor_move(rank, world_size):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=free_port(), backend='nccl')
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


def run_tensor_move(rank):
    colossalai.launch(config={}, rank=0, world_size=1, host='localhost', port=free_port(), backend='nccl')

    assert (GLOBAL_MODEL_DATA_TRACER.cuda_usage == 0)
    GLOBAL_MODEL_DATA_TRACER.start()

    src_t = torch.ones(2, 3).cuda()
    GLOBAL_MODEL_DATA_TRACER.add_tensor(src_t)
    assert (GLOBAL_MODEL_DATA_TRACER.cuda_usage == 24)
    tgt_t = torch.zeros(2, 3)

    colo_model_data_tensor_move(src_t, tgt_t)
    assert (GLOBAL_MODEL_DATA_TRACER.cuda_usage == 0)
    assert (torch.sum(tgt_t) == 6.0), f"{torch.sum(tgt_t.payload)} vs. 6.0"

    src_t = torch.ones(2, 3)
    tgt_t = torch.zeros(2, 3).cuda().half()
    colo_model_data_tensor_move(src_t, tgt_t)
    assert (GLOBAL_MODEL_DATA_TRACER.cuda_usage == 12), f"cuda usage {GLOBAL_MODEL_DATA_TRACER.cuda_usage}"
    # the src_t has been removed
    assert (src_t.numel() == 0)
    assert (torch.sum(tgt_t) == 6.0), f"{torch.sum(tgt_t.payload)} vs. 6.0"

    src_t = ShardedTensor(torch.ones(2, 3))
    tgt_t = ShardedTensor(torch.zeros(2, 3).cuda().half())
    colo_model_data_tensor_move(src_t, tgt_t)
    assert (GLOBAL_MODEL_DATA_TRACER.cuda_usage == 24), f"cuda usage {GLOBAL_MODEL_DATA_TRACER.cuda_usage}"
    assert (torch.sum(tgt_t.payload) == 6.0), f"{torch.sum(tgt_t.payload)} vs. 6.0"

    assert (tgt_t.device.type == 'cuda')
    colo_model_data_tensor_move_inline(tgt_t, torch.device('cpu'))
    assert (tgt_t.device.type == 'cpu')
    assert (GLOBAL_MODEL_DATA_TRACER.cuda_usage == 12), f"cuda usage {GLOBAL_MODEL_DATA_TRACER.cuda_usage}"

    GLOBAL_MODEL_DATA_TRACER.close()


def test_tensor_move():
    mp.spawn(run_tensor_move, nprocs=1)


@pytest.mark.parametrize("world_size", [1, 2])
def test_tensor_move_inline(world_size):
    mp.spawn(partial(run_colo_model_data_tensor_move_inline, world_size=world_size), nprocs=1)


@pytest.mark.parametrize("world_size", [1, 2])
def test_tensor_move(world_size):
    mp.spawn(partial(run_colo_model_data_tensor_move, world_size=world_size), nprocs=1)


if __name__ == '__main__':
    test_tensor_move(world_size=1)
