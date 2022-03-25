from colossalai.utils.memory_tracer.model_data_memtracer import GLOBAL_MODEL_DATA_TRACER
from colossalai.utils.memory_utils.utils import colo_model_data_tensor_move
from colossalai.utils import free_port

from colossalai.zero.sharded_param import ShardedTensor
import colossalai

import torch

from functools import partial
import torch.multiprocessing as mp
import pytest


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
    GLOBAL_MODEL_DATA_TRACER.close()


def test_tensor_move():
    mp.spawn(run_tensor_move, nprocs=1)


if __name__ == '__main__':
    test_tensor_move()
