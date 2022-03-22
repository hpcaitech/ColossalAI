#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pprint
from functools import partial

import colossalai.nn as col_nn
import pytest
import torch.multiprocessing as mp
import torch.nn as nn
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.utils import free_port, is_using_pp
from colossalai.utils.checkpointing import load_checkpoint, save_checkpoint

CONFIG = dict(parallel=dict(pipeline=dict(size=2), tensor=dict(size=4, mode='1d')), )


def build_pipeline(model):
    from colossalai.builder.pipeline import partition_uniform
    pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
    pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    depth = len(model)
    start, end = partition_uniform(depth, pipeline_size, 1)[pipeline_rank][0]
    layers = []
    for i in range(depth):
        if start <= i < end:
            layers.append(model[i])
        else:
            layers.append(nn.Identity())
    return nn.Sequential(*tuple(layers))


def check_checkpoint(rank, world_size, port):
    disable_existing_loggers()
    launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    m1 = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))
    sd1 = m1.state_dict()
    if gpc.get_global_rank() == 0:
        print(f'Rank {gpc.get_global_rank()}:\n{pprint.pformat(sd1)}\n')
    save_checkpoint('test.pt', 0, m1)

    m2 = nn.Sequential(col_nn.Linear(4, 8), col_nn.Linear(8, 4))
    if is_using_pp():
        m2 = build_pipeline(m2)

    # sd = sd1 if gpc.get_local_rank(ParallelMode.MODEL) == 0 else OrderedDict()
    # if is_using_pp():
    #     sd = partition_pipeline_parallel_state_dict(m2, sd)
    # m2.load_state_dict(sd)
    load_checkpoint('test.pt', m2)
    sd2 = m2.state_dict()
    # if is_using_pp() and gpc.get_local_rank(ParallelMode.TENSOR) == 0:
    #     sd2 = gather_pipeline_parallel_state_dict(sd2)
    print(f'Rank {gpc.get_global_rank()}:\n{pprint.pformat(sd2)}\n')


@pytest.mark.dist
@pytest.mark.skip(reason="work in progress")
def test_checkpoint():
    world_size = 8
    run_func = partial(check_checkpoint, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_checkpoint()
