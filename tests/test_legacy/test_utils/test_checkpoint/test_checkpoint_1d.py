#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pprint

import pytest
import torch
import torch.nn as nn

import colossalai.legacy.nn as col_nn
from colossalai.legacy.context.parallel_mode import ParallelMode
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.initialize import launch
from colossalai.legacy.utils import is_using_pp
from colossalai.legacy.utils.checkpointing import gather_pipeline_parallel_state_dict, load_checkpoint, save_checkpoint
from colossalai.logging import disable_existing_loggers
from colossalai.testing import rerun_if_address_is_in_use, skip_if_not_enough_gpus, spawn


def build_pipeline(model):
    from colossalai.legacy.pipeline.utils import partition_uniform

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


def check_equal(A, B):
    assert torch.allclose(A, B, rtol=1e-3, atol=1e-2)


def check_checkpoint_1d(rank, world_size, port):
    config = dict(
        parallel=dict(pipeline=dict(size=2), tensor=dict(size=4, mode="1d")),
    )

    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")

    m1 = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))
    sd1 = m1.state_dict()
    if gpc.get_global_rank() == 0:
        print(f"Rank {gpc.get_global_rank()}:\n{pprint.pformat(sd1)}\n")
    save_checkpoint("test.pt", 0, m1)

    m2 = nn.Sequential(col_nn.Linear(4, 8), col_nn.Linear(8, 4))
    if is_using_pp():
        m2 = build_pipeline(m2)

    load_checkpoint("test.pt", m2)
    sd2 = m2.state_dict()
    if is_using_pp() and gpc.get_local_rank(ParallelMode.TENSOR) == 0:
        sd2 = gather_pipeline_parallel_state_dict(sd2)
    print(f"Rank {gpc.get_global_rank()}:\n{pprint.pformat(sd2)}\n")

    if gpc.get_global_rank() == 0:
        for k, v in sd1.items():
            assert k in sd2
            check_equal(v, sd2[k].to(torch.device("cpu")))


@pytest.mark.dist
@pytest.mark.skip("takes too long")
@skip_if_not_enough_gpus(min_gpus=8)
@rerun_if_address_is_in_use()
def test_checkpoint_1d():
    spawn(check_checkpoint_1d, 8)


if __name__ == "__main__":
    test_checkpoint_1d()
