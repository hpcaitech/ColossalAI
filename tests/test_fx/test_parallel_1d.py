#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers
from colossalai.initialize import launch
from colossalai.utils import free_port
from colossalai.testing import rerun_if_address_is_in_use
from torch.fx import symbolic_trace
from colossalai.fx.passes import column_shard_linear_pass


class MLP(torch.nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim)
        self.linear2 = torch.nn.Linear(dim, dim)
        self.linear3 = torch.nn.Linear(dim, dim)
        self.linear4 = torch.nn.Linear(dim, dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x


CONFIG = dict(parallel=dict(tensor=dict(mode='1d', size=2)))


def check_layer(rank, world_size, port):
    disable_existing_loggers()
    launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    input_tensor = torch.rand(2, 16).cuda()
    model = MLP(16).cuda()
    symbolic_traced = symbolic_trace(model)
    output = model(input_tensor)
    splitted_gm = column_shard_linear_pass(symbolic_traced)
    new_output = splitted_gm(input_tensor)

    assert output.equal(new_output)

    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_1d():
    world_size = 2
    run_func = partial(check_layer, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_1d()
