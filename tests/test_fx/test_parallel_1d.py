#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pytest
import torch
from torch.fx import symbolic_trace

from colossalai.fx.passes import column_shard_linear_pass
from colossalai.initialize import launch
from colossalai.legacy.core import global_context as gpc
from colossalai.logging import disable_existing_loggers
from colossalai.testing import clear_cache_before_run, rerun_if_address_is_in_use, spawn


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


CONFIG = dict(parallel=dict(tensor=dict(mode="1d", size=2)))


def check_layer(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
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
@clear_cache_before_run()
@rerun_if_address_is_in_use()
def test_1d():
    spawn(check_layer, 2)


if __name__ == "__main__":
    test_1d()
