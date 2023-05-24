import os
from copy import deepcopy
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing import assert_close

from colossalai.elixir.search import simple_search
from colossalai.elixir.utils import init_distributed
from colossalai.elixir.wrapper import ElixirModule


def exam_fused_layernorm(nproc, group):
    torch_model = nn.LayerNorm(2048)
    fused_model = deepcopy(torch_model)

    torch_model = torch_model.cuda()
    sr = simple_search(fused_model, nproc, 1, 1.0, verbose=True)
    fused_model = ElixirModule(fused_model, sr, group, use_fused_kernels=True)

    data = torch.randn(2, 2048, device='cuda')

    torch_loss = torch_model(data).sum()
    torch_loss.backward()

    fused_loss = fused_model(data).sum()
    fused_model.backward(fused_loss)

    assert_close(torch_loss, fused_loss)

    grad_state = fused_model.state_dict(from_param=True)
    for name, param in torch_model.named_parameters():
        assert_close(param.grad.cpu(), grad_state[name])


def run_dist(rank, world_size):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(29512)
    init_distributed()
    exam_fused_layernorm(nproc=world_size, group=dist.GroupMember.WORLD)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1])
def test_fused_layernorm(world_size):
    run_func = partial(run_dist, world_size=world_size)
    torch.multiprocessing.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_fused_layernorm(world_size=1)
