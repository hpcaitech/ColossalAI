import os
import random
from typing import Callable, Type

import numpy as np
import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.nn.parallel import ColoDDP
from colossalai.tensor import ProcessGroup
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.utils.cuda import get_current_device
from colossalai.zero import ColoInitContext, ZeroDDP
from colossalai.zero.gemini.chunk import ChunkManager, search_chunk_configuration
from colossalai.zero.gemini.gemini_mgr import GeminiManager


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def init_ddp(module: torch.nn.Module) -> ColoDDP:
    pg = ProcessGroup()
    return ColoDDP(module, process_group=pg)


def init_ddpv2(module: torch.nn.Module) -> ZeroDDP:
    chunk_config, *_ = search_chunk_configuration(module, 4, 1024)
    chunk_manager = ChunkManager(chunk_config)
    gemini_manager = GeminiManager('cuda', chunk_manager)
    return ZeroDDP(module, gemini_manager)


class Net(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 3, bias=False)
        self.fc2 = torch.nn.Linear(3, 1, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def run_fwd_bwd(ddp_cls: Type[ColoDDP], init_ddp_func: Callable[[torch.nn.Module], ColoDDP]):
    with ColoInitContext(device=get_current_device()):
        model = Net().cuda()
    w1 = model.fc1.weight
    w2 = model.fc2.weight
    ddp_cls.set_params_to_ignore([w2])
    model = init_ddp_func(model)
    x = torch.rand(2, 3, device=get_current_device())
    logits = model(x)
    loss = torch.sum(logits)
    model.backward(loss)

    if ddp_cls is ZeroDDP:
        w1s_grad = w1
    else:
        w1s_grad = w1.grad

    w1_grads = [torch.empty_like(w1) for _ in range(dist.get_world_size())]
    dist.all_gather(w1_grads, w1s_grad)
    assert torch.equal(w1_grads[0], w1_grads[1])
    w2_grads = [torch.empty_like(w2) for _ in range(dist.get_world_size())]
    dist.all_gather(w2_grads, w2.grad)
    assert not torch.equal(w2_grads[0], w2_grads[1])


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    set_seed(dist.get_rank())
    run_fwd_bwd(ColoDDP, init_ddp)
    run_fwd_bwd(ZeroDDP, init_ddpv2)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [2])
@rerun_if_address_is_in_use()
def test_ddp_ignore_params(world_size):
    spawn(run_dist, world_size)


if __name__ == '__main__':
    test_ddp_ignore_params(2)
