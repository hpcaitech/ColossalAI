import pytest
import colossalai
import torch
import torch.multiprocessing as mp
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.gemini import ChunkManager
from functools import partial
from colossalai.nn.parallel import ColoDDP, ZeroDDP
from colossalai.gemini.gemini_mgr import GeminiManager
from typing import Callable
import torch.distributed as dist
import os
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def init_ddp(module: torch.nn.Module) -> ColoDDP:
    return ColoDDP(module)


def init_ddpv2(module: torch.nn.Module, use_chunk: bool = False) -> ZeroDDP:
    chunk_size = ChunkManager.search_chunk_size(module, 64, 2) if use_chunk else None
    chunk_manager = ChunkManager(chunk_size)
    gemini_manager = GeminiManager('cuda', chunk_manager)
    return ZeroDDP(module, gemini_manager)


class Net(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 3, bias=False)
        self.fc2 = torch.nn.Linear(3, 1, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def run_fwd_bwd(ddp_cls: ColoDDP, init_ddp_func: Callable[[torch.nn.Module], ColoDDP]):
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
    w1_grads = [torch.empty_like(w1) for _ in range(dist.get_world_size())]
    dist.all_gather(w1_grads, w1.grad)
    assert torch.equal(w1_grads[0], w1_grads[1])
    w2_grads = [torch.empty_like(w2) for _ in range(dist.get_world_size())]
    dist.all_gather(w2_grads, w2.grad)
    assert not torch.equal(w2_grads[0], w2_grads[1])


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    set_seed(dist.get_rank())
    run_fwd_bwd(ColoDDP, init_ddp)
    run_fwd_bwd(ZeroDDP, partial(init_ddpv2, use_chunk=False))
    run_fwd_bwd(ZeroDDP, partial(init_ddpv2, use_chunk=True))


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [2])
@rerun_if_address_is_in_use()
def test_ddp_ignore_params(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_ddp_ignore_params(2)
