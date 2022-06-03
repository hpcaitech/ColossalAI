from colossalai.utils import free_port, ColoInitContext, get_current_device
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.tensor import TensorSpec, ComputePattern, ParallelAction

from functools import partial
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode

from colossalai.nn import init_colo_module
from colossalai.nn.parallel import ColoDDP

import colossalai
import torch
import torch.multiprocessing as mp
import pytest


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.embed = torch.nn.Embedding(20, 4)
        self.proj = torch.nn.Linear(4, 8)

    def forward(self, x):
        # move input to cpu and restore output
        current_dev = x.device
        x = x.to('cpu')
        x = self.embed(x)
        x = x.to(current_dev)

        x = self.proj(x)
        return x


def run_hybrid_device(use_ddp):
    with ColoInitContext(device=get_current_device()):
        model = Net()

    real_model = model
    if use_ddp:
        model = ColoDDP(model)
        real_model = model.module

    print(f'embedding weight size: {real_model.embed.weight.size()} | device: {real_model.embed.weight.device}')
    #print(f'linear weight size: {real_model.proj.weight.size()} | device: {real_model.proj.weight.device}')
    parallel_action = ParallelAction(ComputePattern.TP1D)
    init_colo_module(model, parallel_action, recursive=True, mode='col')

    # use cpu gloo to handle embedding
    real_model.embed.to('cpu')
    gloo_group_tp = gpc.get_cpu_group(ParallelMode.PARALLEL_1D)
    real_model.embed.weight.spec.dist_spec.process_group = gloo_group_tp

    print(f'embedding weight size: {real_model.embed.weight.size()} | new device: {real_model.embed.weight.device}')
    #print(f'linear weight size: {real_model.proj.weight.size()} | new device: {real_model.proj.weight.device}')

    data = torch.randint(low=0, high=20, size=(16,), device=get_current_device())
    out = model(data)
    out.sum().backward()


def run_dist(rank, world_size, port, use_ddp):
    if use_ddp and world_size == 1:
        return
    tp_world_size = world_size // 2 if use_ddp else world_size
    config = dict(parallel=dict(tensor=dict(mode="1d", size=tp_world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_hybrid_device(use_ddp)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@pytest.mark.parametrize('use_ddp', [False, True])
@rerun_if_address_is_in_use()
# Working for simulate the embedding(CPU DP+TP) -> nn(GPU DP+TP)
def _test_hybrid_device(world_size, use_ddp):
    run_func = partial(run_dist, world_size=world_size, port=free_port(), use_ddp=use_ddp)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    _test_hybrid_device(1, False)
