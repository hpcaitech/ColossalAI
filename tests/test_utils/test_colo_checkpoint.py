from abc import ABC, abstractmethod
import os, shutil
import torch
import torch.nn as nn
import pytest
from functools import partial

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import MultiplicativeLR

import colossalai
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.tensor import ComputePattern, ComputeSpec, DistSpecManager, ShardSpec, ProcessGroup
from colossalai.nn.parallel.data_parallel import ColoDDP
from colossalai.utils.checkpoint import save_checkpoint, load_checkpoint
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import ColoOptimizer

from tests.components_to_test.registry import non_distributed_component_funcs


def init_1d_row_for_linear_weight_spec(model, pg: ProcessGroup):
    spec = (ShardSpec([-1], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    for n, p in model.named_parameters():
        print(n, p.shape)
        if 'weight' in n and 'proj' in n:
            with DistSpecManager.no_grad():
                p.set_process_group(pg)
                p.set_tensor_spec(*spec)


def check_param_equal(model, torch_model):
    for p, torch_p in zip(model.parameters(), torch_model.parameters()):
        assert torch.allclose(torch_p, p, rtol=1e-3, atol=1e-1)


def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    else:
        raise ValueError("file {} is not a file or dir.".format(path))


def run_checkpoint(init_spec_func, use_ddp, use_mp_reload, test_scheduler, pg):
    get_components_func = non_distributed_component_funcs.get_callable('simple_net')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    with ColoInitContext(device=get_current_device()):
        model = model_builder()

    with ColoInitContext(device=get_current_device()):
        model_reload = model_builder()

    model = model.cuda()
    model.train()

    model_reload = model_reload.cuda()
    model_reload.train()

    init_spec_func(model, pg)

    if use_mp_reload:
        init_spec_func(model_reload, pg)

    if use_ddp:
        model = ColoDDP(model, pg)
        model_reload = ColoDDP(model_reload, pg)

    optimizer = ColoOptimizer(dict(model.named_parameters()), torch.optim.Adam, lr=0.1)
    optimizer_reload = ColoOptimizer(dict(model_reload.named_parameters()), torch.optim.Adam, lr=0.1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # optimizer_reload = torch.optim.Adam(model_reload.parameters(),
    #                                     lr=0.001,
    #                                     betas=(0.9, 0.999),
    #                                     eps=1e-08,
    #                                     weight_decay=0)

    for i, (data, label) in enumerate(train_dataloader):
        data = data.to(get_current_device())
        label = label.to(get_current_device())

        optimizer.zero_grad()

        if criterion:
            output = model(data)
            loss = criterion(output, label)
        else:
            output = model(data, label)

        loss.backward()

        for p in model.parameters():
            print(p.grad)
        optimizer.step()
        break

    # for k, v in optimizer.state_dict().items():
    #     print(k, v)

    # save_checkpoint('./checkpoint', 0, model, None, None)
    # dist.barrier()
    # load_checkpoint('./checkpoint', 0, model_reload, None, None)

    # # Since model is sharded, we merge them before param checking.
    # for p in model.parameters():
    #     p.to_replicate_()

    # for p in model_reload.parameters():
    #     p.to_replicate_()

    # check_param_equal(model, model_reload)


def run_dist(rank, world_size, port, use_ddp, use_mp_reload, test_scheduler):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    pg = ProcessGroup(tp_degree=world_size)
    run_checkpoint(init_1d_row_for_linear_weight_spec, use_ddp, use_mp_reload, test_scheduler=test_scheduler, pg=pg)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2])
@pytest.mark.parametrize('use_ddp', [True, False])
@pytest.mark.parametrize('use_mp_reload', [True, False])
@pytest.mark.parametrize('test_scheduler', ['colossalai_cosine_warmup', 'torch_cosine', 'torch_lambda'])
@rerun_if_address_is_in_use()
def test_checkpoint(world_size, use_ddp, use_mp_reload, test_scheduler):
    if not os.path.isdir('./checkpoint'):
        os.mkdir('./checkpoint')
    run_func = partial(run_dist,
                       world_size=world_size,
                       port=free_port(),
                       use_ddp=use_ddp,
                       use_mp_reload=use_mp_reload,
                       test_scheduler=test_scheduler)
    mp.spawn(run_func, nprocs=world_size)
    remove('./checkpoint')


if __name__ == '__main__':
    test_checkpoint(4, True, False, "torch_cosine")
