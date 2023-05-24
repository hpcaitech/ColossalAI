import copy
import os
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close

from colossalai.elixir.search import simple_search
from colossalai.elixir.utils import init_distributed, seed_all
from colossalai.elixir.wrapper import ElixirModule
from tests.test_elixir.utils import TEST_MODELS, assert_dict_values, to_cuda


def check_gradient(ddp_model: nn.Module, test_model: ElixirModule):
    grad_state = test_model.state_dict(from_param=True)
    for name, param in ddp_model.named_parameters():
        assert_close(param.grad.cpu(), grad_state[name])


def exam_module_init(nproc, group, grad_flag):
    model_fn, data_fn = TEST_MODELS.get('resnet')
    torch_model = model_fn().cuda()
    test_model = model_fn().cuda()

    for p1, p2 in zip(torch_model.parameters(), test_model.parameters()):
        p1.requires_grad = p2.requires_grad = grad_flag

    sr = simple_search(test_model, nproc)
    model = ElixirModule(test_model, sr, group)
    # check function: ElixirModule.load_state_dict after ElixirModule.__init__
    torch_st = torch_model.state_dict()
    if dist.get_rank() != 0:
        torch_st = None
    test_st = model.load_state_dict(torch_st, only_rank_0=True)
    # check function: ElixirModule.state_dict after ElixirModule.__init__
    torch_st = torch_model.state_dict()
    test_st = model.state_dict()
    assert_dict_values(torch_st, test_st, fn=torch.equal)


def exam_one_module_fwd_bwd(model_fn, data_fn, nproc, group, exam_seed=2261):
    ddp_model = model_fn().cuda()
    test_model = copy.deepcopy(ddp_model)
    sr = simple_search(test_model, nproc, allocate_factor=0.6)
    test_model = ElixirModule(test_model, sr, group)

    # get different data
    seed_all(exam_seed + dist.get_rank(group))
    data = data_fn()
    data = to_cuda(data)

    seed_all(exam_seed, cuda_deterministic=True)
    ddp_model = DDP(ddp_model)
    ddp_loss = ddp_model(**data)
    ddp_loss.backward()

    test_loss = test_model(**data)
    test_model.backward(test_loss)

    assert_close(ddp_loss, test_loss)
    check_gradient(ddp_model.module, test_model)


def exam_modules_fwd_bwd(nproc, group):
    model_fn, data_fn = TEST_MODELS.get('resnet')
    exam_one_module_fwd_bwd(model_fn, data_fn, nproc, group)


def run_dist(rank, world_size):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(29512)
    init_distributed()
    exam_module_init(nproc=world_size, group=dist.GroupMember.WORLD, grad_flag=False)
    exam_module_init(nproc=world_size, group=dist.GroupMember.WORLD, grad_flag=True)
    exam_modules_fwd_bwd(nproc=world_size, group=dist.GroupMember.WORLD)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2, 4])
def test_elixir_module(world_size):
    run_func = partial(run_dist, world_size=world_size)
    torch.multiprocessing.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_elixir_module(world_size=2)
