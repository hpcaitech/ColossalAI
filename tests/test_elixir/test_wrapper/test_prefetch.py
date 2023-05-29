import copy
import os
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close

from colossalai.elixir.cuda import gpu_device
from colossalai.elixir.search import simple_search
from colossalai.elixir.utils import init_distributed, seed_all
from colossalai.elixir.wrapper import ElixirModule
from colossalai.testing import run_on_environment_flag
from tests.test_elixir.utils import TEST_MODELS, to_cuda


def check_gradient(ddp_model: nn.Module, test_model: ElixirModule):
    grad_state = test_model.state_dict(from_param=True)
    for name, param in ddp_model.named_parameters():
        assert_close(param.grad.cpu(), grad_state[name])


def exam_one_module_fwd_bwd(model_fn, data_fn, nproc, group, exam_seed=2263):

    def one_step(local_model, local_input):
        loss = local_model(**local_input)
        loss.backward()
        return loss

    ddp_model = model_fn().cuda()
    test_model = copy.deepcopy(ddp_model)

    # get different data
    seed_all(exam_seed + dist.get_rank(group))
    data = to_cuda(data_fn())

    # wrap as DDP model
    ddp_model = DDP(ddp_model)
    # search how to initialize chunks
    sr = simple_search(test_model,
                       nproc,
                       shard_device=gpu_device(),
                       prefetch=True,
                       verbose=True,
                       inp=data,
                       step_fn=one_step)
    test_model = ElixirModule(test_model, sr, group, prefetch=True)

    seed_all(exam_seed, cuda_deterministic=True)
    ddp_loss = one_step(ddp_model, data)

    with torch.no_grad():
        test_loss = test_model(**data)
    assert_close(ddp_loss, test_loss)

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
    exam_modules_fwd_bwd(nproc=world_size, group=dist.GroupMember.WORLD)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2, 4])
@run_on_environment_flag('ELX')
def test_module_prefetch(world_size):
    run_func = partial(run_dist, world_size=world_size)
    torch.multiprocessing.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_module_prefetch(world_size=2)
