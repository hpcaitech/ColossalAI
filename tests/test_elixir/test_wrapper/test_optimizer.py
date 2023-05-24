import copy
import os
from functools import partial

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close

from colossalai.elixir.cuda import gpu_device
from colossalai.elixir.search import simple_search
from colossalai.elixir.utils import init_distributed, seed_all
from colossalai.elixir.wrapper import ElixirModule, ElixirOptimizer
from colossalai.nn.optimizer import HybridAdam
from tests.test_elixir.utils import TEST_MODELS, allclose, assert_dict_values, to_cuda


def exam_optimizer_one_model(model_fn, data_fn, nproc, group, exam_seed=2261):
    ddp_model = model_fn().cuda()
    test_model = copy.deepcopy(ddp_model)

    ddp_model = DDP(ddp_model)
    ddp_optim = HybridAdam(ddp_model.parameters(), lr=1e-1, weight_decay=0)

    test_optim = HybridAdam(test_model.parameters(), lr=1e-1, weight_decay=0)
    sr = simple_search(test_model, nproc, shard_device=gpu_device())
    test_model = ElixirModule(test_model, sr, group)
    test_optim = ElixirOptimizer(test_model, test_optim)

    # get different data
    seed_all(exam_seed + dist.get_rank(group))
    data = to_cuda(data_fn())

    seed_all(exam_seed, cuda_deterministic=True)
    ddp_optim.zero_grad()
    ddp_loss = ddp_model(**data)
    ddp_loss.backward()
    ddp_optim.step()

    test_optim.zero_grad()
    test_loss = test_model(**data)
    test_optim.backward(test_loss)
    test_optim.step()

    assert_close(ddp_loss, test_loss)
    torch_st = ddp_model.module.state_dict()
    test_st = test_model.state_dict()
    assert_dict_values(torch_st, test_st, fn=partial(allclose, rtol=2e-6, atol=2e-5))


def exam_optimizer_in_models(nproc, group):
    model_fn, data_fn = TEST_MODELS.get('resnet')
    exam_optimizer_one_model(model_fn, data_fn, nproc, group)


def run_dist(rank, world_size):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(29512)
    init_distributed()
    exam_optimizer_in_models(nproc=world_size, group=dist.GroupMember.WORLD)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2, 4])
def test_elixir_optimizer(world_size):
    run_func = partial(run_dist, world_size=world_size)
    torch.multiprocessing.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_elixir_optimizer(world_size=4)
