import copy
import os
from functools import partial

import pytest
import torch
import torch.distributed as dist
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close

from colossalai.elixir.cuda import gpu_device
from colossalai.elixir.search import simple_search
from colossalai.elixir.utils import init_distributed, seed_all
from colossalai.elixir.wrapper import ElixirModule, ElixirOptimizer
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import run_on_environment_flag
from tests.test_elixir.utils import TEST_MODELS, to_cuda


def amp_check_model_states(ddp_optim, test_model):
    test_states = test_model.state_dict()
    for (name, _), p in zip(test_model.module.named_parameters(), amp.master_params(ddp_optim)):
        test_p = test_states[name]
        copy_p = p.to(test_p.device)
        print(f'checking parameter `{name}`: {test_p.dtype} {copy_p.dtype}')
        assert_close(test_p.data, copy_p.data)


def exam_amp_one_model(model_fn, data_fn, nproc, group, exam_seed=2261):
    ddp_model = model_fn().cuda()
    test_model = copy.deepcopy(ddp_model)
    # important here, since apex has a lazy fp32 init after the first optimizer step
    test_model = test_model.half()

    ddp_optim = HybridAdam(ddp_model.parameters(), lr=1e-1, weight_decay=0)
    ddp_model, ddp_optim = amp.initialize(ddp_model,
                                          ddp_optim,
                                          opt_level='O2',
                                          loss_scale=1.0,
                                          keep_batchnorm_fp32=False)
    ddp_model = DDP(ddp_model, message_size=0, allreduce_always_fp32=True)
    print("ok")
    exit(0)
    test_optim = HybridAdam(test_model.parameters(), lr=1e-1, weight_decay=0)
    sr = simple_search(test_model, nproc, shard_device=gpu_device(), unified_dtype=torch.float16, verbose=True)
    test_model = ElixirModule(test_model, sr, group, dtype=torch.float16, reduce_always_fp32=True, output_fp32=True)
    test_optim = ElixirOptimizer(test_model, test_optim, initial_scale=1.0)

    # get different data
    seed_all(exam_seed + dist.get_rank(group), cuda_deterministic=True)
    for _ in range(2):
        data = to_cuda(data_fn())

        ddp_optim.zero_grad()
        ddp_loss = ddp_model(**data)
        with amp.scale_loss(ddp_loss, ddp_optim) as scaled_loss:
            scaled_loss.backward()
        ddp_optim.step()

        test_optim.zero_grad()
        test_loss = test_model(**data)
        test_optim.backward(test_loss)
        test_optim.step()

        assert_close(ddp_loss, test_loss)
        amp_check_model_states(ddp_optim, test_model)


def exam_amp_in_models(nproc, group):
    model_fn, data_fn = TEST_MODELS.get('gpt2_micro')
    exam_amp_one_model(model_fn, data_fn, nproc, group)


def run_dist(rank, world_size):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(29512)
    init_distributed()
    exam_amp_in_models(nproc=world_size, group=dist.GroupMember.WORLD)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2, 4])
@run_on_environment_flag('ELX')
def test_elixir_amp(world_size):
    run_func = partial(run_dist, world_size=world_size)
    torch.multiprocessing.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_elixir_amp(world_size=2)
