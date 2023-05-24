import copy
import os
from functools import partial

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close

from colossalai.elixir.chunk import ChunkGroup
from colossalai.elixir.utils import init_distributed, seed_all
from tests.test_elixir.test_chunk.fetcher_utils import hook_transform
from tests.test_elixir.utils import TEST_MODELS, to_cuda


def check_gradient(ddp_model, my_model, cg: ChunkGroup):
    for chunk in cg.fused_chunks:
        cg.access_chunk(chunk)

    for (name, p0), p1 in zip(ddp_model.named_parameters(), my_model.parameters()):
        torch.cuda.synchronize()
        print(f'checking parameter {name}')
        assert_close(p0.grad.data, p1.data)


def exam_chunk_fetcher(nproc, group):
    model_fn, data_fn = TEST_MODELS.get('resnet')
    torch_model = model_fn().cuda()
    test_model = copy.deepcopy(torch_model)

    rank = dist.get_rank(group)
    # get different data
    seed_all(1001 + rank)
    data = to_cuda(data_fn())

    seed_all(1001, cuda_deterministic=True)
    ddp_model = DDP(torch_model)
    ddp_loss = ddp_model(**data)
    ddp_loss.backward()

    hook_model, cg = hook_transform(test_model, group)
    my_loss = hook_model(**data)
    my_loss.backward()

    assert_close(ddp_loss, my_loss)
    check_gradient(ddp_model, hook_model, cg)
    print('private chunk fetcher is ok')


def run_dist(rank, world_size):
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(29512)
    init_distributed()
    exam_chunk_fetcher(nproc=world_size, group=dist.GroupMember.WORLD)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2, 4])
def test_chunk_fetcher(world_size):
    run_func = partial(run_dist, world_size=world_size)
    torch.multiprocessing.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_chunk_fetcher(world_size=2)
