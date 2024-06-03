from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing import assert_close

import colossalai
from colossalai.lazy import LazyInitContext
from colossalai.shardformer.layer import Embedding1D
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn


@parameterize("lazy_init", [False, True])
def check_embedding_1d(lazy_init: bool):
    ctx = LazyInitContext() if lazy_init else nullcontext()

    embedding = nn.Embedding(32, 128).cuda()
    with ctx:
        embedding_copy = nn.Embedding(32, 128).cuda()
    embedding_1d = Embedding1D.from_native_module(embedding_copy, process_group=None)

    assert embedding_1d.weight.shape == torch.Size([32, 64])
    assert embedding_1d.weight is embedding_copy.weight

    # ensure state dict is reversibly loadable
    embedding.load_state_dict(embedding_1d.state_dict())
    embedding_1d.load_state_dict(embedding.state_dict())

    # check computation correctness
    x = torch.randint(low=0, high=32, size=(4, 32)).cuda()
    out = embedding(x)
    gather_out = embedding_1d(x)
    assert_close(out, gather_out)

    # check backward correctness
    out.sum().backward()
    gather_out.sum().backward()

    rank = dist.get_rank()
    target_grad = torch.chunk(embedding.weight.grad, 2, dim=1)[rank]
    assert_close(target_grad, embedding_1d.weight.grad)


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    check_embedding_1d()


@rerun_if_address_is_in_use()
def test_embedding_1d():
    spawn(run_dist, nprocs=2)


if __name__ == "__main__":
    test_embedding_1d()
