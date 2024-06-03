from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing import assert_close

import colossalai
from colossalai.lazy import LazyInitContext
from colossalai.shardformer.layer import VocabParallelEmbedding1D
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn


@parameterize("lazy_init", [False, True])
def check_vocab_embedding_1d(lazy_init: bool):
    ctx = LazyInitContext() if lazy_init else nullcontext()

    embedding = nn.Embedding(128, 32).to("cuda")
    with ctx:
        embedding_copy = nn.Embedding(128, 32).to("cuda")
    dist_embedding_1d = VocabParallelEmbedding1D.from_native_module(embedding_copy, process_group=None)

    assert dist_embedding_1d.weight.shape == torch.Size([64, 32])
    assert dist_embedding_1d.num_embeddings == 128
    assert dist_embedding_1d.embedding_dim == 32
    assert embedding_copy.weight is dist_embedding_1d.weight

    # ensure state dict is reversibly loadable
    embedding.load_state_dict(dist_embedding_1d.state_dict())
    dist_embedding_1d.load_state_dict(embedding.state_dict())

    # check embedding correctness
    x = torch.randint(0, 128, (4, 32)).to("cuda")
    org_out = embedding(x)
    dist_out = dist_embedding_1d(x)
    assert_close(org_out, dist_out)

    # check backward correctness
    org_out.sum().backward()
    dist_out.sum().backward()

    rank = dist.get_rank()
    target_grad = torch.chunk(embedding.weight.grad, 2, dim=0)[rank]
    assert_close(target_grad, dist_embedding_1d.weight.grad)


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    check_vocab_embedding_1d()


@rerun_if_address_is_in_use()
def test_vocab_embedding():
    spawn(run_dist, nprocs=2)


if __name__ == "__main__":
    test_vocab_embedding()
