import pytest
import torch
import torch.distributed as dist

from colossalai.accelerator import get_accelerator
from colossalai.legacy.communication import all_gather, all_reduce, reduce_scatter
from colossalai.legacy.context import ParallelMode
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.initialize import launch
from colossalai.testing import rerun_if_address_is_in_use, spawn

CONFIG = dict(parallel=dict(data=8, pipeline=1, tensor=dict(mode=None, size=1)))

SIZE = 8


def check_all_gather():
    tensor = torch.tensor([dist.get_rank() * SIZE + j for j in range(SIZE)])
    tensor = tensor.to(get_accelerator().get_current_device())
    print("Before:   Rank {0} - {1}".format(dist.get_rank(), tensor))
    tensor, op = all_gather(tensor, 0, ParallelMode.GLOBAL, async_op=True)
    print("After:    Rank {0} - {1}".format(dist.get_rank(), tensor))
    op.wait()
    print("Complete: Rank {0} - {1}".format(dist.get_rank(), tensor))
    torch.cuda.synchronize()


def check_reduce_scatter():
    tensor = torch.tensor([dist.get_rank() * SIZE + j for j in range(SIZE)])
    tensor = tensor.to(get_accelerator().get_current_device())
    print("Before:   Rank {0} - {1}".format(dist.get_rank(), tensor))
    tensor, op = reduce_scatter(tensor, 0, ParallelMode.GLOBAL, async_op=True)
    print("After:    Rank {0} - {1}".format(dist.get_rank(), tensor))
    op.wait()
    print("Complete: Rank {0} - {1}".format(dist.get_rank(), tensor))
    torch.cuda.synchronize()


def check_all_reduce():
    tensor = torch.tensor([dist.get_rank() * SIZE + j for j in range(SIZE)])
    tensor = tensor.to(get_accelerator().get_current_device())
    print("Before:   Rank {0} - {1}".format(dist.get_rank(), tensor))
    tensor, op = all_reduce(tensor, ParallelMode.GLOBAL, async_op=True)
    print("After:    Rank {0} - {1}".format(dist.get_rank(), tensor))
    op.wait()
    print("Complete: Rank {0} - {1}".format(dist.get_rank(), tensor))
    torch.cuda.synchronize()


def check_layer(rank, world_size, port):
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")

    assert dist.get_rank() == gpc.get_global_rank()
    print("Rank {} / {}".format(dist.get_rank(), dist.get_world_size()))

    check_all_gather()
    check_reduce_scatter()
    check_all_reduce()

    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_comm():
    spawn(check_layer, 4)


if __name__ == "__main__":
    test_comm()
