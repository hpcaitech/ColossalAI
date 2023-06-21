import torch
import torch.distributed as dist
import torch.nn as nn

import colossalai
from colossalai.shardformer.layer import Dropout1D
from colossalai.testing import assert_equal, assert_not_equal, rerun_if_address_is_in_use, spawn


def check_dropout():
    dropout = nn.Dropout().cuda()
    dropout_1d = Dropout1D.from_native_module(dropout, process_group=None)

    # check computation correctness
    x = torch.rand(4, 128).cuda()

    # we set seed so that dropout will generate the same mask
    torch.cuda.manual_seed(1024)
    out = dropout(x)

    # we set seed to simulate the same scenario
    # but expect the dropout mask to be different
    # due to the internal randomness control
    torch.cuda.manual_seed(1024)
    out_1d = dropout_1d(x)

    # ensure out is the same across all ranks
    world_size = dist.get_world_size()
    out_all = [torch.empty_like(out) for _ in range(world_size)]
    dist.all_gather(out_all, out)

    for i in range(world_size):
        assert_equal(out_all[i], out_all[0])

    # ensure out_1d is different across ranks
    out_1d_all = [torch.zeros_like(out_1d) for _ in range(world_size)]
    dist.all_gather(out_1d_all, out_1d)
    for i in range(1, world_size):
        assert_not_equal(out_1d_all[i], out_1d_all[0])


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    check_dropout()


@rerun_if_address_is_in_use()
def test_dropout():
    spawn(run_dist, nprocs=2)


if __name__ == '__main__':
    test_dropout()
