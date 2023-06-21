import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing import assert_close

import colossalai
from colossalai.shardformer.layer import Linear1D_Col, Linear1D_Row
from colossalai.testing import rerun_if_address_is_in_use, spawn


def check_linear_1d_col():
    linear = nn.Linear(32, 128).cuda()
    linear_col = Linear1D_Col.from_native_module(linear, process_group=None, gather_output=True)

    assert linear_col.weight.shape == torch.Size([64, 32])
    assert linear_col.bias.shape == torch.Size([64])

    # check computation correctness
    x = torch.rand(4, 32).cuda()
    out = linear(x)
    gather_out = linear_col(x)
    assert_close(out, gather_out)

    # check backward correctness
    out.sum().backward()
    gather_out.sum().backward()

    rank = dist.get_rank()
    target_grad = torch.chunk(linear.weight.grad, 2, dim=0)[rank]
    assert_close(target_grad, linear_col.weight.grad)


def check_linear_1d_row():
    linear = nn.Linear(32, 128).cuda()
    linear_row = Linear1D_Row.from_native_module(linear, process_group=None, parallel_input=False)

    assert linear_row.weight.shape == torch.Size([128, 16])
    assert linear_row.bias.shape == torch.Size([128])

    # check computation correctness
    x = torch.rand(4, 32).cuda()
    out = linear(x)
    gather_out = linear_row(x)
    assert_close(out, gather_out)

    # check backward correctness
    out.sum().backward()
    gather_out.sum().backward()

    rank = dist.get_rank()
    target_grad = torch.chunk(linear.weight.grad, 2, dim=1)[rank]
    assert_close(target_grad, linear_row.weight.grad)


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    check_linear_1d_col()
    check_linear_1d_row()


@rerun_if_address_is_in_use()
def test_linear():
    spawn(run_dist, nprocs=2)


if __name__ == '__main__':
    test_linear()
