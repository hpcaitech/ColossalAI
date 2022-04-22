from joblib import Parallel
from numpy import allclose, require
import torch
from colossalai.context.parallel_mode import ParallelMode
from colossalai.tensor import ColoTensor
from copy import deepcopy

from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.logging import get_dist_logger
from colossalai.testing import parameterize, rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.core import global_context as gpc


def run_linear_tp1d_row_test():
    in_dim = 4
    out_dim = 5

    fc = torch.nn.Linear(in_dim, out_dim, bias=True)
    fc_ref = deepcopy(fc)

    input_ref = torch.randn(1, in_dim)
    input_tensor = input_ref.clone()

    # sharded_weight = ColoTensor.init_from_torch_tensor(fc_ref.weight, "1Drow")

    # shard weight at begiin
    world_size = gpc.get_world_size(ParallelMode.PARALLEL_1D)
    sharded_weight = ColoTensor(in_dim / world_size, out_dim, shard_spec="1Drow")
    sharded_bias = ColoTensor.init_from_torch_tensor(fc_ref.bias)

    # replace the torch nn.Parameters with ShardedTensor
    delattr(fc, 'weight')
    setattr(fc, 'weight', sharded_weight)
    delattr(fc, 'bias')
    setattr(fc, 'bias', sharded_bias)

    fc.weight.requires_grad = True
    fc.bias.requires_grad = True

    # torch.nn.functional.linear(torch.randn(1, in_dim), sharded_weight, sharded_bias)
    out = fc(input_tensor)
    loss = out.sum()
    loss.backward()

    out_ref = fc_ref(input_ref)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    assert (loss_ref == loss)
    assert allclose(fc_ref.weight.grad, fc.weight.torch_tensor().grad)


def run_dist(rank, world_size, port):
    config = dict(parallel=dict(tensor=dict(mode="1d", size=world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_linear_tp1d_row_test()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [4])
@rerun_if_address_is_in_use()
def test_linear_1d(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_linear_1d(4)
