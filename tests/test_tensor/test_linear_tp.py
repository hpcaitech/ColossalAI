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
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.core import global_context as gpc
import torch.distributed as dist

def check_equal(A, B):
    assert torch.allclose(A, B, rtol=1e-3, atol=1e-1) == True

def run_linear_tp1d_row_test():
    device = get_current_device()
    dtype = torch.float32
    DEPTH = 4
    in_features = 4
    out_features = 5

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    layer = torch.nn.Linear(in_features, out_features)

    A_shape = (2, in_features)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    dist.broadcast(A_master, src=0)
    A = A_master.clone()
    A.requires_grad = True

    W_shape = (out_features, in_features)
    W_master = torch.randn(W_shape, dtype=dtype, device=device)
    dist.broadcast(W_master, src=0)
    W = torch.chunk(W_master, DEPTH, dim=-1)[i]
    W = W.clone()
    W.requires_grad = True

    B_shape = (out_features)
    B_master = torch.randn(B_shape, dtype=dtype, device=device)
    dist.broadcast(B_master, src=0)
    B = B_master.clone()
    B.requires_grad = True

    # replace the torch nn.Parameters with ShardedTensor
    sharded_weight = ColoTensor.init_from_torch_tensor(W)
    sharded_weight._shard_spec = "1Drow"
    sharded_bias = ColoTensor.init_from_torch_tensor(B)
    delattr(layer, 'weight')
    setattr(layer, 'weight', sharded_weight)
    delattr(layer, 'bias')
    setattr(layer, 'bias', sharded_bias)
    layer.weight.requires_grad = True
    layer.bias.requires_grad = True
    out = layer(A)

    A_master = A_master.clone()
    A_master.requires_grad = True
    W_master = W_master.clone()
    W_master.requires_grad = True
    B_master = B_master.clone()
    B_master.requires_grad = True
    C_master = torch.matmul(A_master, W_master.transpose(0, 1)) + B_master
    C = C_master.clone()

    check_equal(out, C)
    print('linear_row forward: pass')

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_current_device())
    dist.broadcast(grad_master, src=0)
    grad = grad_master.clone()
    out.backward(grad)

    grad_master = grad_master.clone()
    C_master.backward(grad_master)

    W_grad = W_master.grad
    W_grad = torch.chunk(W_grad, DEPTH, dim=-1)[i]
    check_equal(W_grad, layer.weight.grad)

    B_grad = B_master.grad
    check_equal(B_grad, layer.bias.grad)

    print('linear_row backward: pass')


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
