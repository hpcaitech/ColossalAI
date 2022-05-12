import torch
from colossalai.context.parallel_mode import ParallelMode
from colossalai.tensor import ColoTensor

from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.core import global_context as gpc
from colossalai.tensor import TensorSpec, ComputePattern, ParallelAction

from _utils import check_equal, replace_parameter_add_grad, broadcast_tensor_chunk

def run_linear_tp1d_col_test():
    device = get_current_device()
    dtype = torch.float32
    DEPTH = gpc.get_world_size(ParallelMode.PARALLEL_1D)
    in_features = 4
    out_features = 8

    local_rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    layer_master = torch.nn.Linear(in_features, out_features)
    layer = torch.nn.Linear(in_features, out_features)

    A_shape = (2, in_features)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    A = broadcast_tensor_chunk(A_master, chunk_size=1)
    A.requires_grad = True

    W_shape = (out_features, in_features)
    W_master = torch.randn(W_shape, dtype=dtype, device=device)
    W = broadcast_tensor_chunk(W_master, chunk_size=1)
    W.requires_grad = True

    B_shape = (out_features)
    B_master = torch.randn(B_shape, dtype=dtype, device=device)
    B = broadcast_tensor_chunk(B_master, chunk_size=1)
    B.requires_grad = True

    # replace the torch nn.Parameters with ColoTensor
    sharded_weight = ColoTensor.init_from_torch_tensor(W)
    sharded_bias = ColoTensor.init_from_torch_tensor(B)
    parallel_action_list = [
        ParallelAction(priority=1, compute_pattern=ComputePattern.TP1DCol_Linear, parallel_mode=ParallelMode.PARALLEL_1D)
    ]
    spec = TensorSpec(parallel_action_list)
    sharded_weight.set_spec(spec) # reshard
    sharded_bias.set_spec(spec)

    replace_parameter_add_grad(layer, sharded_weight, sharded_bias)
    out = layer(A)

    replace_parameter_add_grad(layer_master, W_master, B_master)
    A_master.requires_grad = True
    #C_master = torch.matmul(A_master, W_master.transpose(0, 1)) + B_master
    C_master = layer_master(A_master)
    C = C_master.clone()

    check_equal(out, C)

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_current_device())
    grad = broadcast_tensor_chunk(grad_master, chunk_size=1)
    out.backward(grad)

    grad_master = grad_master.clone()
    C_master.backward(grad_master)

    W_grad = W_master.grad
    W_grad = torch.chunk(W_grad, DEPTH, dim=0)[local_rank]
    check_equal(W_grad, layer.weight.grad)

    B_grad = B_master.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[local_rank]
    check_equal(B_grad, layer.bias.grad)

def run_linear_tp1d_row_test():
    device = get_current_device()
    dtype = torch.float32
    DEPTH = gpc.get_world_size(ParallelMode.PARALLEL_1D)
    in_features = 4
    out_features = 5

    local_rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    layer_master = torch.nn.Linear(in_features, out_features)
    layer = torch.nn.Linear(in_features, out_features)

    A_shape = (2, in_features)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    A = broadcast_tensor_chunk(A_master, chunk_size=1)
    A.requires_grad = True

    W_shape = (out_features, in_features)
    W_master = torch.randn(W_shape, dtype=dtype, device=device)
    W = broadcast_tensor_chunk(W_master, chunk_size=1)
    W.requires_grad = True

    B_shape = (out_features)
    B_master = torch.randn(B_shape, dtype=dtype, device=device)
    B = broadcast_tensor_chunk(B_master, chunk_size=1)
    B.requires_grad = True

    # replace the torch nn.Parameters with ColoTensor
    sharded_weight = ColoTensor.init_from_torch_tensor(W)
    parallel_action_list = [
        ParallelAction(priority=1, compute_pattern=ComputePattern.TP1DRow_Linear, parallel_mode=ParallelMode.PARALLEL_1D)
    ]
    spec = TensorSpec(parallel_action_list)
    sharded_weight.set_spec(spec=spec) # reshard
    sharded_bias = ColoTensor.init_from_torch_tensor(B)
    replace_parameter_add_grad(layer, sharded_weight, sharded_bias)
    out = layer(A)

    replace_parameter_add_grad(layer_master, W_master, B_master)
    A_master.requires_grad = True
    #C_master = torch.matmul(A_master, W_master.transpose(0, 1)) + B_master
    C_master = layer_master(A_master)
    C = C_master.clone()

    check_equal(out, C)

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_current_device())
    grad = broadcast_tensor_chunk(grad_master, chunk_size=1)
    out.backward(grad)

    grad_master = grad_master.clone()
    C_master.backward(grad_master)

    W_grad = W_master.grad
    W_grad = torch.chunk(W_grad, DEPTH, dim=-1)[local_rank]
    check_equal(W_grad, layer.weight.grad)

    B_grad = B_master.grad
    check_equal(B_grad, layer.bias.grad)


def run_dist(rank, world_size, port):
    config = dict(parallel=dict(tensor=dict(mode="1d", size=world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_linear_tp1d_row_test()
    run_linear_tp1d_col_test()

@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_linear_1d(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_linear_1d()
