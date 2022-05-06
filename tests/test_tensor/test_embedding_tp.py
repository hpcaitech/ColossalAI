import torch
from colossalai.context.parallel_mode import ParallelMode
from colossalai.tensor import ColoTensor

from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.testing import parameterize, rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.core import global_context as gpc
from colossalai.tensor import TensorSpec, ComputePattern, ParallelAction

from _utils import check_equal, replace_parameter_add_grad, broadcast_tensor_chunk

def run_embedding_tp1d_col_test():
    device = get_current_device()
    dtype = torch.float32
    DEPTH = gpc.get_world_size(ParallelMode.PARALLEL_1D)
    num_embeddings = 12
    embedding_dim = 32

    local_rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    layer_master = torch.nn.Embedding(num_embeddings, embedding_dim)
    layer = torch.nn.Embedding(num_embeddings, embedding_dim)

    A_master = torch.tensor((0,3,6,9), device=device)
    A = broadcast_tensor_chunk(A_master, chunk_size=1)

    W_shape = (num_embeddings, embedding_dim)
    W_master = torch.randn(W_shape, dtype=dtype, device=device)
    W = broadcast_tensor_chunk(W_master, chunk_size=1)
    W.requires_grad = True

    # replace the torch nn.Parameters with ColoTensor
    sharded_weight = ColoTensor.init_from_torch_tensor(W)
    parallel_action_list = [
        ParallelAction(priority=1, compute_pattern=ComputePattern.TP1DCol_Embedding, 
        parallel_mode=ParallelMode.PARALLEL_1D)
    ]
    spec = TensorSpec(parallel_action_list)
    sharded_weight.set_spec(spec) # reshard
    replace_parameter_add_grad(layer, sharded_weight)
    out = layer(A)

    replace_parameter_add_grad(layer_master, W_master)
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

def run_embedding_tp1d_row_test():
    device = get_current_device()
    dtype = torch.float32
    DEPTH = gpc.get_world_size(ParallelMode.PARALLEL_1D)
    num_embeddings = 12
    embedding_dim = 32

    local_rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    layer_master = torch.nn.Embedding(num_embeddings, embedding_dim)
    layer = torch.nn.Embedding(num_embeddings, embedding_dim)

    A_master = torch.tensor((0,3,6,9), device=device)
    A = broadcast_tensor_chunk(A_master, chunk_size=1)

    W_shape = (num_embeddings, embedding_dim)
    W_master = torch.randn(W_shape, dtype=dtype, device=device)
    W = broadcast_tensor_chunk(W_master, chunk_size=1)
    W.requires_grad = True

    # replace the torch nn.Parameters with ColoTensor
    sharded_weight = ColoTensor.init_from_torch_tensor(W)
    parallel_action_list = [
        ParallelAction(priority=1, compute_pattern=ComputePattern.TP1DRow_Embedding, 
        parallel_mode=ParallelMode.PARALLEL_1D)
    ]
    spec = TensorSpec(parallel_action_list)
    sharded_weight.set_spec(spec) # reshard
    replace_parameter_add_grad(layer, sharded_weight)
    out = layer(A)

    replace_parameter_add_grad(layer_master, W_master)
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

def run_dist(rank, world_size, port):
    config = dict(parallel=dict(tensor=dict(mode="1d", size=world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_embedding_tp1d_col_test()
    run_embedding_tp1d_row_test()

@pytest.mark.dist
@parameterize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_embedding_1d(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_embedding_1d()
