import torch
import torch.distributed as dist
from torch.nn import Parameter
import time
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn import Linear1D_Col, Linear1D_Row
from colossalai.utils import get_current_device, print_rank_0
from .common import HIDDEN_SIZE, DEPTH, BATCH_SIZE, SEQ_LENGTH, NUM_CLASSES, check_equal, IMG_SIZE


def check_linear_col():
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE
    OUTPUT_SIZE = 2 * HIDDEN_SIZE

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    layer = Linear1D_Col(INPUT_SIZE, OUTPUT_SIZE)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    dist.broadcast(A_master, src=0)
    A = A_master.clone()
    A.requires_grad = True

    W_shape = (OUTPUT_SIZE, INPUT_SIZE)
    W_master = torch.randn(W_shape, dtype=dtype, device=device)
    dist.broadcast(W_master, src=0)
    W = torch.chunk(W_master, DEPTH, dim=0)[i]
    W = W.clone()
    W.requires_grad = True

    B_shape = (OUTPUT_SIZE)
    B_master = torch.randn(B_shape, dtype=dtype, device=device)
    dist.broadcast(B_master, src=0)
    B = torch.chunk(B_master, DEPTH, dim=0)[i]
    B = B.clone()
    B.requires_grad = True

    layer.weight = Parameter(W)
    layer.bias = Parameter(B)
    out = layer(A)

    A_master = A_master.clone()
    A_master.requires_grad = True
    W_master = W_master.clone()
    W_master.requires_grad = True
    B_master = B_master.clone()
    B_master.requires_grad = True
    C_master = torch.matmul(A_master, W_master.transpose(0, 1)) + B_master
    C = torch.chunk(C_master, DEPTH, dim=-1)[i]

    check_equal(out, C)
    print_rank_0('linear_col forward: pass')

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_current_device())
    dist.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=-1)[i]
    grad = grad.clone()
    out.backward(grad)

    grad_master = grad_master.clone()
    C_master.backward(grad_master)
    A_grad = A_master.grad
    check_equal(A_grad, A.grad)

    W_grad = W_master.grad
    W_grad = torch.chunk(W_grad, DEPTH, dim=0)[i]
    check_equal(W_grad, layer.weight.grad)

    B_grad = B_master.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[i]
    check_equal(B_grad, layer.bias.grad)

    print_rank_0('linear_col backward: pass')


def check_linear_row():
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE
    OUTPUT_SIZE = 2 * HIDDEN_SIZE

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    layer = Linear1D_Row(OUTPUT_SIZE, INPUT_SIZE)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, OUTPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    dist.broadcast(A_master, src=0)
    A = torch.chunk(A_master, DEPTH, dim=-1)[i]
    A = A.clone()
    A.requires_grad = True

    W_shape = (INPUT_SIZE, OUTPUT_SIZE)
    W_master = torch.randn(W_shape, dtype=dtype, device=device)
    dist.broadcast(W_master, src=0)
    W = torch.chunk(W_master, DEPTH, dim=-1)[i]
    W = W.clone()
    W.requires_grad = True

    B_shape = (INPUT_SIZE)
    B_master = torch.randn(B_shape, dtype=dtype, device=device)
    dist.broadcast(B_master, src=0)
    B = B_master.clone()
    B.requires_grad = True

    layer.weight = Parameter(W)
    layer.bias = Parameter(B)
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
    print_rank_0('linear_row forward: pass')

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_current_device())
    dist.broadcast(grad_master, src=0)
    grad = grad_master.clone()
    out.backward(grad)

    grad_master = grad_master.clone()
    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, DEPTH, dim=-1)[i]
    check_equal(A_grad, A.grad)

    W_grad = W_master.grad
    W_grad = torch.chunk(W_grad, DEPTH, dim=-1)[i]
    check_equal(W_grad, layer.weight.grad)

    B_grad = B_master.grad
    check_equal(B_grad, layer.bias.grad)

    print_rank_0('linear_row backward: pass')
