import torch
import torch.distributed as dist
from torch.nn import Parameter

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn import Linear1D_Col, Linear1D_Row
# TransformerMLP1D, \
# TransformerSelfAttention1D, TransformerEncoderLayer1D
from colossalai.utils import get_current_device, print_rank_0
from common import HIDDEN_SIZE, DEPTH, BATCH_SIZE, SEQ_LENGTH, check_equal


def check_linear_col():
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE
    OUTPUT_SIZE = 2 * HIDDEN_SIZE

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    layer = Linear1D_Col(INPUT_SIZE, OUTPUT_SIZE, gather_output=True)

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
    C = C_master.clone()

    check_equal(out, C)
    print_rank_0('linear_col gather_output forward: pass')

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_current_device())
    dist.broadcast(grad_master, src=0)
    grad = grad_master.detach()
    out.backward(grad)

    C_master.backward(grad)
    A_grad = A_master.grad
    check_equal(A_grad, A.grad)

    W_grad = W_master.grad
    W_grad = torch.chunk(W_grad, DEPTH, dim=0)[i]
    check_equal(W_grad, layer.weight.grad)

    B_grad = B_master.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[i]
    check_equal(B_grad, layer.bias.grad)

    print_rank_0('linear_col gather_output backward: pass')


def check_linear_row():
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE
    OUTPUT_SIZE = 2 * HIDDEN_SIZE

    i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    layer = Linear1D_Row(OUTPUT_SIZE, INPUT_SIZE, parallel_input=False)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, OUTPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    dist.broadcast(A_master, src=0)
    A = A_master.clone()
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
    print_rank_0('linear_row no parallel_input forward: pass')

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_current_device())
    dist.broadcast(grad_master, src=0)
    grad = grad_master.detach()
    out.backward(grad)

    C_master.backward(grad)
    A_grad = A_master.grad
    check_equal(A_grad, A.grad)

    W_grad = W_master.grad
    W_grad = torch.chunk(W_grad, DEPTH, dim=-1)[i]
    check_equal(W_grad, layer.weight.grad)

    B_grad = B_master.grad
    check_equal(B_grad, layer.bias.grad)

    print_rank_0('linear_row no parallel_input backward: pass')

#
# def check_attention():
#     device = get_current_device()
#     dtype = torch.float32
#     INPUT_SIZE = HIDDEN_SIZE
#     NUM_ATTENTION_HEADS = 2
#
#     i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
#
#     layer = TransformerSelfAttention1D(
#         1,
#         HIDDEN_SIZE // NUM_ATTENTION_HEADS,
#         HIDDEN_SIZE,
#         NUM_ATTENTION_HEADS,
#         0.5
#     )
#
#     A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
#     A_master = torch.randn(A_shape, dtype=dtype, device=device)
#     torch.distributed.broadcast(A_master, src=0)
#     A = A_master.clone()
#     A.requires_grad = True
#
#     mask_shape = (BATCH_SIZE, NUM_ATTENTION_HEADS // DEPTH, SEQ_LENGTH, SEQ_LENGTH)
#     attention_mask = torch.zeros(mask_shape, dtype=dtype, device=device)
#
#     out = layer(A, attention_mask)
#     assert out.shape == (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
#     print_rank_0('self attention forward: pass')
#
#     grad_shape = out.shape
#     grad = torch.randn(grad_shape, dtype=dtype, device=device)
#
#     out.backward(grad)
#     assert A.grad.shape == A.shape
#     print_rank_0('self attention backward: pass')
#
#
# def check_mlp():
#     device = get_current_device()
#     dtype = torch.float32
#     INPUT_SIZE = HIDDEN_SIZE
#
#     i = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
#
#     layer = TransformerMLP1D(
#         HIDDEN_SIZE,
#         HIDDEN_SIZE,
#         4.0
#     )
#
#     A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
#     A_master = torch.randn(A_shape, dtype=dtype, device=device)
#     torch.distributed.broadcast(A_master, src=0)
#     A = A_master.clone()
#     A.requires_grad = True
#
#     out = layer(A)
#     assert out.shape == (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
#     print_rank_0('mlp forward: pass')
#
#     grad_shape = out.shape
#     grad = torch.randn(grad_shape, dtype=dtype, device=device)
#
#     out.backward(grad)
#     assert A.grad.shape == A.shape
#     print_rank_0('mlp backward: pass')
