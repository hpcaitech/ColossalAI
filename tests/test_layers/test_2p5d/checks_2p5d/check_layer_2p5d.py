import torch
from torch.nn import Parameter

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn import Linear2p5D, LayerNorm2p5D, Classifier2p5D
from colossalai.utils import get_current_device
from colossalai.utils import print_rank_0
from .common import *


def check_linear():
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE
    OUTPUT_SIZE = 2 * HIDDEN_SIZE

    i = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)
    j = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)
    k = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_DEP)

    layer = Linear2p5D(
        INPUT_SIZE,
        OUTPUT_SIZE,
        dtype=dtype,
        skip_bias_add=False)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, TESSERACT_DIM, dim=0)[i]
    A = torch.chunk(A, TESSERACT_DIM, dim=-1)[j]
    A = A.clone()
    A.requires_grad = True

    W_shape = (INPUT_SIZE, OUTPUT_SIZE)
    W_master = torch.randn(W_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(W_master, src=0)
    W = torch.chunk(W_master, TESSERACT_DIM, dim=0)[i]
    W = torch.chunk(W, TESSERACT_DIM, dim=-1)[j]
    W = W.clone()
    W.requires_grad = True

    B_shape = (OUTPUT_SIZE)
    B_master = torch.randn(B_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(B_master, src=0)
    B = torch.chunk(B_master, TESSERACT_DIM, dim=0)[j]
    B = B.clone()
    B.requires_grad = True

    layer.weight = Parameter(W)
    layer.bias = Parameter(B)
    out = layer(A)
    bias = layer.bias

    A_master = A_master.clone()
    A_master.requires_grad = True
    W_master = W_master.clone()
    W_master.requires_grad = True
    B_master = B_master.clone()
    B_master.requires_grad = True
    C_master = torch.matmul(A_master, W_master) + B_master
    C = torch.chunk(C_master, TESSERACT_DIM, dim=0)[i]
    C = torch.chunk(C, TESSERACT_DIM, dim=-1)[j]

    check_equal(out, C)
    print_rank_0('linear forward: pass')

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_current_device())
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, TESSERACT_DIM, dim=0)[i]
    grad = torch.chunk(grad, TESSERACT_DIM, dim=-1)[j]
    grad = grad.clone()
    out.backward(grad)

    grad_master = grad_master.clone()
    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, TESSERACT_DIM, dim=0)[i]
    A_grad = torch.chunk(A_grad, TESSERACT_DIM, dim=-1)[j]
    check_equal(A_grad, A.grad)

    W_grad = W_master.grad
    W_grad = torch.chunk(W_grad, TESSERACT_DIM, dim=0)[i]
    W_grad = torch.chunk(W_grad, TESSERACT_DIM, dim=-1)[j]
    check_equal(W_grad, layer.weight.grad)

    B_grad = B_master.grad
    B_grad = torch.chunk(B_grad, TESSERACT_DIM, dim=0)[j]
    if i == 0:
        check_equal(B_grad, layer.bias.grad)

    print_rank_0('linear backward: pass')


def check_classifier():
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE
    OUTPUT_SIZE = NUM_CLASSES

    j = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)
    i = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)

    layer = Classifier2p5D(INPUT_SIZE, OUTPUT_SIZE)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    A_master = torch.randint(5, A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, TESSERACT_DIM, dim=0)[i]
    A = torch.chunk(A, TESSERACT_DIM, dim=-1)[j]
    A = A.clone()
    A.requires_grad = True

    W_shape = (OUTPUT_SIZE, INPUT_SIZE)
    W_master = torch.randint(5, W_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(W_master, src=0)
    # W = torch.chunk(W_master, TESSERACT_DIM, dim=-1)[j]
    W = torch.chunk(W_master, TESSERACT_DIM, dim=-1)[j]
    W = torch.chunk(W, TESSERACT_DIM, dim=-1)[i]
    W = W.clone()
    layer.weight.data.copy_(W)
    # W.requires_grad = True

    B_shape = (OUTPUT_SIZE,)
    B_master = torch.randint(5, B_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(B_master, src=0)
    # B = torch.chunk(B_master, TESSERACT_DIM, dim=0)[j]
    B = B_master.clone()
    layer.bias.data.copy_(B)


    out = layer(A)

    A_master = A_master.clone()
    A_master.requires_grad = True
    W_master = W_master.clone()
    W_master.requires_grad = True
    B_master = B_master.clone()
    B_master.requires_grad = True
    C_master = torch.matmul(A_master, W_master.transpose(0, 1)) + B_master
    C = torch.chunk(C_master, TESSERACT_DIM, dim=0)[i]
    # C = torch.chunk(C, TESSERACT_DIM, dim=-1)[j]

    check_equal(out, C)
    print_rank_0('classifier forward: pass')

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_current_device())
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, TESSERACT_DIM, dim=0)[i]
    # grad = torch.chunk(grad, TESSERACT_DIM, dim=-1)[j]
    grad = grad.clone()
    out.backward(grad)

    grad_master = grad_master.clone()
    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, TESSERACT_DIM, dim=0)[i]
    A_grad = torch.chunk(A_grad, TESSERACT_DIM, dim=-1)[j]
    check_equal(A_grad, A.grad)

    W_grad = W_master.grad
    W_grad = torch.chunk(W_grad, TESSERACT_DIM, dim=-1)[j]
    W_grad = torch.chunk(W_grad, TESSERACT_DIM, dim=-1)[i]
    check_equal(W_grad, layer.weight.grad)

    B_grad = B_master.grad
    # B_grad = torch.chunk(B_grad, TESSERACT_DIM, dim=0)[j]
    # if i == 0:
    check_equal(B_grad, layer.bias.grad)

    print_rank_0('classifier backward: pass')
    

def check_layernorm():
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE
    EPS = 1e-12

    i = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)
    j = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)
    k = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_DEP)

    layernorm = LayerNorm2p5D(
        INPUT_SIZE,
        dtype=dtype)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, TESSERACT_DIM, dim=0)[i]
    A = torch.chunk(A, TESSERACT_DIM, dim=-1)[j]
    A = A.clone()
    A.requires_grad = True

    out = layernorm(A)

    A_master = A_master.clone()
    A_master.requires_grad = True
    E_master = torch.sum(A_master, dim=-1, keepdim=True)
    E_master /= INPUT_SIZE
    V_master = torch.sum(A_master * A_master, dim=-1, keepdim=True)
    V_master /= INPUT_SIZE
    V_master = V_master - E_master * E_master
    V_master = 1.0 / torch.sqrt(V_master + EPS)
    C_master = (A_master - E_master) * V_master
    C = torch.chunk(C_master, TESSERACT_DIM, dim=0)[i]
    C = torch.chunk(C, TESSERACT_DIM, dim=-1)[j]

    check_equal(out, C)
    print_rank_0('layer norm forward: pass')

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_current_device())
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, TESSERACT_DIM, dim=0)[i]
    grad = torch.chunk(grad, TESSERACT_DIM, dim=-1)[j]
    out.backward(grad)

    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, TESSERACT_DIM, dim=0)[i]
    A_grad = torch.chunk(A_grad, TESSERACT_DIM, dim=-1)[j]
    check_equal(A_grad, A.grad)
    print_rank_0('layer norm backward: pass')


# def check_attention():
#     device = get_current_device()
#     dtype = torch.float32
#     INPUT_SIZE = HIDDEN_SIZE
#     NUM_ATTENTION_HEADS = 2

#     i = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)
#     j = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)
#     k = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_DEP)

#     layer = TransformerSelfAttention2p5D(
#         HIDDEN_SIZE, NUM_ATTENTION_HEADS,
#         attention_dropout_prob=0.5,
#         hidden_dropout_prob=0.5,
#         dtype=dtype,
#     )

#     A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
#     A_master = torch.randn(A_shape, dtype=dtype, device=device)
#     torch.distributed.broadcast(A_master, src=0)
#     A = torch.chunk(A_master, TESSERACT_DIM, dim=0)[i]
#     A = torch.chunk(A, TESSERACT_DIM, dim=-1)[j]
#     A = A.clone()
#     A.requires_grad = True

#     mask_shape = (BATCH_SIZE // TESSERACT_DIM, NUM_ATTENTION_HEADS // TESSERACT_DIM, SEQ_LENGTH, SEQ_LENGTH)
#     attention_mask = torch.zeros(mask_shape, dtype=dtype, device=device)

#     out = layer(A, attention_mask)
#     assert out.shape == (BATCH_SIZE // TESSERACT_DIM, SEQ_LENGTH, INPUT_SIZE // TESSERACT_DIM)
#     print_rank_0('self attention forward: pass')

#     grad_shape = out.shape
#     grad = torch.randn(grad_shape, dtype=dtype, device=device)

#     out.backward(grad)
#     assert A.grad.shape == A.shape
#     print_rank_0('self attention backward: pass')


# def check_mlp():
#     device = get_current_device()
#     dtype = torch.float32
#     INPUT_SIZE = HIDDEN_SIZE

#     i = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)
#     j = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)
#     k = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_DEP)

#     layer = TransformerMLP2p5D(
#         HIDDEN_SIZE,
#         mlp_ratio=1,
#         dropout_prob=0.5,
#         act_func='gelu',
#         dtype=dtype,
#     )

#     A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
#     A_master = torch.randn(A_shape, dtype=dtype, device=device)
#     torch.distributed.broadcast(A_master, src=0)
#     A = torch.chunk(A_master, TESSERACT_DIM, dim=0)[i]
#     A = torch.chunk(A, TESSERACT_DIM, dim=-1)[j]
#     A = A.clone()
#     A.requires_grad = True

#     out = layer(A)
#     assert out.shape == (BATCH_SIZE // TESSERACT_DIM, SEQ_LENGTH, INPUT_SIZE // TESSERACT_DIM)
#     print_rank_0('mlp forward: pass')

#     grad_shape = out.shape
#     grad = torch.randn(grad_shape, dtype=dtype, device=device)

#     out.backward(grad)
#     assert A.grad.shape == A.shape
#     print_rank_0('mlp backward: pass')


# def check_transformerlayer():
#     device = get_current_device()
#     dtype = torch.float32
#     INPUT_SIZE = HIDDEN_SIZE
#     NUM_ATTENTION_HEADS = 2

#     i = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)
#     j = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)
#     k = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_DEP)

#     layer = TransformerLayer2p5D(
#         HIDDEN_SIZE,
#         NUM_ATTENTION_HEADS,
#         act_func='gelu',
#         attention_dropout_prob=0.5,
#         hidden_dropout_prob=0.5,
#         dtype=dtype,
#     )

#     A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
#     A_master = torch.randn(A_shape, dtype=dtype, device=device)
#     torch.distributed.broadcast(A_master, src=0)
#     A = torch.chunk(A_master, TESSERACT_DIM, dim=0)[i]
#     A = torch.chunk(A, TESSERACT_DIM, dim=-1)[j]
#     A = A.clone()
#     A.requires_grad = True

#     mask_shape = (BATCH_SIZE // TESSERACT_DIM, NUM_ATTENTION_HEADS // TESSERACT_DIM, SEQ_LENGTH, SEQ_LENGTH)
#     attention_mask = torch.zeros(mask_shape, dtype=dtype, device=device)

#     out = layer(A, attention_mask)
#     assert out.shape == (BATCH_SIZE // TESSERACT_DIM, SEQ_LENGTH, INPUT_SIZE // TESSERACT_DIM)
#     print_rank_0('transformerlayer forward: pass')

#     grad_shape = out.shape
#     grad = torch.randn(grad_shape, dtype=dtype, device=device)

#     out.backward(grad)
#     assert A.grad.shape == A.shape
#     print_rank_0('transformerlayer backward: pass')