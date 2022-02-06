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


def check_embed():
    rank = torch.distributed.get_rank()
    device = get_current_device()
    logger = get_dist_logger()
    dtype = torch.float32

    input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
    weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
    output_parallel_mode = get_parallel_mode_from_env(OUTPUT_GROUP_3D)

    j = A_rank = global_context.get_local_rank(input_parallel_mode)
    i = B_rank = global_context.get_local_rank(weight_parallel_mode)
    k = C_rank = global_context.get_local_rank(output_parallel_mode)

    layer = PatchEmbedding3D(IMG_SIZE, 4, 3, HIDDEN_SIZE, dtype=dtype)
    torch.nn.init.ones_(layer.cls_token)
    torch.nn.init.ones_(layer.pos_embed)
    layer = layer.to(device)

    layer_master = VanillaPatchEmbedding(IMG_SIZE, 4, 3, HIDDEN_SIZE, dtype=dtype)
    torch.nn.init.ones_(layer_master.cls_token)
    torch.nn.init.ones_(layer_master.pos_embed)
    layer_master = layer_master.to(device)

    proj_weight_master = layer_master.weight.data
    torch.distributed.broadcast(proj_weight_master, src=0)
    proj_weight = torch.chunk(proj_weight_master, DEPTH, dim=0)[k]
    layer.weight.data.copy_(proj_weight)
    proj_bias_master = layer_master.bias.data
    torch.distributed.broadcast(proj_bias_master, src=0)
    proj_bias = torch.chunk(proj_bias_master, DEPTH)[k]
    layer.bias.data.copy_(proj_bias)

    A_shape = (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()
    A.requires_grad = True

    fwd_start = time.time()
    out = layer(A)
    torch.cuda.synchronize()
    fwd_end = time.time()
    print_rank_0(
        'embedding forward: pass | {0} --> {1} | {2:.3f} s'.format(tuple(A.shape), tuple(out.shape),
                                                                   fwd_end - fwd_start), logger)

    A_master = A_master.clone()
    A_master.requires_grad = True
    C_master = layer_master(A_master)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[k]
    C = torch.chunk(C, DEPTH, dim=0)[j]
    logger.info('Rank {} embed forward: {}'.format(rank, check_equal(out, C)))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_current_device())
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[k]
    grad = torch.chunk(grad, DEPTH, dim=0)[j]
    grad = grad.clone()
    bwd_start = time.time()
    out.backward(grad)
    torch.cuda.synchronize()
    bwd_end = time.time()
    print_rank_0('embedding backward: pass | {:.3f} s'.format(bwd_end - bwd_start), logger)

    grad_master = grad_master.clone()
    C_master.backward(grad_master)

    cls_grad_master = layer_master.cls_token.grad
    cls_grad = torch.chunk(cls_grad_master, DEPTH, dim=-1)[k]
    logger.info('Rank {} embed backward (cls_grad): {}'.format(rank, check_equal(cls_grad, layer.cls_token.grad)))

    pos_grad_master = layer_master.pos_embed.grad
    pos_grad = torch.chunk(pos_grad_master, DEPTH, dim=-1)[k]
    logger.info('Rank {} embed backward (pos_embed_grad): {}'.format(rank, check_equal(pos_grad, layer.pos_embed.grad)))

    B_grad = layer_master.weight.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[k]
    if j == k:
        logger.info('Rank {} embed backward (proj_weight_grad): {}'.format(rank, check_equal(B_grad,
                                                                                             layer.weight.grad)))
    else:
        logger.info('Rank {} embed backward (proj_weight_grad): {}'.format(rank, layer.weight.grad is None))

    bias_grad = layer_master.bias.grad
    bias_grad = torch.chunk(bias_grad, DEPTH)[k]
    logger.info('Rank {} embed backward (proj_bias_grad): {}'.format(rank, check_equal(bias_grad, layer.bias.grad)))

    return fwd_end - fwd_start, bwd_end - bwd_start


def check_loss():
    rank = torch.distributed.get_rank()
    logger = get_dist_logger()
    device = get_current_device()
    dtype = torch.float32

    input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
    weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
    output_parallel_mode = get_parallel_mode_from_env(OUTPUT_GROUP_3D)

    j = A_rank = global_context.get_local_rank(input_parallel_mode)
    i = B_rank = global_context.get_local_rank(weight_parallel_mode)
    k = C_rank = global_context.get_local_rank(output_parallel_mode)

    criterion = CrossEntropyLoss3D()
    criterion_master = torch.nn.CrossEntropyLoss()

    out_shape = (BATCH_SIZE, NUM_CLASSES)
    out_master = torch.randn(out_shape, dtype=dtype, device=device)
    target_master = torch.randint(NUM_CLASSES, (BATCH_SIZE, ), dtype=torch.long, device=device)
    torch.distributed.broadcast(out_master, src=0)
    torch.distributed.broadcast(target_master, src=0)
    out = torch.chunk(out_master, DEPTH, dim=0)[i]
    out = torch.chunk(out, DEPTH, dim=0)[j]
    out = out.clone()
    out.requires_grad = True

    fwd_start = time.time()
    loss = criterion(out, target_master)
    fwd_end = time.time()
    print_rank_0(
        'loss forward: pass | {0} --> {1} | {2:.3f} s'.format(tuple(out.shape), tuple(loss.shape), fwd_end - fwd_start),
        logger)

    out_master = out_master.clone()
    out_master.requires_grad = True
    loss_master = criterion_master(out_master, target_master)
    logger.info('Rank {} CrossEntropyLoss forward: {}'.format(rank, check_equal(loss, loss_master)))

    bwd_start = time.time()
    loss.backward()
    bwd_end = time.time()
    print_rank_0('loss backward: pass | {:.3f} s'.format(bwd_end - bwd_start), logger)

    loss_master.backward()
    out_grad = out_master.grad
    out_grad = torch.chunk(out_grad, DEPTH, dim=0)[i]
    out_grad = torch.chunk(out_grad, DEPTH, dim=0)[j]
    logger.info('Rank {} CrossEntropyLoss backward: {}'.format(rank, check_equal(out_grad, out.grad)))

    return fwd_end - fwd_start, bwd_end - bwd_start
