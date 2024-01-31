import torch

from colossalai.accelerator import get_accelerator
from colossalai.legacy.context import ParallelMode
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.nn.layer.parallel_2p5d._operation import Matmul_AB_2p5D, Matmul_ABT_2p5D, Matmul_ATB_2p5D
from colossalai.legacy.utils import print_rank_0

from .common import *


def check_AB():
    data_parallel_rank = 0 if not gpc.is_initialized(ParallelMode.DATA) else gpc.get_local_rank(ParallelMode.DATA)
    pipeline_parallel_rank = (
        0 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_local_rank(ParallelMode.PIPELINE)
    )
    pipeline_parallel_size = (
        1 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_world_size(ParallelMode.PIPELINE)
    )
    tensor_parallel_size = gpc.get_world_size(ParallelMode.TENSOR)

    dtype = torch.float
    i = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)
    j = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)
    k = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_DEP)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=get_accelerator().get_current_device())
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, TESSERACT_DIM, dim=0)[i]
    A = torch.chunk(A, TESSERACT_DIM, dim=-1)[j]
    A = A.clone()
    A.requires_grad = True

    B_shape = (HIDDEN_SIZE, 4 * HIDDEN_SIZE)
    B_master = torch.randn(B_shape, dtype=dtype, device=get_accelerator().get_current_device())
    torch.distributed.broadcast(B_master, src=0)
    B = torch.chunk(B_master, TESSERACT_DIM, dim=0)[i]
    B = torch.chunk(B, TESSERACT_DIM, dim=-1)[j]
    B = B.clone()
    B.requires_grad = True

    out_shape = (BATCH_SIZE // TESSERACT_DIM, SEQ_LENGTH, 4 * HIDDEN_SIZE // TESSERACT_DIM)
    out = Matmul_AB_2p5D.apply(
        A,
        B,
        TESSERACT_DIM,
        out_shape,
        i,
        j,
        k,
        ParallelMode.PARALLEL_2P5D_ROW,
        ParallelMode.PARALLEL_2P5D_COL,
        data_parallel_rank,
        pipeline_parallel_rank,
        pipeline_parallel_size,
        tensor_parallel_size,
    )

    (BATCH_SIZE, SEQ_LENGTH, 4 * HIDDEN_SIZE)
    A_master = A_master.clone()
    A_master.requires_grad = True
    B_master = B_master.clone()
    B_master.requires_grad = True
    C_master = torch.matmul(A_master, B_master)
    C = torch.chunk(C_master, TESSERACT_DIM, dim=0)[i]
    C = torch.chunk(C, TESSERACT_DIM, dim=-1)[j]
    # check forward correctness
    check_equal(out, C)
    print_rank_0("AB forward: pass")

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_accelerator().get_current_device())
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, TESSERACT_DIM, dim=0)[i]
    grad = torch.chunk(grad, TESSERACT_DIM, dim=-1)[j]

    out.backward(grad)

    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, TESSERACT_DIM, dim=0)[i]
    A_grad = torch.chunk(A_grad, TESSERACT_DIM, dim=-1)[j]
    # check backward correctness
    check_equal(A_grad, A.grad)

    B_grad = B_master.grad
    B_grad = torch.chunk(B_grad, TESSERACT_DIM, dim=0)[i]
    B_grad = torch.chunk(B_grad, TESSERACT_DIM, dim=-1)[j]
    # check backward correctness
    check_equal(B_grad, B.grad)
    print_rank_0("AB backward: pass")


def check_ABT():
    data_parallel_rank = 0 if not gpc.is_initialized(ParallelMode.DATA) else gpc.get_local_rank(ParallelMode.DATA)
    pipeline_parallel_rank = (
        0 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_local_rank(ParallelMode.PIPELINE)
    )
    pipeline_parallel_size = (
        1 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_world_size(ParallelMode.PIPELINE)
    )
    tensor_parallel_size = gpc.get_world_size(ParallelMode.TENSOR)

    dtype = torch.float
    device = get_accelerator().get_current_device()

    i = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)
    j = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)
    k = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_DEP)

    C_shape = (BATCH_SIZE, SEQ_LENGTH, 4 * HIDDEN_SIZE)
    C_master = torch.randn(C_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(C_master, src=0)
    C = torch.chunk(C_master, TESSERACT_DIM, dim=0)[i]
    C = torch.chunk(C, TESSERACT_DIM, dim=-1)[j]
    C = C.clone()
    C.requires_grad = True

    B_shape = (HIDDEN_SIZE, 4 * HIDDEN_SIZE)
    B_master = torch.randn(B_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(B_master, src=0)
    B = torch.chunk(B_master, TESSERACT_DIM, dim=0)[i]
    B = torch.chunk(B, TESSERACT_DIM, dim=-1)[j]
    B = B.clone()
    B.requires_grad = True

    out = Matmul_ABT_2p5D.apply(
        C,
        B,
        TESSERACT_DIM,
        (BATCH_SIZE // TESSERACT_DIM, SEQ_LENGTH, HIDDEN_SIZE // TESSERACT_DIM),
        i,
        j,
        k,
        ParallelMode.PARALLEL_2P5D_ROW,
        ParallelMode.PARALLEL_2P5D_COL,
        data_parallel_rank,
        pipeline_parallel_rank,
        pipeline_parallel_size,
        tensor_parallel_size,
    )

    (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    C_master = C_master.clone()
    C_master.requires_grad = True
    B_master = B_master.clone()
    B_master.requires_grad = True
    A_master = torch.matmul(C_master, B_master.transpose(0, 1))
    A = torch.chunk(A_master, TESSERACT_DIM, dim=0)[i]
    A = torch.chunk(A, TESSERACT_DIM, dim=-1)[j]
    check_equal(out, A)
    print_rank_0("ABT forward: pass")

    grad_shape = A_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, TESSERACT_DIM, dim=0)[i]
    grad = torch.chunk(grad, TESSERACT_DIM, dim=-1)[j]

    # backward
    out.backward(grad)

    A_master.backward(grad_master)
    C_grad = C_master.grad
    C_grad = torch.chunk(C_grad, TESSERACT_DIM, dim=0)[i]
    C_grad = torch.chunk(C_grad, TESSERACT_DIM, dim=-1)[j]
    check_equal(C_grad, C.grad)

    B_grad = B_master.grad
    B_grad = torch.chunk(B_grad, TESSERACT_DIM, dim=0)[i]
    B_grad = torch.chunk(B_grad, TESSERACT_DIM, dim=-1)[j]
    check_equal(B_grad, B.grad)
    print_rank_0("ABT backward: pass")


def check_ATB():
    data_parallel_rank = 0 if not gpc.is_initialized(ParallelMode.DATA) else gpc.get_local_rank(ParallelMode.DATA)
    pipeline_parallel_rank = (
        0 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_local_rank(ParallelMode.PIPELINE)
    )
    pipeline_parallel_size = (
        1 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_world_size(ParallelMode.PIPELINE)
    )
    tensor_parallel_size = gpc.get_world_size(ParallelMode.TENSOR)

    device = get_accelerator().get_current_device()
    dtype = torch.float

    i = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)
    j = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)
    k = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_DEP)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, TESSERACT_DIM, dim=0)[i]
    A = torch.chunk(A, TESSERACT_DIM, dim=-1)[j]
    A = A.clone()
    A.requires_grad = True

    C_shape = (BATCH_SIZE, SEQ_LENGTH, 4 * HIDDEN_SIZE)
    C_master = torch.randn(C_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(C_master, src=0)
    C = torch.chunk(C_master, TESSERACT_DIM, dim=0)[i]
    C = torch.chunk(C, TESSERACT_DIM, dim=-1)[j]
    C = C.clone()
    C.requires_grad = True

    out = Matmul_ATB_2p5D.apply(
        A,
        C,
        TESSERACT_DIM,
        (HIDDEN_SIZE // TESSERACT_DIM, 4 * HIDDEN_SIZE // TESSERACT_DIM),
        i,
        j,
        k,
        ParallelMode.PARALLEL_2P5D_ROW,
        ParallelMode.PARALLEL_2P5D_COL,
        data_parallel_rank,
        pipeline_parallel_rank,
        pipeline_parallel_size,
        tensor_parallel_size,
    )

    (HIDDEN_SIZE, 4 * HIDDEN_SIZE)
    A_master = A_master.clone()
    A_master.requires_grad = True
    C_master = C_master.clone()
    C_master.requires_grad = True
    B_master = torch.matmul(
        A_master.view(-1, A_master.shape[-1]).transpose(0, 1), C_master.view(-1, C_master.shape[-1])
    )
    B = torch.chunk(B_master, TESSERACT_DIM, dim=0)[i]
    B = torch.chunk(B, TESSERACT_DIM, dim=-1)[j]
    check_equal(out, B)
    print_rank_0("ATB forward: pass")

    grad_shape = B_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, TESSERACT_DIM, dim=0)[i]
    grad = torch.chunk(grad, TESSERACT_DIM, dim=-1)[j]

    out.backward(grad)

    B_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, TESSERACT_DIM, dim=0)[i]
    A_grad = torch.chunk(A_grad, TESSERACT_DIM, dim=-1)[j]
    check_equal(A_grad, A.grad)

    C_grad = C_master.grad
    C_grad = torch.chunk(C_grad, TESSERACT_DIM, dim=0)[i]
    C_grad = torch.chunk(C_grad, TESSERACT_DIM, dim=-1)[j]
    check_equal(C_grad, C.grad)
    print_rank_0("ATB backward: pass")
