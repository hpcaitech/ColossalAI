from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from colossalai.communication.collective import (all_gather, all_reduce, reduce, reduce_scatter)
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import get_current_device
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd


def matmul_2d(
    a,
    b,
    summa_dim,
    out_shape,
    row_rank=None,
    col_rank=None,
    row_parallel_mode=ParallelMode.PARALLEL_2D_ROW,
    col_parallel_mode=ParallelMode.PARALLEL_2D_COL,
):
    """Matrix multiplication for 2D parallelism
    :param a: matrix :math:`A`
    :type a: torch.tensor
    :param b: matrix :math:`B`
    :type b: torch.tensor
    :param summa_dim: dimension of SUMMA fo 2D parallelism
    :type summa_dim: int
    :param out_shape: shape of output tensor
    :type out_shape: tuple
    :param row_rank: the rank of row, defaults to None
    :type row_rank: int, optional
    :param col_rank: the rank of column, defaults to None
    :type col_rank: int, optional
    :param row_parallel_mode: row parallel mode, defaults to ParallelMode.PARALLEL_2D_ROW
    :type row_parallel_mode: str, optional
    :param col_parallel_mode: column parallel mode, defaults to ParallelMode.PARALLEL_2D_COL
    :type col_parallel_mode: str, optional
    :return: :math:`C = AB`
    :rtype: torch.tensor
    """
    if row_rank is None:
        row_rank = gpc.get_local_rank(col_parallel_mode)
    if col_rank is None:
        col_rank = gpc.get_local_rank(row_parallel_mode)

    data_parallel_rank = 0 if not gpc.is_initialized(ParallelMode.DATA) else gpc.get_local_rank(ParallelMode.DATA)
    pipeline_parallel_rank = 0 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_local_rank(
        ParallelMode.PIPELINE)
    pipeline_parallel_size = 1 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_world_size(
        ParallelMode.PIPELINE)
    tensor_parallel_size = summa_dim**2
    return Matmul_AB_2D(a, b, summa_dim, out_shape, row_rank, col_rank, row_parallel_mode, col_parallel_mode,
                        data_parallel_rank, pipeline_parallel_rank, pipeline_parallel_size, tensor_parallel_size)


class classifier_2d(torch.autograd.Function):
    """Matrix multiplication for :math:`C = AB`
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        A: Tensor,
        B: Tensor,
        bias: Optional[Tensor],
        summa_dim: int,
        out_shape: Tuple[int, ...],
        row_rank: int,
        col_rank: int,
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
        data_parallel_rank: int,
        pipeline_parallel_rank: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
    ) -> Tensor:

        A_shape = A.shape
        A = A.reshape((-1, A_shape[-1]))
        B_shape = B.shape
        B = B.reshape((-1, B_shape[-1]))
        B_temp = all_gather(B, -1, col_parallel_mode)
        if ctx:
            ctx.save_for_backward(A, B_temp)

        C = torch.matmul(A, B_temp.transpose(0, 1))

        C = all_reduce(C, row_parallel_mode)

        ctx.use_bias = bias is not None
        if bias is not None:
            C = C + bias

        out = C.reshape(out_shape)

        if ctx:
            ctx.summa_dim = summa_dim
            ctx.row_rank = row_rank
            ctx.col_rank = col_rank
            ctx.row_parallel_mode = row_parallel_mode
            ctx.col_parallel_mode = col_parallel_mode
            ctx.A_shape = A_shape
            ctx.B_shape = B_shape
            ctx.data_parallel_rank = data_parallel_rank
            ctx.pipeline_parallel_rank = pipeline_parallel_rank
            ctx.pipeline_parallel_size = pipeline_parallel_size
            ctx.tensor_parallel_size = tensor_parallel_size

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors

        with torch.no_grad():
            A_grad = torch.matmul(output_grad, B)
            A_grad = A_grad.reshape(ctx.A_shape)
            B_grad = torch.matmul(output_grad.reshape(-1, output_grad.shape[-1]).transpose(0, 1), A)
            B_grad = reduce_scatter(B_grad, -1, ctx.col_parallel_mode)
            B_grad = B_grad.reshape(ctx.B_shape)

            bias_grad = torch.sum(output_grad, dim=tuple(range(output_grad.ndim - 1)))
            bias_grad = all_reduce(bias_grad, ctx.col_parallel_mode)

        return A_grad, B_grad, bias_grad, None, None, None, None, None, None, None, None, None, None


class Matmul_AB_2D(torch.autograd.Function):
    """Matrix multiplication for :math:`C = AB`
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        A: Tensor,
        B: Tensor,
        summa_dim: int,
        out_shape: Tuple[int, ...],
        row_rank: int,
        col_rank: int,
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
        data_parallel_rank: int,
        pipeline_parallel_rank: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
    ) -> Tensor:
        # A: [b / q, s, h / q] -> [(b * s) / q, h / q]
        # B: [h / q, s / q]
        # C: [b / q, s, s / q] -> [(b * s) / q, s / q]

        assert A.shape[-1] == B.shape[-2], \
            'Invalid shapes: A={}, B={} for AB.'.format(A.shape, B.shape)

        if ctx:
            ctx.save_for_backward(A, B)

        A_shape = A.shape
        A = A.reshape((-1, A_shape[-1]))
        B_shape = B.shape
        B = B.reshape((-1, B_shape[-1]))
        C_shape = (A.shape[0], B.shape[-1])
        C = torch.zeros(C_shape, dtype=A.dtype, device=get_current_device())

        # use circular buffer to store the communication tensor
        # 2 is enough for all cases
        A_list = [torch.empty_like(A) for _ in range(2)]
        B_list = [torch.empty_like(B) for _ in range(2)]

        row_group = gpc.get_group(row_parallel_mode)
        col_group = gpc.get_group(col_parallel_mode)

        src_a = summa_dim * row_rank + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                pipeline_parallel_rank * tensor_parallel_size
        src_b = col_rank + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                pipeline_parallel_rank * tensor_parallel_size

        opa = [None] * 2
        opb = [None] * 2

        A_list[0].copy_(A)
        B_list[0].copy_(B)
        opa[0] = dist.broadcast(A_list[0], src=src_a, group=row_group, async_op=True)
        opb[0] = dist.broadcast(B_list[0], src=src_b, group=col_group, async_op=True)
        cur = 0

        for i in range(summa_dim):
            if i != summa_dim - 1:
                A_list[1 - cur].copy_(A)
                opa[1 - cur] = dist.broadcast(A_list[1 - cur], src=src_a + 1, group=row_group, async_op=True)
                B_list[1 - cur].copy_(B)
                opb[1 - cur] = dist.broadcast(B_list[1 - cur], src=src_b + summa_dim, group=col_group, async_op=True)

            if opa[cur] is not None:
                opa[cur].wait()
            if opb[cur] is not None:
                opb[cur].wait()

            torch.addmm(C, A_list[cur], B_list[cur], out=C)
            cur = 1 - cur
            src_a += 1
            src_b += summa_dim

        out = C.reshape(out_shape)

        if ctx:
            ctx.summa_dim = summa_dim
            ctx.row_rank = row_rank
            ctx.col_rank = col_rank
            ctx.row_parallel_mode = row_parallel_mode
            ctx.col_parallel_mode = col_parallel_mode
            ctx.A_shape = A_shape
            ctx.B_shape = B_shape
            ctx.data_parallel_rank = data_parallel_rank
            ctx.pipeline_parallel_rank = pipeline_parallel_rank
            ctx.pipeline_parallel_size = pipeline_parallel_size
            ctx.tensor_parallel_size = tensor_parallel_size
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors
        with torch.no_grad():
            A_grad = Matmul_ABT_2D.apply(output_grad, B, ctx.summa_dim, ctx.A_shape, ctx.row_rank, ctx.col_rank,
                                         ctx.row_parallel_mode, ctx.col_parallel_mode, ctx.data_parallel_rank,
                                         ctx.pipeline_parallel_rank, ctx.pipeline_parallel_size,
                                         ctx.tensor_parallel_size)
            B_grad = Matmul_ATB_2D.apply(A, output_grad, ctx.summa_dim, ctx.B_shape, ctx.row_rank, ctx.col_rank,
                                         ctx.row_parallel_mode, ctx.col_parallel_mode, ctx.data_parallel_rank,
                                         ctx.pipeline_parallel_rank, ctx.pipeline_parallel_size,
                                         ctx.tensor_parallel_size)
        return A_grad, B_grad, None, None, None, None, None, None, None, None, None, None


class Matmul_ABT_2D(torch.autograd.Function):
    """Matrix multiplication for :math:`C = AB^T`
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        A: Tensor,
        B: Tensor,
        summa_dim: int,
        out_shape: Tuple[int, ...],
        row_rank: int,
        col_rank: int,
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
        data_parallel_rank: int,
        pipeline_parallel_rank: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
    ) -> Tensor:

        assert A.shape[-1] == B.shape[-1], \
            'Invalid shapes: A={}, B={} for ABT.'.format(A.shape, B.shape)

        if ctx:
            ctx.save_for_backward(A, B)

        A_shape = A.shape
        A = A.reshape((-1, A_shape[-1]))
        B_shape = B.shape
        B = B.reshape((-1, B_shape[-1]))
        C_shape = (A.shape[0], B.shape[0])
        C = torch.empty(C_shape, dtype=A.dtype, device=get_current_device())

        # use circular buffer to store the communication tensor
        # 2 is enough for all cases
        B_list = [torch.empty_like(B) for _ in range(2)]
        C_list = [torch.empty_like(C) for _ in range(2)]

        row_group = gpc.get_group(row_parallel_mode)
        col_group = gpc.get_group(col_parallel_mode)

        src_b = col_rank + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                pipeline_parallel_rank * tensor_parallel_size
        src_c = summa_dim * row_rank + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                pipeline_parallel_rank * tensor_parallel_size

        opb = [None] * 2
        opr = [None] * 2

        B_list[0].copy_(B)
        opb[0] = dist.broadcast(B_list[0], src=src_b, group=col_group, async_op=True)
        cur = 0

        for i in range(summa_dim):
            if i != summa_dim - 1:
                B_list[1 - cur].copy_(B)
                opb[1 - cur] = dist.broadcast(B_list[1 - cur], src=src_b + summa_dim, group=col_group, async_op=True)

            if opr[cur] is not None:
                opr[cur].wait()
                if i - 2 == col_rank:
                    C.copy_(C_list[cur])

            if opb[cur] is not None:
                opb[cur].wait()

            torch.matmul(A, B_list[cur].transpose(0, 1), out=C_list[cur])
            opr[cur] = dist.reduce(C_list[cur], dst=src_c, group=row_group, async_op=True)
            cur = 1 - cur
            src_b += summa_dim
            src_c += 1

        for op in opr:
            op.wait()

        if summa_dim - 2 == col_rank:
            C.copy_(C_list[cur])
        if summa_dim - 1 == col_rank:
            C.copy_(C_list[1 - cur])
        out = C.reshape(out_shape)

        if ctx:
            ctx.summa_dim = summa_dim
            ctx.row_rank = row_rank
            ctx.col_rank = col_rank
            ctx.row_parallel_mode = row_parallel_mode
            ctx.col_parallel_mode = col_parallel_mode
            ctx.A_shape = A_shape
            ctx.B_shape = B_shape
            ctx.data_parallel_rank = data_parallel_rank
            ctx.pipeline_parallel_rank = pipeline_parallel_rank
            ctx.pipeline_parallel_size = pipeline_parallel_size
            ctx.tensor_parallel_size = tensor_parallel_size

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors

        with torch.no_grad():
            A_grad = Matmul_AB_2D.apply(output_grad, B, ctx.summa_dim, ctx.A_shape, ctx.row_rank, ctx.col_rank,
                                        ctx.row_parallel_mode, ctx.col_parallel_mode, ctx.data_parallel_rank,
                                        ctx.pipeline_parallel_rank, ctx.pipeline_parallel_size,
                                        ctx.tensor_parallel_size)
            B_grad = Matmul_ATB_2D.apply(output_grad, A, ctx.summa_dim, ctx.B_shape, ctx.row_rank, ctx.col_rank,
                                         ctx.row_parallel_mode, ctx.col_parallel_mode, ctx.data_parallel_rank,
                                         ctx.pipeline_parallel_rank, ctx.pipeline_parallel_size,
                                         ctx.tensor_parallel_size)
        return A_grad, B_grad, None, None, None, None, None, None, None, None, None, None


class Matmul_ATB_2D(torch.autograd.Function):
    """Matrix multiplication for :math:`C = A^TB`
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        A: Tensor,
        B: Tensor,
        summa_dim: int,
        out_shape: Tuple[int, ...],
        row_rank: int,
        col_rank: int,
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
        data_parallel_rank: int,
        pipeline_parallel_rank: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
    ) -> Tensor:

        assert A.shape[-2] == B.shape[-2], \
            'Invalid shapes: A={}, B={} for ATB.'.format(A.shape, B.shape)

        if ctx:
            ctx.save_for_backward(A, B)

        A_shape = A.shape
        A = A.reshape((-1, A_shape[-1]))
        B_shape = B.shape
        B = B.reshape((-1, B_shape[-1]))
        C_shape = (A.shape[-1], B.shape[-1])
        C = torch.empty(C_shape, dtype=A.dtype, device=get_current_device())

        # use circular buffer to store the communication tensor
        # 2 is enough for all cases
        A_list = [torch.empty_like(A) for _ in range(2)]
        C_list = [torch.empty_like(C) for _ in range(2)]

        row_group = gpc.get_group(row_parallel_mode)
        col_group = gpc.get_group(col_parallel_mode)

        src_a = summa_dim * row_rank + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                pipeline_parallel_rank * tensor_parallel_size
        src_c = col_rank + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                pipeline_parallel_rank * tensor_parallel_size

        opa = [None] * 2
        opr = [None] * 2

        A_list[0].copy_(A)
        opa[0] = dist.broadcast(A_list[0], src=src_a, group=row_group, async_op=True)
        cur = 0

        for i in range(summa_dim):
            if i != summa_dim - 1:
                A_list[1 - cur].copy_(A)
                opa[1 - cur] = dist.broadcast(A_list[1 - cur], src=src_a + 1, group=row_group, async_op=True)

            if opr[cur] is not None:
                opr[cur].wait()
                if i - 2 == row_rank:
                    C.copy_(C_list[cur])

            if opa[cur] is not None:
                opa[cur].wait()

            torch.matmul(A_list[cur].transpose(0, 1), B, out=C_list[cur])
            opr[cur] = dist.reduce(C_list[cur], dst=src_c, group=col_group, async_op=True)
            cur = 1 - cur
            src_a += 1
            src_c += summa_dim

        for op in opr:
            op.wait()

        if summa_dim - 2 == row_rank:
            C.copy_(C_list[cur])
        if summa_dim - 1 == row_rank:
            C.copy_(C_list[1 - cur])
        out = C.reshape(out_shape)

        if ctx:
            ctx.summa_dim = summa_dim
            ctx.row_rank = row_rank
            ctx.col_rank = col_rank
            ctx.row_parallel_mode = row_parallel_mode
            ctx.col_parallel_mode = col_parallel_mode
            ctx.A_shape = A_shape
            ctx.B_shape = B_shape
            ctx.data_parallel_rank = data_parallel_rank
            ctx.pipeline_parallel_rank = pipeline_parallel_rank
            ctx.pipeline_parallel_size = pipeline_parallel_size
            ctx.tensor_parallel_size = tensor_parallel_size

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors

        with torch.no_grad():
            A_grad = Matmul_ABT_2D.apply(B, output_grad, ctx.summa_dim, ctx.A_shape, ctx.row_rank, ctx.col_rank,
                                         ctx.row_parallel_mode, ctx.col_parallel_mode, ctx.data_parallel_rank,
                                         ctx.pipeline_parallel_rank, ctx.pipeline_parallel_size,
                                         ctx.tensor_parallel_size)
            B_grad = Matmul_AB_2D.apply(A, output_grad, ctx.summa_dim, ctx.B_shape, ctx.row_rank, ctx.col_rank,
                                        ctx.row_parallel_mode, ctx.col_parallel_mode, ctx.data_parallel_rank,
                                        ctx.pipeline_parallel_rank, ctx.pipeline_parallel_size,
                                        ctx.tensor_parallel_size)
        return A_grad, B_grad, None, None, None, None, None, None, None, None, None, None


class add_bias_2d(torch.autograd.Function):
    """Matrix add bias: :math:`C = A + b`
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        input_: Tensor,
        bias: Tensor,
        output_size_per_partition: int,
        row_rank: int,
        col_rank: int,
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
        skip_bias_add: bool,
        data_parallel_rank: int,
        pipeline_parallel_rank: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
    ) -> Tensor:
        bias_temp = all_gather(bias, -1, col_parallel_mode)

        ctx.row_rank = row_rank
        ctx.col_rank = col_rank
        ctx.row_parallel_mode = row_parallel_mode
        ctx.col_parallel_mode = col_parallel_mode
        ctx.bias = skip_bias_add
        ctx.data_parallel_rank = data_parallel_rank
        ctx.pipeline_parallel_rank = pipeline_parallel_rank
        ctx.pipeline_parallel_size = pipeline_parallel_size
        ctx.tensor_parallel_size = tensor_parallel_size

        if skip_bias_add:
            return bias_temp
        else:
            output = input_ + bias_temp
            return output

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        col_parallel_mode = ctx.col_parallel_mode

        if ctx.bias:
            grad = reduce_scatter(output_grad, -1, col_parallel_mode)
            return None, grad, None, None, None, None, None, None, None, None, None, None
        else:
            reduce_dim = tuple(range(output_grad.ndim - 1))
            reduce = torch.sum(output_grad, dim=reduce_dim)
            grad = reduce_scatter(reduce, -1, col_parallel_mode)
            return output_grad, grad, None, None, None, None, None, None, None, None, None, None


class layernorm_2d(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx: Any, input_: Tensor, E_x: Tensor, Var_x: Tensor, hidden_size: int, row_parallel_mode: ParallelMode,
                col_parallel_mode: ParallelMode) -> Tensor:
        input_ = input_ - E_x
        # in here, input = x - E[x], Var_x = 1 / sqrt(Var[x] + eps)
        ctx.normalized_shape = hidden_size
        output = input_ * Var_x
        ctx.save_for_backward(output, Var_x)
        ctx.row_parallel_mode = row_parallel_mode
        ctx.col_parallel_mode = col_parallel_mode
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        row_parallel_mode = ctx.row_parallel_mode
        col_parallel_mode = ctx.col_parallel_mode
        x, Var_x = ctx.saved_tensors
        # in here, Var_x = 1 / sqrt(Var[x] + eps), x = (x - E[x]) * Var_x
        output_grad_sum = torch.sum(output_grad, dim=-1, keepdim=True)
        torch.distributed.all_reduce(output_grad_sum, group=gpc.get_group(row_parallel_mode))
        output_grad_sum /= ctx.normalized_shape

        output_grad_mul_x_sum = torch.sum(output_grad * x, dim=-1, keepdim=True)
        torch.distributed.all_reduce(output_grad_mul_x_sum, group=gpc.get_group(row_parallel_mode))
        output_grad_mul_x_sum /= ctx.normalized_shape

        input_grad = output_grad.clone()
        input_grad -= x * output_grad_mul_x_sum
        input_grad -= output_grad_sum
        input_grad *= Var_x

        return input_grad, None, None, None, None, None


class all_gather_weight_2d(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx: Any, inputs: Tensor, dim: int, summa_dim: int, col_parallel_mode: ParallelMode) -> Tensor:
        ctx.dim = dim
        ctx.summa_dim = summa_dim
        ctx.row_rank = gpc.get_local_rank(col_parallel_mode)

        outputs = all_gather(inputs, dim, col_parallel_mode)
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        grad = output_grad.chunk(ctx.summa_dim, dim=ctx.dim)[ctx.row_rank]
        return grad.contiguous(), None, None, None


class SplitFirst(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx: Any, inputs: Tensor, summa_dim: int, col_parallel_mode: ParallelMode) -> Tensor:
        ctx.summa_dim = summa_dim
        ctx.batch_size = inputs.size(0)
        ctx.para_mode = col_parallel_mode
        row_rank = gpc.get_local_rank(col_parallel_mode)

        outputs = inputs.chunk(summa_dim, dim=0)[row_rank]
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        grad_shape = (ctx.batch_size, ) + output_grad.shape[1:]
        grad = torch.empty(grad_shape, dtype=output_grad.dtype, device=get_current_device())
        dist.all_gather(list(grad.chunk(ctx.summa_dim, dim=0)),
                        output_grad.contiguous(),
                        group=gpc.get_group(ctx.para_mode))
        return grad, None, None


def split_tensor_2d(input_: Tensor, dim: int = 0) -> Tensor:
    if input_.size(dim) <= 1:
        return input_
    return torch.chunk(input_, gpc.get_world_size(ParallelMode.PARALLEL_2D_COL),
                       dim=dim)[gpc.get_local_rank(ParallelMode.PARALLEL_2D_COL)].contiguous()


class reduce_by_batch_2d(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""
    @staticmethod
    def symbolic(graph, input_, reduce_mean: bool = False):
        output = all_reduce(input_, ParallelMode.PARALLEL_2D_COL)
        if reduce_mean:
            reduce_size = gpc.get_world_size(ParallelMode.PARALLEL_2D_COL)
            return output / reduce_size
        return output

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input_, reduce_mean: bool = False):
        output = all_reduce(input_, ParallelMode.PARALLEL_2D_COL)
        ctx.reduce_mean = reduce_mean
        if reduce_mean:
            reduce_size = gpc.get_world_size(ParallelMode.PARALLEL_2D_COL)
            ctx.reduce_size = reduce_size
            return output.clone() / reduce_size
        return output.clone()

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad):
        if ctx.reduce_mean:
            return output_grad / ctx.reduce_size, None
        else:
            return output_grad, None
