from typing import Any, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import get_current_device
from torch.cuda.amp import custom_bwd, custom_fwd


def matmul_2d(a,
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

    data_parallel_rank = 0 if not gpc.is_initialized(
        ParallelMode.DATA) else gpc.get_local_rank(ParallelMode.DATA)
    pipeline_parallel_rank = 0 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_local_rank(
        ParallelMode.PIPELINE)
    pipeline_parallel_size = 1 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_world_size(
        ParallelMode.PIPELINE)
    tensor_parallel_size = summa_dim ** 2
    return Matmul_AB_2D(a, b, summa_dim, out_shape, row_rank, col_rank, row_parallel_mode, col_parallel_mode,
                        data_parallel_rank, pipeline_parallel_rank, pipeline_parallel_size, tensor_parallel_size
                        )


class Matmul_AB_2D(torch.autograd.Function):
    """Matrix multiplication for :math:`C = AB`
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx: Any,
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
                tensor_parallel_size: int) -> Tensor:
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

        for i in range(summa_dim):
            A_temp = A.clone()
            B_temp = B.clone()
            src_a = i + summa_dim * row_rank + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                pipeline_parallel_rank * tensor_parallel_size
            dist.broadcast(A_temp, src=src_a,
                           group=gpc.get_group(row_parallel_mode))
            src_b = col_rank + summa_dim * i + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                pipeline_parallel_rank * tensor_parallel_size
            dist.broadcast(B_temp, src=src_b,
                           group=gpc.get_group(col_parallel_mode))
            torch.addmm(C, A_temp, B_temp, out=C)

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
            A_grad = Matmul_ABT_2D.apply(
                output_grad, B,
                ctx.summa_dim, ctx.A_shape,
                ctx.row_rank, ctx.col_rank,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
                ctx.data_parallel_rank,
                ctx.pipeline_parallel_rank,
                ctx.pipeline_parallel_size,
                ctx.tensor_parallel_size
            )
            B_grad = Matmul_ATB_2D.apply(
                A, output_grad,
                ctx.summa_dim, ctx.B_shape,
                ctx.row_rank, ctx.col_rank,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
                ctx.data_parallel_rank,
                ctx.pipeline_parallel_rank,
                ctx.pipeline_parallel_size,
                ctx.tensor_parallel_size
            )
        return A_grad, B_grad, None, None, None, None, None, None, None, None, None, None


class Matmul_ABT_2D(torch.autograd.Function):
    """Matrix multiplication for :math:`C = AB^T`
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx: Any,
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
                tensor_parallel_size: int
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

        for i in range(summa_dim):
            B_temp = B.clone()
            # C_temp = torch.zeros(C_shape, dtype=C.dtype, device=get_current_device())
            src_b = col_rank + summa_dim * i + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                pipeline_parallel_rank * tensor_parallel_size
            dist.broadcast(B_temp, src=src_b,
                           group=gpc.get_group(col_parallel_mode))
            C_temp = torch.matmul(A, B_temp.transpose(0, 1))
            src_c = i + summa_dim * row_rank + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                pipeline_parallel_rank * tensor_parallel_size
            dist.reduce(C_temp, dst=src_c,
                        group=gpc.get_group(row_parallel_mode))
            if i == col_rank:
                C = C_temp.clone()

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
            A_grad = Matmul_AB_2D.apply(
                output_grad, B,
                ctx.summa_dim, ctx.A_shape,
                ctx.row_rank, ctx.col_rank,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
                ctx.data_parallel_rank,
                ctx.pipeline_parallel_rank,
                ctx.pipeline_parallel_size,
                ctx.tensor_parallel_size
            )
            B_grad = Matmul_ATB_2D.apply(
                output_grad, A,
                ctx.summa_dim, ctx.B_shape,
                ctx.row_rank, ctx.col_rank,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
                ctx.data_parallel_rank,
                ctx.pipeline_parallel_rank,
                ctx.pipeline_parallel_size,
                ctx.tensor_parallel_size
            )
        return A_grad, B_grad, None, None, None, None, None, None, None, None, None, None


class Matmul_ATB_2D(torch.autograd.Function):
    """Matrix multiplication for :math:`C = A^TB`
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx: Any,
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
                tensor_parallel_size: int
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

        for i in range(summa_dim):
            A_temp = A.clone()
            # C_temp = torch.zeros(C_shape, dtype=C.dtype, device=get_current_device())
            src_a = i + summa_dim * row_rank + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                pipeline_parallel_rank * tensor_parallel_size
            dist.broadcast(A_temp, src=src_a,
                           group=gpc.get_group(row_parallel_mode))
            C_temp = torch.matmul(A_temp.transpose(0, 1), B)
            src_c = col_rank + summa_dim * i + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                pipeline_parallel_rank * tensor_parallel_size
            dist.reduce(C_temp, dst=src_c,
                        group=gpc.get_group(col_parallel_mode))
            if i == row_rank:
                C = C_temp.clone()

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
            A_grad = Matmul_ABT_2D.apply(
                B, output_grad,
                ctx.summa_dim, ctx.A_shape,
                ctx.row_rank, ctx.col_rank,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
                ctx.data_parallel_rank,
                ctx.pipeline_parallel_rank,
                ctx.pipeline_parallel_size,
                ctx.tensor_parallel_size
            )
            B_grad = Matmul_AB_2D.apply(
                A, output_grad,
                ctx.summa_dim, ctx.B_shape,
                ctx.row_rank, ctx.col_rank,
                ctx.row_parallel_mode,
                ctx.col_parallel_mode,
                ctx.data_parallel_rank,
                ctx.pipeline_parallel_rank,
                ctx.pipeline_parallel_size,
                ctx.tensor_parallel_size
            )
        return A_grad, B_grad, None, None, None, None, None, None, None, None, None, None


class Add_Bias_2D(torch.autograd.Function):
    """Matrix add bias: :math:`C = A + b`
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx: Any,
                input: Tensor,
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
                tensor_parallel_size: int
                ) -> Tensor:
        if row_rank == 0:
            bias_temp = bias.clone()
        else:
            bias_temp = torch.zeros(
                output_size_per_partition,
                dtype=bias.dtype,
                device=get_current_device())
        src_rank = col_rank + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
            pipeline_parallel_rank * tensor_parallel_size
        dist.broadcast(bias_temp, src=src_rank,
                       group=gpc.get_group(col_parallel_mode))

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
            output = input + bias_temp
            return output

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        row_rank = ctx.row_rank
        col_rank = ctx.col_rank
        row_parallel_mode = ctx.row_parallel_mode
        col_parallel_mode = ctx.col_parallel_mode
        data_parallel_rank = ctx.data_parallel_rank
        pipeline_parallel_rank = ctx.pipeline_parallel_rank
        pipeline_parallel_size = ctx.pipeline_parallel_size
        tensor_parallel_size = ctx.tensor_parallel_size

        if ctx.bias:
            dst_rank = col_rank + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                pipeline_parallel_rank * tensor_parallel_size
            dist.reduce(output_grad, dst=dst_rank,
                        group=gpc.get_group(col_parallel_mode))
            if row_rank == 0:
                return None, output_grad, None, None, None, None, None, None, None, None, None, None
            else:
                # for compatibility with zero optimizer, no grad should be None
                grad_tmp = torch.zeros_like(output_grad)
                return None, grad_tmp, None, None, None, None, None, None, None, None, None, None
        else:
            reduce_dim = tuple(range(output_grad.ndim - 1))
            reduce = torch.sum(output_grad, dim=reduce_dim)
            dst_rank = col_rank + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                pipeline_parallel_rank * tensor_parallel_size
            dist.reduce(reduce, dst=dst_rank,
                        group=gpc.get_group(col_parallel_mode))
            if row_rank == 0:
                return output_grad, reduce, None, None, None, None, None, None, None, None, None, None
            else:
                # for compatibility with zero optimizer, no grad should be None
                reduce_tmp = torch.zeros_like(reduce)
                return output_grad, reduce_tmp, None, None, None, None, None, None, None, None, None, None


class _LayerNorm_2D(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx: Any,
                input: Tensor,
                E_x: Tensor,
                Var_x: Tensor,
                hidden_size: int,
                row_parallel_mode: ParallelMode,
                col_parallel_mode: ParallelMode) -> Tensor:
        input = input - E_x
        # in here, input = x - E[x], Var_x = 1 / sqrt(Var[x] + eps)
        ctx.normalized_shape = hidden_size
        output = input * Var_x
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
        torch.distributed.all_reduce(
            output_grad_sum, group=gpc.get_group(row_parallel_mode))
        output_grad_sum /= ctx.normalized_shape

        output_grad_mul_x_sum = torch.sum(
            output_grad * x, dim=-1, keepdim=True)
        torch.distributed.all_reduce(
            output_grad_mul_x_sum, group=gpc.get_group(row_parallel_mode))
        output_grad_mul_x_sum /= ctx.normalized_shape

        input_grad = output_grad.clone()
        input_grad -= x * output_grad_mul_x_sum
        input_grad -= output_grad_sum
        input_grad *= Var_x

        return input_grad, None, None, None, None, None


# class Sum_2D(torch.autograd.Function):
#
#     @staticmethod
#     def forward(ctx: Any,
#                 inputs: Tensor,
#                 dim: int,
#                 summa_dim: int,
#                 row_parallel_mode: ParallelMode,
#                 keepdim: bool = False) -> Tensor:
#         # input: [b/q, s, h/q]
#         empty_cache()
#         ctx.save_for_backward(inputs)
#         # sum: [b/q, s]
#         out = torch.sum(inputs, dim=dim, keepdim=keepdim)
#         torch.distributed.all_reduce(out, group=gpc.get_group(row_parallel_mode))
#         return out
#
#     @staticmethod
#     def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
#         with torch.no_grad():
#             inputs = ctx.saved_tensors
#             input_grad = torch.ones(inputs.shape, dtype=output_grad.dtype)
#         return input_grad, None, None, None, None, None


class _ViT_Split_Input_2D(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx: Any,
                inputs: Tensor,
                batch_size: int,
                summa_dim: int,
                col_parallel_mode: ParallelMode) -> Tensor:
        # inputs: [b, s, h/q]
        # output: [b/q, s, h/q]

        ctx.BATCH_SIZE = batch_size
        ctx.summa_dim = summa_dim
        ctx.col_parallel_mode = col_parallel_mode
        row_rank = gpc.get_local_rank(col_parallel_mode)
        output = torch.chunk(inputs, summa_dim, dim=0)[row_rank]
        output = output.clone()
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        # output_grad: [b/q, s, h/q]
        # grads: [b, s, h/q]
        grads_shape = (ctx.BATCH_SIZE,) + output_grad.shape[1:]
        grads = torch.empty(grads_shape,
                            dtype=output_grad.dtype,
                            device=get_current_device())
        dist.all_gather(list(grads.chunk(ctx.summa_dim, dim=0)),
                        output_grad.contiguous(),
                        group=gpc.get_group(ctx.col_parallel_mode))
        return grads, None, None, None
