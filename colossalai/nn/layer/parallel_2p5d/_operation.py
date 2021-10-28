from typing import Any, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import get_current_device, empty_cache


def get_parallel_group(parallel_mode: ParallelMode):
    return gpc.get_group(parallel_mode)


def get_global_rank():
    return gpc.get_global_rank()


def get_parallel_rank(parallel_mode: ParallelMode):
    return gpc.get_local_rank(parallel_mode)


class Matmul_AB_2p5D(torch.autograd.Function):
    """Matrix multiplication for :math:`C = AB`
    """

    @staticmethod
    def forward(ctx: Any,
                A: Tensor,
                B: Tensor,
                tesseract_dim: int,
                tesseract_dep: int,
                out_shape: Tuple[int, ...],
                row_rank: int,
                col_rank: int,
                dep_rank: int,
                row_parallel_mode: ParallelMode,
                col_parallel_mode: ParallelMode,
                dep_parallel_mode: ParallelMode,
                data_parallel_rank: int,
                pipeline_parallel_rank: int,
                pipeline_parallel_size: int,
                tensor_parallel_size: int) -> Tensor:
        # A: [b / dq, s, h / q] -> [(b * s) / dq, h / q]
        # B: [h / dq, s / q]
        # C: [b / dq, s, s / q] -> [(b * s) / dq, s / q]

        assert A.shape[-1] == B.shape[-2], \
            'Invalid shapes: A={}, B={} for AB.'.format(A.shape, B.shape)

        empty_cache()
        if ctx:
            ctx.save_for_backward(A, B)

        A_shape = A.shape
        A = A.reshape((-1, A_shape[-1]))
        B_shape = B.shape
        B = B.reshape((-1, B_shape[-1]))
        C_shape = (A.shape[0], B.shape[-1])
        C = torch.zeros(C_shape, dtype=A.dtype, device=get_current_device())

        for i in range(tesseract_dim):
            A_temp = A.clone()
            B_temp = B.clone()
            src_a = i + row_rank * tesseract_dim + dep_rank * (
                        tesseract_dim ** 2) + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + pipeline_parallel_rank * tensor_parallel_size
            dist.broadcast(A_temp, src=src_a,
                           group=get_parallel_group(row_parallel_mode))
            src_b = col_rank + i * tesseract_dim + dep_rank * (
                        tesseract_dim ** 2) + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + pipeline_parallel_rank * tensor_parallel_size
            dist.broadcast(B_temp, src=src_b,
                           group=get_parallel_group(col_parallel_mode))
            torch.addmm(C, A_temp, B_temp, out=C)

        out = C.reshape(out_shape)

        if ctx:
            ctx.tesseract_dim = tesseract_dim
            ctx.tesseract_dep = tesseract_dep
            ctx.row_rank = row_rank
            ctx.col_rank = col_rank
            ctx.dep_rank = dep_rank
            ctx.row_parallel_mode = row_parallel_mode
            ctx.col_parallel_mode = col_parallel_mode
            ctx.dep_parallel_mode = dep_parallel_mode
            ctx.A_shape = A_shape
            ctx.B_shape = B_shape
            ctx.data_parallel_rank = data_parallel_rank
            ctx.pipeline_parallel_rank = pipeline_parallel_rank
            ctx.pipeline_parallel_size = pipeline_parallel_size
            ctx.tensor_parallel_size = tensor_parallel_size

        return out

    @staticmethod
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors
        A_grad = Matmul_ABT_2p5D.forward(
            None,
            output_grad, B,
            ctx.tesseract_dim, ctx.tesseract_dep, ctx.A_shape,
            ctx.row_rank, ctx.col_rank, ctx.dep_rank,
            ctx.row_parallel_mode,
            ctx.col_parallel_mode,
            ctx.dep_parallel_mode,
            ctx.data_parallel_rank,
            ctx.pipeline_parallel_rank,
            ctx.pipeline_parallel_size,
            ctx.tensor_parallel_size
        )
        B_grad = Matmul_ATB_2p5D.forward(
            None,
            A, output_grad,
            ctx.tesseract_dim, ctx.tesseract_dep, ctx.B_shape,
            ctx.row_rank, ctx.col_rank, ctx.dep_rank,
            ctx.row_parallel_mode,
            ctx.col_parallel_mode,
            ctx.dep_parallel_mode,
            ctx.data_parallel_rank,
            ctx.pipeline_parallel_rank,
            ctx.pipeline_parallel_size,
            ctx.tensor_parallel_size
        )
        return A_grad, B_grad, None, None, None, None, None, None, None, None, None, None, None, None, None


class Matmul_ABT_2p5D(torch.autograd.Function):
    """Matrix multiplication for :math:`C = AB^T`
    """

    @staticmethod
    def forward(ctx: Any,
                A: Tensor,
                B: Tensor,
                tesseract_dim: int,
                tesseract_dep: int,
                out_shape: Tuple[int, ...],
                row_rank: int,
                col_rank: int,
                dep_rank: int,
                row_parallel_mode: ParallelMode,
                col_parallel_mode: ParallelMode,
                dep_parallel_mode: ParallelMode,
                data_parallel_rank: int,
                pipeline_parallel_rank: int,
                pipeline_parallel_size: int,
                tensor_parallel_size: int
                ) -> Tensor:

        assert A.shape[-1] == B.shape[-1], \
            'Invalid shapes: A={}, B={} for ABT.'.format(A.shape, B.shape)

        empty_cache()
        if ctx:
            ctx.save_for_backward(A, B)

        A_shape = A.shape
        A = A.reshape((-1, A_shape[-1]))
        B_shape = B.shape
        B = B.reshape((-1, B_shape[-1]))
        C_shape = (A.shape[0], B.shape[0])
        C = torch.empty(C_shape, dtype=A.dtype, device=get_current_device())

        for i in range(tesseract_dim):
            B_temp = B.clone()
            src_b = col_rank + i * tesseract_dim + dep_rank * (
                        tesseract_dim ** 2) + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                    pipeline_parallel_rank * tensor_parallel_size
            dist.broadcast(B_temp, src=src_b, group=gpc.get_group(col_parallel_mode))
            C_temp = torch.matmul(A, B_temp.transpose(0, 1))
            src_c = i + row_rank * tesseract_dim + dep_rank * (
                        tesseract_dim ** 2) + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                    pipeline_parallel_rank * tensor_parallel_size
            dist.reduce(C_temp, dst=src_c, group=gpc.get_group(row_parallel_mode))
            if i == col_rank:
                C = C_temp.clone()

        out = C.reshape(out_shape)

        if ctx:
            ctx.tesseract_dim = tesseract_dim
            ctx.tesseract_dep = tesseract_dep
            ctx.row_rank = row_rank
            ctx.col_rank = col_rank
            ctx.dep_rank = dep_rank
            ctx.row_parallel_mode = row_parallel_mode
            ctx.col_parallel_mode = col_parallel_mode
            ctx.dep_parallel_mode = dep_parallel_mode
            ctx.A_shape = A_shape
            ctx.B_shape = B_shape
            ctx.data_parallel_rank = data_parallel_rank
            ctx.pipeline_parallel_rank = pipeline_parallel_rank
            ctx.pipeline_parallel_size = pipeline_parallel_size
            ctx.tensor_parallel_size = tensor_parallel_size

        return out

    @staticmethod
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors
        A_grad = Matmul_AB_2p5D.forward(
            None,
            output_grad, B,
            ctx.tesseract_dim, ctx.tesseract_dep, ctx.A_shape,
            ctx.row_rank, ctx.col_rank, ctx.dep_rank,
            ctx.row_parallel_mode,
            ctx.col_parallel_mode,
            ctx.dep_parallel_mode,
            ctx.data_parallel_rank,
            ctx.pipeline_parallel_rank,
            ctx.pipeline_parallel_size,
            ctx.tensor_parallel_size
        )
        B_grad = Matmul_ATB_2p5D.forward(
            None,
            output_grad, A,
            ctx.tesseract_dim, ctx.tesseract_dep, ctx.B_shape,
            ctx.row_rank, ctx.col_rank, ctx.dep_rank,
            ctx.row_parallel_mode,
            ctx.col_parallel_mode,
            ctx.dep_parallel_mode,
            ctx.data_parallel_rank,
            ctx.pipeline_parallel_rank,
            ctx.pipeline_parallel_size,
            ctx.tensor_parallel_size
        )
        return A_grad, B_grad, None, None, None, None, None, None, None, None, None, None, None, None, None


class Matmul_ATB_2p5D(torch.autograd.Function):
    """Matrix multiplication for :math:`C = A^TB`
    """

    @staticmethod
    def forward(ctx: Any,
                A: Tensor,
                B: Tensor,
                tesseract_dim: int,
                tesseract_dep: int,
                out_shape: Tuple[int, ...],
                row_rank: int,
                col_rank: int,
                dep_rank: int,
                row_parallel_mode: ParallelMode,
                col_parallel_mode: ParallelMode,
                dep_parallel_mode: ParallelMode,
                data_parallel_rank: int,
                pipeline_parallel_rank: int,
                pipeline_parallel_size: int,
                tensor_parallel_size: int):

        assert A.shape[-2] == B.shape[-2], \
            'Invalid shapes: A={}, B={} for ATB.'.format(A.shape, B.shape)

        empty_cache()
        if ctx:
            ctx.save_for_backward(A, B)

        A_shape = A.shape
        A = A.reshape((-1, A_shape[-1]))
        B_shape = B.shape
        B = B.reshape((-1, B_shape[-1]))
        C_shape = (A.shape[-1], B.shape[-1])
        C = torch.empty(C_shape, dtype=A.dtype, device=get_current_device())

        for i in range(tesseract_dim):
            A_temp = A.clone()
            src_a = i + row_rank * tesseract_dim + dep_rank * (
                        tesseract_dim ** 2) + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                    pipeline_parallel_rank * tensor_parallel_size
            dist.broadcast(A_temp, src=src_a,
                           group=get_parallel_group(row_parallel_mode))
            C_temp = torch.matmul(A_temp.transpose(0, 1), B)
            src_c = col_rank + i * tesseract_dim + dep_rank * (
                        tesseract_dim ** 2) + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                    pipeline_parallel_rank * tensor_parallel_size
            dist.reduce(C_temp, dst=src_c,
                        group=get_parallel_group(col_parallel_mode))
            if i == row_rank:
                C = C_temp.clone()

        out = C.reshape(out_shape)

        if ctx:
            ctx.tesseract_dim = tesseract_dim
            ctx.tesseract_dep = tesseract_dep
            ctx.row_rank = row_rank
            ctx.col_rank = col_rank
            ctx.dep_rank = dep_rank
            ctx.row_parallel_mode = row_parallel_mode
            ctx.col_parallel_mode = col_parallel_mode
            ctx.dep_parallel_mode = dep_parallel_mode
            ctx.A_shape = A_shape
            ctx.B_shape = B_shape
            ctx.data_parallel_rank = data_parallel_rank
            ctx.pipeline_parallel_rank = pipeline_parallel_rank
            ctx.pipeline_parallel_size = pipeline_parallel_size
            ctx.tensor_parallel_size = tensor_parallel_size

        return out

    @staticmethod
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors
        A_grad = Matmul_ABT_2p5D.forward(
            None,
            B, output_grad,
            ctx.tesseract_dim, ctx.tesseract_dep, ctx.A_shape,
            ctx.row_rank, ctx.col_rank, ctx.dep_rank,
            ctx.row_parallel_mode,
            ctx.col_parallel_mode,
            ctx.dep_parallel_mode,
            ctx.data_parallel_rank,
            ctx.pipeline_parallel_rank,
            ctx.pipeline_parallel_size,
            ctx.tensor_parallel_size
        )
        B_grad = Matmul_AB_2p5D.forward(
            None,
            A, output_grad,
            ctx.tesseract_dim, ctx.tesseract_dep, ctx.B_shape,
            ctx.row_rank, ctx.col_rank, ctx.dep_rank,
            ctx.row_parallel_mode,
            ctx.col_parallel_mode,
            ctx.dep_parallel_mode,
            ctx.data_parallel_rank,
            ctx.pipeline_parallel_rank,
            ctx.pipeline_parallel_size,
            ctx.tensor_parallel_size
        )
        return A_grad, B_grad, None, None, None, None, None, None, None, None, None, None, None, None, None


class Add_Bias_2p5D(torch.autograd.Function):
    """Matrix add bias: :math:`C = A + b`
    """

    @staticmethod
    def forward(ctx: Any,
                input: Tensor,
                bias: Tensor,
                output_size_per_partition: int,
                tesseract_dim: int,
                tesseract_dep: int,
                row_rank: int,
                col_rank: int,
                dep_rank: int,
                row_parallel_mode: ParallelMode,
                col_parallel_mode: ParallelMode,
                dep_parallel_mode: ParallelMode,
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
        src_rank = col_rank + dep_rank * (
                    tesseract_dim ** 2) + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                   pipeline_parallel_rank * tensor_parallel_size
        dist.broadcast(bias_temp, src=src_rank, group=get_parallel_group(col_parallel_mode))

        ctx.row_rank = row_rank
        ctx.col_rank = col_rank
        ctx.dep_rank = dep_rank
        ctx.tesseract_dim = tesseract_dim
        ctx.tesseract_dep = tesseract_dep
        ctx.row_parallel_mode = row_parallel_mode
        ctx.col_parallel_mode = col_parallel_mode
        ctx.dep_parallel_mode = dep_parallel_mode
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
    def backward(ctx, output_grad):
        row_rank = ctx.row_rank
        col_rank = ctx.col_rank
        dep_rank = ctx.dep_rank
        tesseract_dim = ctx.tesseract_dim
        tesseract_dep = ctx.tesseract_dep
        row_parallel_mode = ctx.row_parallel_mode
        col_parallel_mode = ctx.col_parallel_mode
        dep_parallel_mode = ctx.dep_parallel_mode
        data_parallel_rank = ctx.data_parallel_rank
        pipeline_parallel_rank = ctx.pipeline_parallel_rank
        pipeline_parallel_size = ctx.pipeline_parallel_size
        tensor_parallel_size = ctx.tensor_parallel_size

        if ctx.bias:
            dst_rank = col_rank + dep_rank * (
                        tesseract_dim ** 2) + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                       pipeline_parallel_rank * tensor_parallel_size
            dist.reduce(output_grad, dst=dst_rank, group=get_parallel_group(col_parallel_mode))
            if row_rank == 0:
                return None, output_grad, None, None, None, None, None, None, None, None, None, None, None, None, None, None
            else:
                grad_tmp = torch.zeros_like(output_grad)
                return None, grad_tmp, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        else:
            reduce_dim = tuple(range(output_grad.ndim - 1))
            reduce = torch.sum(output_grad, dim=reduce_dim)
            dst_rank = col_rank + dep_rank * (
                        tesseract_dim ** 2) + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                       pipeline_parallel_rank * tensor_parallel_size
            dist.reduce(reduce, dst=dst_rank, group=get_parallel_group(col_parallel_mode))
            if row_rank == 0:
                return output_grad, reduce, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
            else:
                reduce_tmp = torch.zeros_like(reduce)
                return output_grad, reduce_tmp, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class _LayerNorm_2p5D(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any,
                input: Tensor,
                E_x: Tensor,
                Var_x: Tensor,
                hidden_size: int,
                row_parallel_mode: ParallelMode,
                col_parallel_mode: ParallelMode,
                dep_parallel_mode: ParallelMode) -> Tensor:
        input = input - E_x
        # in here, input = x - E[x], Var_x = 1 / sqrt(Var[x] + eps)
        ctx.hidden_size = hidden_size
        output = input * Var_x
        ctx.save_for_backward(output, Var_x)
        ctx.row_parallel_mode = row_parallel_mode
        ctx.col_parallel_mode = col_parallel_mode
        ctx.dep_parallel_mode = dep_parallel_mode
        return output

    @staticmethod
    def backward(ctx, output_grad):
        row_parallel_mode = ctx.row_parallel_mode
        col_parallel_mode = ctx.col_parallel_mode
        dep_parallel_mode = ctx.dep_parallel_mode
        x, Var_x = ctx.saved_tensors
        # in here, Var_x = 1 / sqrt(Var[x] + eps), x = (x - E[x]) * Var_x
        with torch.no_grad():
            output_grad_sum = torch.sum(output_grad, dim=-1, keepdim=True)
            torch.distributed.all_reduce(
                output_grad_sum, group=get_parallel_group(row_parallel_mode))
            output_grad_sum /= ctx.hidden_size

            output_grad_mul_x_sum = torch.sum(
                output_grad * x, dim=-1, keepdim=True)
            torch.distributed.all_reduce(
                output_grad_mul_x_sum, group=get_parallel_group(row_parallel_mode))
            output_grad_mul_x_sum /= ctx.hidden_size

            input_grad = output_grad.clone()
            input_grad -= x * output_grad_mul_x_sum
            input_grad -= output_grad_sum
            input_grad *= Var_x

        return input_grad, None, None, None, None, None, None


class Sum_2p5D(torch.autograd.Function):
    """Compute the sum of input tensors
    """

    @staticmethod
    def forward(ctx,
                inputs,
                dim,
                tesseract_dim,
                row_parallel_mode,
                keepdim=False):
        # input: [b/q, s, h/q]
        empty_cache()
        ctx.save_for_backward(inputs)
        # sum: [b/q, s]
        out = torch.sum(inputs, dim=dim, keepdim=keepdim)
        torch.distributed.all_reduce(
            out, group=gpc.get_group(row_parallel_mode))
        return out

    @staticmethod
    def backward(ctx, output_grad):
        with torch.no_grad():
            inputs = ctx.saved_tensors
            input_grad = torch.ones(inputs.shape, dtype=output_grad.dtype)
        return input_grad, None, None, None, None, None


class _ViT_Split_2p5D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, batch_size,
                tesseract_dim, tesseract_dep,
                xz_parallel_mode):
        # inputs: [b, s, h/q]
        # output: [b/dq, s, h/q]
        empty_cache()

        ctx.batch_size = batch_size
        ctx.tesseract_dim = tesseract_dim
        ctx.tesseract_dep = tesseract_dep
        ctx.xz_parallel_mode = xz_parallel_mode
        xz_rank = gpc.get_local_rank(xz_parallel_mode)
        output = torch.chunk(inputs, tesseract_dep *
                             tesseract_dim, dim=0)[xz_rank]
        output = output.clone()
        return output

    @staticmethod
    def backward(ctx, output_grad):
        # output_grad: [b/dq, s, h/q]
        # grads: [b, s, h/q]
        # *
        grads_shape = (ctx.batch_size,) + output_grad.shape[1:]
        grads = torch.empty(grads_shape,
                            dtype=output_grad.dtype,
                            device=get_current_device())
        dist.all_gather(list(grads.chunk(ctx.tesseract_dim * ctx.tesseract_dep, dim=0)),
                        output_grad.contiguous(),
                        group=get_parallel_group(ctx.xz_parallel_mode))
        return grads, None, None, None, None
