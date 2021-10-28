#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Any, Tuple

import torch
import torch.distributed as dist
from colossalai.communication import all_gather, reduce_scatter, scatter
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import empty_cache, get_current_device
from torch import Tensor


class Matmul_AB_3D(torch.autograd.Function):
    """Matrix multiplication for :math:`C = AB`
    """
    @staticmethod
    def forward(ctx: Any,
                A: Tensor,
                B: Tensor,
                depth: int,
                input_parallel_mode: ParallelMode,
                weight_parallel_mode: ParallelMode,
                output_parallel_mode: ParallelMode,
                input_dim: int = 0,
                weight_dim: int = -1,
                output_dim: int = 0) -> Tensor:
        # A: [m/q^2, n, k/q]
        # B: [k/q, h/q^2]
        # C: [m/q^2, n, h/q]
        empty_cache()
        ctx.save_for_backward(A, B)

        assert A.shape[-1] == B.shape[0], \
            'Invalid shapes: A={}, B={}.'.format(A.shape, B.shape)

        A_temp = all_gather(A, input_dim, input_parallel_mode)
        B_temp = all_gather(B, weight_dim, weight_parallel_mode)

        C = torch.matmul(A_temp, B_temp)
        out = reduce_scatter(C, output_dim, output_parallel_mode)

        ctx.depth = depth
        ctx.A_group_parallel_mode = input_parallel_mode
        ctx.B_group_parallel_mode = weight_parallel_mode
        ctx.C_group_parallel_mode = output_parallel_mode
        ctx.A_dim = input_dim
        ctx.B_dim = weight_dim
        ctx.C_dim = output_dim

        return out

    @staticmethod
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors
        with torch.no_grad():
            A_grad = Matmul_ABT_3D.apply(output_grad, B, ctx.depth,
                                         ctx.C_group_parallel_mode,
                                         ctx.B_group_parallel_mode,
                                         ctx.A_group_parallel_mode, ctx.C_dim,
                                         ctx.B_dim, ctx.A_dim)
            B_grad = Matmul_ATB_3D.apply(A, output_grad, ctx.depth,
                                         ctx.A_group_parallel_mode,
                                         ctx.C_group_parallel_mode,
                                         ctx.B_group_parallel_mode, ctx.A_dim,
                                         ctx.C_dim, ctx.B_dim)
        return A_grad, B_grad, None, None, None, None, None, None, None


class Matmul_ABT_3D(torch.autograd.Function):
    """Matrix multiplication for :math:`C = AB^T`
    """
    @staticmethod
    def forward(ctx: Any,
                A: Tensor,
                B: Tensor,
                depth: int,
                input_parallel_mode: ParallelMode,
                weight_parallel_mode: ParallelMode,
                output_parallel_mode: ParallelMode,
                input_dim: int = 0,
                weight_dim: int = -1,
                output_dim: int = 0) -> Tensor:
        # A: [m/q^2, n, h/q]
        # B: [k/q, h/q^2]
        # C: [m/q^2, n, k/q]
        empty_cache()
        ctx.save_for_backward(A, B)

        A_temp = all_gather(A, input_dim, input_parallel_mode)
        B_temp = all_gather(B, weight_dim, weight_parallel_mode)

        C = torch.matmul(A_temp, B_temp.transpose(0, 1))
        out = reduce_scatter(C, output_dim, output_parallel_mode)

        ctx.depth = depth
        ctx.A_group_parallel_mode = input_parallel_mode
        ctx.B_group_parallel_mode = weight_parallel_mode
        ctx.C_group_parallel_mode = output_parallel_mode
        ctx.A_dim = input_dim
        ctx.B_dim = weight_dim
        ctx.C_dim = output_dim

        return out

    @staticmethod
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors
        with torch.no_grad():
            A_grad = Matmul_AB_3D.apply(output_grad, B, ctx.depth,
                                        ctx.C_group_parallel_mode,
                                        ctx.B_group_parallel_mode,
                                        ctx.A_group_parallel_mode, ctx.C_dim,
                                        ctx.B_dim, ctx.A_dim)
            B_grad = Matmul_ATB_3D.apply(output_grad, A, ctx.depth,
                                         ctx.C_group_parallel_mode,
                                         ctx.A_group_parallel_mode,
                                         ctx.B_group_parallel_mode, ctx.C_dim,
                                         ctx.A_dim, ctx.B_dim)
        return A_grad, B_grad, None, None, None, None, None, None, None


class Matmul_ATB_3D(torch.autograd.Function):
    """Matrix multiplication for :math:`C = A^TB`
    """
    @staticmethod
    def forward(ctx: Any,
                A: Tensor,
                B: Tensor,
                depth: int,
                input_parallel_mode: ParallelMode,
                weight_parallel_mode: ParallelMode,
                output_parallel_mode: ParallelMode,
                input_dim: int = 0,
                weight_dim: int = 0,
                output_dim: int = -1) -> Tensor:
        # A: [m/q^2, n, k/q]
        # B: [m/q^2, n, h/q]
        # C: [k/q, h/q^2]
        empty_cache()
        ctx.save_for_backward(A, B)

        A_temp = all_gather(A, input_dim, input_parallel_mode)
        A_temp = A_temp.reshape(-1, A.shape[-1])
        B_temp = all_gather(B, weight_dim, weight_parallel_mode)
        B_temp = B_temp.reshape(-1, B.shape[-1])

        C = torch.matmul(A_temp.transpose(0, 1), B_temp)
        out = reduce_scatter(C, output_dim, output_parallel_mode)

        ctx.depth = depth
        ctx.A_group_parallel_mode = input_parallel_mode
        ctx.B_group_parallel_mode = weight_parallel_mode
        ctx.C_group_parallel_mode = output_parallel_mode
        ctx.A_dim = input_dim
        ctx.B_dim = weight_dim
        ctx.C_dim = output_dim

        return out

    @staticmethod
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors
        with torch.no_grad():
            A_grad = Matmul_ABT_3D.apply(B, output_grad, ctx.depth,
                                         ctx.B_group_parallel_mode,
                                         ctx.C_group_parallel_mode,
                                         ctx.A_group_parallel_mode, ctx.B_dim,
                                         ctx.C_dim, ctx.A_dim)
            B_grad = Matmul_AB_3D.apply(A, output_grad, ctx.depth,
                                        ctx.A_group_parallel_mode,
                                        ctx.C_group_parallel_mode,
                                        ctx.B_group_parallel_mode, ctx.A_dim,
                                        ctx.C_dim, ctx.B_dim)
        return A_grad, B_grad, None, None, None, None, None, None, None


class Add_3D(torch.autograd.Function):
    """Matrix add bias: :math:`C = A + b`
    """
    @staticmethod
    def forward(ctx: Any, input_: Tensor, bias: Tensor, depth: int,
                input_parallel_mode: ParallelMode,
                weight_parallel_mode: ParallelMode,
                output_parallel_mode: ParallelMode) -> Tensor:
        # input: [m/q^2, n, h/q]
        # bias: [h/q^2]
        ranks_in_group = gpc.get_ranks_in_group(input_parallel_mode)
        src_rank = ranks_in_group[gpc.get_local_rank(output_parallel_mode)]
        bias_temp = bias.clone()
        dist.broadcast(bias_temp,
                       src=src_rank,
                       group=gpc.get_group(input_parallel_mode))
        # [h/q]
        bias_temp = all_gather(bias_temp, -1, weight_parallel_mode)

        out = input_ + bias_temp

        ctx.depth = depth
        ctx.src_rank = src_rank
        ctx.A_group_parallel_mode = input_parallel_mode
        ctx.B_group_parallel_mode = weight_parallel_mode
        ctx.C_group_parallel_mode = output_parallel_mode

        return out

    @staticmethod
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        # output_grad: [m/q^2, n, h/q]
        with torch.no_grad():
            # [h/q]
            grad = torch.sum(output_grad,
                             dim=tuple(range(len(output_grad.shape))[:-1]))
            bias_grad = reduce_scatter(grad, -1, ctx.B_group_parallel_mode)
            dist.reduce(bias_grad,
                        dst=ctx.src_rank,
                        group=gpc.get_group(ctx.A_group_parallel_mode))
            if gpc.get_local_rank(
                ctx.A_group_parallel_mode) != gpc.get_local_rank(
                    ctx.C_group_parallel_mode):
                bias_grad = None
        return output_grad, bias_grad, None, None, None, None


class Mul_3D(torch.autograd.Function):
    """Matrix multiplication for :math:`C = A * b`
    """
    @staticmethod
    def forward(ctx: Any, input_: Tensor, bias: Tensor, depth: int,
                input_parallel_mode: ParallelMode,
                weight_parallel_mode: ParallelMode,
                output_parallel_mode: ParallelMode) -> Tensor:
        # input: [m/q^2, n, h/q]
        # bias: [h/q^2]
        ranks_in_group = gpc.get_ranks_in_group(input_parallel_mode)
        src_rank = ranks_in_group[gpc.get_local_rank(output_parallel_mode)]
        # [h/q^2]
        bias_temp = bias.clone()
        dist.broadcast(bias_temp,
                       src=src_rank,
                       group=gpc.get_group(input_parallel_mode))
        # [h/q]
        bias_temp = all_gather(bias_temp, -1, weight_parallel_mode)

        empty_cache()
        ctx.save_for_backward(input_, bias_temp)

        out = torch.mul(input_, bias_temp)

        ctx.depth = depth
        ctx.src_rank = src_rank
        ctx.A_group_parallel_mode = input_parallel_mode
        ctx.B_group_parallel_mode = weight_parallel_mode
        ctx.C_group_parallel_mode = output_parallel_mode

        return out

    @staticmethod
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        # output_grad: [m/q^2, n, h/q]
        with torch.no_grad():
            input_, bias = ctx.saved_tensors
            # [m/q^2, n, h/q]
            input_grad = torch.mul(output_grad, bias)
            # [h/q]
            grad = torch.mul(output_grad, input_)
            grad = torch.sum(grad,
                             dim=tuple(range(len(output_grad.shape))[:-1]))
            bias_grad = reduce_scatter(grad, -1, ctx.B_group_parallel_mode)
            dist.reduce(bias_grad,
                        dst=ctx.src_rank,
                        group=gpc.get_group(ctx.A_group_parallel_mode))
            if gpc.get_local_rank(
                ctx.A_group_parallel_mode) != gpc.get_local_rank(
                    ctx.C_group_parallel_mode):
                bias_grad = None
        return input_grad, bias_grad, None, None, None, None


class Sum_3D(torch.autograd.Function):
    """Compute the sum of input tensors
    """
    @staticmethod
    def forward(ctx: Any,
                input_: Tensor,
                dim: int,
                depth: int,
                parallel_mode: ParallelMode,
                keepdim: bool = False) -> Tensor:
        # input: [m/q^2, n, h/q]
        out = torch.sum(input_, dim=dim, keepdim=keepdim)
        dist.all_reduce(out, group=gpc.get_group(parallel_mode))

        ctx.input_shape = input_.shape
        ctx.depth = depth
        ctx.group = parallel_mode
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        with torch.no_grad():
            output_grad = output_grad.contiguous()
            dist.all_reduce(output_grad, group=gpc.get_group(ctx.group))
            if len(output_grad.shape) < len(ctx.input_shape):
                output_grad = torch.unsqueeze(output_grad, ctx.dim)
            dims = [1 for _ in range(len(output_grad.shape))]
            dims[ctx.dim] = ctx.input_shape[ctx.dim]
            input_grad = output_grad.repeat(tuple(dims))
        return input_grad, None, None, None, None, None


class Reduce_3D(torch.autograd.Function):
    """Reduce input tensors
    """
    @staticmethod
    def forward(ctx: Any, input_: Tensor, depth: int,
                parallel_mode: ParallelMode) -> Tensor:
        dist.all_reduce(input_, group=gpc.get_group(parallel_mode))
        return input_.clone()

    @staticmethod
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        return output_grad, None, None


class Slice_3D(torch.autograd.Function):
    """Slice input tensor
    """
    @staticmethod
    def forward(ctx: Any, input_: Tensor, dim: int, depth: int,
                parallel_mode: ParallelMode) -> Tensor:
        rank = gpc.get_local_rank(parallel_mode)
        out = torch.chunk(input_, depth, dim=dim)[rank].contiguous()

        ctx.depth = depth
        ctx.parallel_mode = parallel_mode
        ctx.dim = dim
        ctx.input_shape = input_.shape

        return out

    @staticmethod
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        with torch.no_grad():
            input_grad = all_gather(output_grad, ctx.dim, ctx.parallel_mode)
            input_grad.reshape(ctx.input_shape)
        return input_grad, None, None, None
