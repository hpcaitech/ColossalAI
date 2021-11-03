#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional, Tuple

import torch
from colossalai.communication import all_gather, all_reduce, reduce_scatter, broadcast, reduce
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd


class linear_3d(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx,
                input_: Tensor,
                weight: Tensor,
                bias: Optional[Tensor],
                input_parallel_mode: ParallelMode,
                weight_parallel_mode: ParallelMode,
                output_parallel_mode: ParallelMode,
                input_dim: int = 0,
                weight_dim: int = -1,
                output_dim: int = 0) -> Tensor:
        ctx.use_bias = bias is not None

        input_ = all_gather(input_, input_dim, input_parallel_mode)
        ctx.save_for_backward(input_, weight)

        output = torch.matmul(input_, weight)
        output = reduce_scatter(output, output_dim, output_parallel_mode)

        if bias is not None:
            output += bias

        ctx.input_parallel_mode = input_parallel_mode
        ctx.weight_parallel_mode = weight_parallel_mode
        ctx.output_parallel_mode = output_parallel_mode
        ctx.input_dim = input_dim
        ctx.weight_dim = weight_dim
        ctx.output_dim = output_dim
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
        input_, weight = ctx.saved_tensors
        with torch.no_grad():
            output_grad = all_gather(output_grad, ctx.output_dim, ctx.output_parallel_mode)

            async_ops = list()

            input_grad = torch.matmul(output_grad, weight.transpose(0, 1))
            input_grad, op = reduce_scatter(input_grad, ctx.input_dim, ctx.input_parallel_mode, async_op=True)
            async_ops.append(op)

            weight_grad = torch.matmul(
                input_.reshape(-1, input_.shape[-1]).transpose(0, 1), output_grad.reshape(-1, output_grad.shape[-1]))
            weight_grad, op = all_reduce(weight_grad, ctx.weight_parallel_mode, async_op=True)
            async_ops.append(op)

            if ctx.use_bias:
                bias_grad = torch.sum(output_grad, dim=tuple(range(len(output_grad.shape))[:-1]))
                bias_grad, op = all_reduce(bias_grad, ctx.weight_parallel_mode, async_op=True)
                async_ops.append(op)

            for op in async_ops:
                if op is not None:
                    op.wait()

        return input_grad, weight_grad, bias_grad, None, None, None, None, None, None


class classifier_3d(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input_: Tensor, weight: Tensor, bias: Optional[Tensor], input_parallel_mode: ParallelMode,
                weight_parallel_mode: ParallelMode, output_parallel_mode: ParallelMode) -> Tensor:
        ctx.use_bias = bias is not None

        ranks_in_group = gpc.get_ranks_in_group(input_parallel_mode)
        src_rank = ranks_in_group[gpc.get_local_rank(output_parallel_mode)]
        weight = broadcast(weight, src_rank, input_parallel_mode)
        ctx.save_for_backward(input_, weight)

        output = torch.matmul(input_, weight.transpose(0, 1))
        output = all_reduce(output, output_parallel_mode)

        if bias is not None:
            output += bias

        ctx.src_rank = src_rank
        ctx.input_parallel_mode = input_parallel_mode
        ctx.weight_parallel_mode = weight_parallel_mode
        ctx.output_parallel_mode = output_parallel_mode
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
        input_, weight = ctx.saved_tensors
        with torch.no_grad():
            async_ops = list()

            weight_grad = torch.matmul(
                output_grad.reshape(-1, output_grad.shape[-1]).transpose(0, 1), input_.reshape(-1, input_.shape[-1]))
            weight_grad = reduce(weight_grad, ctx.src_rank, ctx.input_parallel_mode)
            if gpc.get_local_rank(ctx.input_parallel_mode) == gpc.get_local_rank(ctx.output_parallel_mode):
                weight_grad, op = all_reduce(weight_grad, ctx.weight_parallel_mode, async_op=True)
                async_ops.append(op)
            else:
                weight_grad = None

            if ctx.use_bias:
                bias_grad = torch.sum(output_grad, dim=tuple(range(len(output_grad.shape))[:-1]))
                bias_grad = all_reduce(bias_grad, ctx.input_parallel_mode)
                bias_grad, op = all_reduce(bias_grad, ctx.weight_parallel_mode, async_op=True)
                async_ops.append(op)

            input_grad = torch.matmul(output_grad, weight)

            for op in async_ops:
                if op is not None:
                    op.wait()

        return input_grad, weight_grad, bias_grad, None, None, None, None, None, None


class layernorm_3d(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input_: Tensor, weight: Tensor, bias: Tensor, normalized_shape: int, eps: float,
                input_parallel_mode: ParallelMode, weight_parallel_mode: ParallelMode,
                output_parallel_mode: ParallelMode) -> Tensor:
        mean = all_reduce(torch.sum(input_, dim=-1, keepdim=True), output_parallel_mode) / normalized_shape
        mu = input_ - mean
        var = all_reduce(torch.sum(mu**2, dim=-1, keepdim=True), output_parallel_mode) / normalized_shape
        sigma = torch.sqrt(var + eps)

        ctx.save_for_backward(mu, sigma, weight)

        z = mu / sigma
        output = weight * z + bias

        ctx.normalized_shape = normalized_shape
        ctx.input_parallel_mode = input_parallel_mode
        ctx.weight_parallel_mode = weight_parallel_mode
        ctx.output_parallel_mode = output_parallel_mode

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
        mu, sigma, weight = ctx.saved_tensors
        with torch.no_grad():
            bias_grad, weight_grad = output_grad, output_grad * mu / sigma
            grads = torch.stack([bias_grad, weight_grad]).contiguous()
            grads = torch.sum(grads, dim=tuple(range(len(grads.shape))[1:-1]))
            grads = all_reduce(grads, ctx.weight_parallel_mode)
            grads = all_reduce(grads, ctx.input_parallel_mode)
            bias_grad, weight_grad = grads[0], grads[1]

            dz = output_grad * weight
            dvar = dz * mu * (-0.5) * sigma**(-3)
            dvar = all_reduce(torch.sum(dvar, dim=-1, keepdim=True), ctx.output_parallel_mode)
            dmean = dz * (-1 / sigma) + dvar * -2 * mu / ctx.normalized_shape
            dmean = all_reduce(torch.sum(dmean, dim=-1, keepdim=True), ctx.output_parallel_mode)

            input_grad = dz / sigma + dvar * 2 * mu / \
                ctx.normalized_shape + dmean / ctx.normalized_shape

        return input_grad, weight_grad, bias_grad, None, None, None, None, None


# class reduce_3d(torch.autograd.Function):
#     """Reduce input tensors
#     """
#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float16)
#     def forward(ctx, input_: Tensor, parallel_mode: ParallelMode) -> Tensor:
#         output = all_reduce(input_, parallel_mode)
#         return output.clone()

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
#         return output_grad, None, None

# class gather_3d(torch.autograd.Function):
#     """Reduce input tensors
#     """
#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float16)
#     def forward(ctx, input_: Tensor, dim: int, parallel_mode: ParallelMode) -> Tensor:
#         output = all_gather(input_, dim, parallel_mode)
#         ctx.dim = dim
#         ctx.depth = gpc.get_world_size(parallel_mode)
#         ctx.rank = gpc.get_local_rank(parallel_mode)
#         return torch.cat(output, dim=dim)

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
#         input_grad = torch.chunk(output_grad, ctx.depth, dim=ctx.dim)[ctx.rank].contiguous()
#         return input_grad, None, None


def split_batch_3d(input_: Tensor,
                   input_parallel_mode: ParallelMode,
                   weight_parallel_mode: ParallelMode,
                   dim: int = 0) -> Tensor:
    output = torch.chunk(input_, gpc.get_world_size(weight_parallel_mode),
                         dim=dim)[gpc.get_local_rank(weight_parallel_mode)].contiguous()
    output = torch.chunk(output, gpc.get_world_size(input_parallel_mode),
                         dim=dim)[gpc.get_local_rank(input_parallel_mode)].contiguous()
    return output


class reduce_by_batch_3d(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input_: Tensor, input_parallel_mode: ParallelMode, weight_parallel_mode: ParallelMode) -> Tensor:
        output = all_reduce(input_, input_parallel_mode)
        output = all_reduce(output, weight_parallel_mode)
        return output.clone()

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
        return output_grad, None, None


class broadcast_weight_3d_from_diagonal(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input_: Tensor, input_parallel_mode: ParallelMode, weight_parallel_mode: ParallelMode,
                output_parallel_mode: ParallelMode) -> Tensor:
        ranks_in_group = gpc.get_ranks_in_group(input_parallel_mode)
        src_rank = ranks_in_group[gpc.get_local_rank(output_parallel_mode)]
        output = broadcast(input_, src_rank, input_parallel_mode)
        ctx.src_rank = src_rank
        ctx.input_parallel_mode = input_parallel_mode
        ctx.weight_parallel_mode = weight_parallel_mode
        ctx.output_parallel_mode = output_parallel_mode
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
        input_grad = reduce(output_grad, ctx.src_rank, ctx.input_parallel_mode)
        if gpc.get_local_rank(ctx.input_parallel_mode) == gpc.get_local_rank(ctx.output_parallel_mode):
            input_grad = all_reduce(input_grad, ctx.weight_parallel_mode)
        else:
            input_grad = None
        return input_grad, None, None, None


# class Matmul_AB_3D(torch.autograd.Function):
#     """Matrix multiplication for :math:`C = AB`
#     """
#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float16)
#     def forward(ctx: Any,
#                 A: Tensor,
#                 B: Tensor,
#                 depth: int,
#                 input_parallel_mode: ParallelMode,
#                 weight_parallel_mode: ParallelMode,
#                 output_parallel_mode: ParallelMode,
#                 input_dim: int = 0,
#                 weight_dim: int = -1,
#                 output_dim: int = 0) -> Tensor:
#         # A: [m/q^2, n, k/q]
#         # B: [k/q, h/q^2]
#         # C: [m/q^2, n, h/q]
#         ctx.save_for_backward(A, B)

#         assert A.shape[-1] == B.shape[0], \
#             'Invalid shapes: A={}, B={}.'.format(A.shape, B.shape)

#         A_temp = all_gather(A, input_dim, input_parallel_mode)
#         B_temp = all_gather(B, weight_dim, weight_parallel_mode)

#         C = torch.matmul(A_temp, B_temp)
#         out = reduce_scatter(C, output_dim, output_parallel_mode)

#         ctx.depth = depth
#         ctx.A_group_parallel_mode = input_parallel_mode
#         ctx.B_group_parallel_mode = weight_parallel_mode
#         ctx.C_group_parallel_mode = output_parallel_mode
#         ctx.A_dim = input_dim
#         ctx.B_dim = weight_dim
#         ctx.C_dim = output_dim

#         return out

#     @staticmethod
#     @custom_bwd
#     def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
#         A, B = ctx.saved_tensors
#         with torch.no_grad():
#             A_grad = Matmul_ABT_3D.apply(output_grad, B, ctx.depth,
#                                          ctx.C_group_parallel_mode,
#                                          ctx.B_group_parallel_mode,
#                                          ctx.A_group_parallel_mode, ctx.C_dim,
#                                          ctx.B_dim, ctx.A_dim)
#             B_grad = Matmul_ATB_3D.apply(A, output_grad, ctx.depth,
#                                          ctx.A_group_parallel_mode,
#                                          ctx.C_group_parallel_mode,
#                                          ctx.B_group_parallel_mode, ctx.A_dim,
#                                          ctx.C_dim, ctx.B_dim)
#         return A_grad, B_grad, None, None, None, None, None, None, None

# class Matmul_ABT_3D(torch.autograd.Function):
#     """Matrix multiplication for :math:`C = AB^T`
#     """
#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float16)
#     def forward(ctx: Any,
#                 A: Tensor,
#                 B: Tensor,
#                 depth: int,
#                 input_parallel_mode: ParallelMode,
#                 weight_parallel_mode: ParallelMode,
#                 output_parallel_mode: ParallelMode,
#                 input_dim: int = 0,
#                 weight_dim: int = -1,
#                 output_dim: int = 0) -> Tensor:
#         # A: [m/q^2, n, h/q]
#         # B: [k/q, h/q^2]
#         # C: [m/q^2, n, k/q]
#         ctx.save_for_backward(A, B)

#         A_temp = all_gather(A, input_dim, input_parallel_mode)
#         B_temp = all_gather(B, weight_dim, weight_parallel_mode)

#         C = torch.matmul(A_temp, B_temp.transpose(0, 1))
#         out = reduce_scatter(C, output_dim, output_parallel_mode)

#         ctx.depth = depth
#         ctx.A_group_parallel_mode = input_parallel_mode
#         ctx.B_group_parallel_mode = weight_parallel_mode
#         ctx.C_group_parallel_mode = output_parallel_mode
#         ctx.A_dim = input_dim
#         ctx.B_dim = weight_dim
#         ctx.C_dim = output_dim

#         return out

#     @staticmethod
#     @custom_bwd
#     def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
#         A, B = ctx.saved_tensors
#         with torch.no_grad():
#             A_grad = Matmul_AB_3D.apply(output_grad, B, ctx.depth,
#                                         ctx.C_group_parallel_mode,
#                                         ctx.B_group_parallel_mode,
#                                         ctx.A_group_parallel_mode, ctx.C_dim,
#                                         ctx.B_dim, ctx.A_dim)
#             B_grad = Matmul_ATB_3D.apply(output_grad, A, ctx.depth,
#                                          ctx.C_group_parallel_mode,
#                                          ctx.A_group_parallel_mode,
#                                          ctx.B_group_parallel_mode, ctx.C_dim,
#                                          ctx.A_dim, ctx.B_dim)
#         return A_grad, B_grad, None, None, None, None, None, None, None

# class Matmul_ATB_3D(torch.autograd.Function):
#     """Matrix multiplication for :math:`C = A^TB`
#     """
#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float16)
#     def forward(ctx: Any,
#                 A: Tensor,
#                 B: Tensor,
#                 depth: int,
#                 input_parallel_mode: ParallelMode,
#                 weight_parallel_mode: ParallelMode,
#                 output_parallel_mode: ParallelMode,
#                 input_dim: int = 0,
#                 weight_dim: int = 0,
#                 output_dim: int = -1) -> Tensor:
#         # A: [m/q^2, n, k/q]
#         # B: [m/q^2, n, h/q]
#         # C: [k/q, h/q^2]
#         ctx.save_for_backward(A, B)

#         A_temp = all_gather(A, input_dim, input_parallel_mode)
#         A_temp = A_temp.reshape(-1, A.shape[-1])
#         B_temp = all_gather(B, weight_dim, weight_parallel_mode)
#         B_temp = B_temp.reshape(-1, B.shape[-1])

#         C = torch.matmul(A_temp.transpose(0, 1), B_temp)
#         out = reduce_scatter(C, output_dim, output_parallel_mode)

#         ctx.depth = depth
#         ctx.A_group_parallel_mode = input_parallel_mode
#         ctx.B_group_parallel_mode = weight_parallel_mode
#         ctx.C_group_parallel_mode = output_parallel_mode
#         ctx.A_dim = input_dim
#         ctx.B_dim = weight_dim
#         ctx.C_dim = output_dim

#         return out

#     @staticmethod
#     @custom_bwd
#     def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
#         A, B = ctx.saved_tensors
#         with torch.no_grad():
#             A_grad = Matmul_ABT_3D.apply(B, output_grad, ctx.depth,
#                                          ctx.B_group_parallel_mode,
#                                          ctx.C_group_parallel_mode,
#                                          ctx.A_group_parallel_mode, ctx.B_dim,
#                                          ctx.C_dim, ctx.A_dim)
#             B_grad = Matmul_AB_3D.apply(A, output_grad, ctx.depth,
#                                         ctx.A_group_parallel_mode,
#                                         ctx.C_group_parallel_mode,
#                                         ctx.B_group_parallel_mode, ctx.A_dim,
#                                         ctx.C_dim, ctx.B_dim)
#         return A_grad, B_grad, None, None, None, None, None, None, None

# class Add_3D(torch.autograd.Function):
#     """Matrix add bias: :math:`C = A + b`
#     """
#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float16)
#     def forward(ctx: Any, input_: Tensor, bias: Tensor, depth: int,
#                 input_parallel_mode: ParallelMode,
#                 weight_parallel_mode: ParallelMode,
#                 output_parallel_mode: ParallelMode) -> Tensor:
#         # input: [m/q^2, n, h/q]
#         # bias: [h/q^2]
#         ranks_in_group = gpc.get_ranks_in_group(input_parallel_mode)
#         src_rank = ranks_in_group[gpc.get_local_rank(output_parallel_mode)]
#         bias_temp = bias.clone()
#         dist.broadcast(bias_temp,
#                        src=src_rank,
#                        group=gpc.get_group(input_parallel_mode))
#         # [h/q]
#         bias_temp = all_gather(bias_temp, -1, weight_parallel_mode)

#         out = input_ + bias_temp

#         ctx.depth = depth
#         ctx.src_rank = src_rank
#         ctx.A_group_parallel_mode = input_parallel_mode
#         ctx.B_group_parallel_mode = weight_parallel_mode
#         ctx.C_group_parallel_mode = output_parallel_mode

#         return out

#     @staticmethod
#     @custom_bwd
#     def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
#         # output_grad: [m/q^2, n, h/q]
#         with torch.no_grad():
#             # [h/q]
#             grad = torch.sum(output_grad,
#                              dim=tuple(range(len(output_grad.shape))[:-1]))
#             bias_grad = reduce_scatter(grad, -1, ctx.B_group_parallel_mode)
#             dist.reduce(bias_grad,
#                         dst=ctx.src_rank,
#                         group=gpc.get_group(ctx.A_group_parallel_mode))
#             if gpc.get_local_rank(
#                     ctx.A_group_parallel_mode) != gpc.get_local_rank(
#                         ctx.C_group_parallel_mode):
#                 bias_grad = None
#         return output_grad, bias_grad, None, None, None, None

# class Mul_3D(torch.autograd.Function):
#     """Matrix multiplication for :math:`C = A * b`
#     """
#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float16)
#     def forward(ctx: Any, input_: Tensor, bias: Tensor, depth: int,
#                 input_parallel_mode: ParallelMode,
#                 weight_parallel_mode: ParallelMode,
#                 output_parallel_mode: ParallelMode) -> Tensor:
#         # input: [m/q^2, n, h/q]
#         # bias: [h/q^2]
#         ranks_in_group = gpc.get_ranks_in_group(input_parallel_mode)
#         src_rank = ranks_in_group[gpc.get_local_rank(output_parallel_mode)]
#         # [h/q^2]
#         bias_temp = bias.clone()
#         dist.broadcast(bias_temp,
#                        src=src_rank,
#                        group=gpc.get_group(input_parallel_mode))
#         # [h/q]
#         bias_temp = all_gather(bias_temp, -1, weight_parallel_mode)

#         # empty_cache()
#         ctx.save_for_backward(input_, bias_temp)

#         out = torch.mul(input_, bias_temp)

#         ctx.depth = depth
#         ctx.src_rank = src_rank
#         ctx.A_group_parallel_mode = input_parallel_mode
#         ctx.B_group_parallel_mode = weight_parallel_mode
#         ctx.C_group_parallel_mode = output_parallel_mode

#         return out

#     @staticmethod
#     @custom_bwd
#     def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
#         # output_grad: [m/q^2, n, h/q]
#         with torch.no_grad():
#             input_, bias = ctx.saved_tensors
#             # [m/q^2, n, h/q]
#             input_grad = torch.mul(output_grad, bias)
#             # [h/q]
#             grad = torch.mul(output_grad, input_)
#             grad = torch.sum(grad,
#                              dim=tuple(range(len(output_grad.shape))[:-1]))
#             bias_grad = reduce_scatter(grad, -1, ctx.B_group_parallel_mode)
#             dist.reduce(bias_grad,
#                         dst=ctx.src_rank,
#                         group=gpc.get_group(ctx.A_group_parallel_mode))
#             if gpc.get_local_rank(
#                     ctx.A_group_parallel_mode) != gpc.get_local_rank(
#                         ctx.C_group_parallel_mode):
#                 bias_grad = None
#         return input_grad, bias_grad, None, None, None, None

# class Sum_3D(torch.autograd.Function):
#     """Compute the sum of input tensors
#     """
#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float16)
#     def forward(ctx: Any,
#                 input_: Tensor,
#                 dim: int,
#                 depth: int,
#                 parallel_mode: ParallelMode,
#                 keepdim: bool = False) -> Tensor:
#         # input: [m/q^2, n, h/q]
#         out = torch.sum(input_, dim=dim, keepdim=keepdim)
#         dist.all_reduce(out, group=gpc.get_group(parallel_mode))

#         ctx.input_shape = input_.shape
#         ctx.depth = depth
#         ctx.group = parallel_mode
#         ctx.dim = dim
#         return out

#     @staticmethod
#     @custom_bwd
#     def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
#         with torch.no_grad():
#             output_grad = output_grad.contiguous()
#             dist.all_reduce(output_grad, group=gpc.get_group(ctx.group))
#             if len(output_grad.shape) < len(ctx.input_shape):
#                 output_grad = torch.unsqueeze(output_grad, ctx.dim)
#             dims = [1 for _ in range(len(output_grad.shape))]
#             dims[ctx.dim] = ctx.input_shape[ctx.dim]
#             input_grad = output_grad.repeat(tuple(dims))
#         return input_grad, None, None, None, None, None

# class Slice_3D(torch.autograd.Function):
#     """Slice input tensor
#     """
#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float16)
#     def forward(ctx: Any, input_: Tensor, dim: int, depth: int,
#                 parallel_mode: ParallelMode) -> Tensor:
#         rank = gpc.get_local_rank(parallel_mode)
#         out = torch.chunk(input_, depth, dim=dim)[rank].contiguous()

#         ctx.depth = depth
#         ctx.parallel_mode = parallel_mode
#         ctx.dim = dim
#         ctx.input_shape = input_.shape

#         return out

#     @staticmethod
#     @custom_bwd
#     def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
#         with torch.no_grad():
#             input_grad = all_gather(output_grad, ctx.dim, ctx.parallel_mode)
#             input_grad.reshape(ctx.input_shape)
#         return input_grad, None, None, None
