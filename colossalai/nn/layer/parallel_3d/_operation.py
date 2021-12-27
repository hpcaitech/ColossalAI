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
