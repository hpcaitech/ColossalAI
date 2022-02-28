#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional, Tuple

import torch
from colossalai.communication import (all_gather, all_reduce, broadcast, reduce, reduce_scatter)
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd
from ._utils import get_parallel_mode_from_env
from colossalai.constants import INPUT_GROUP_3D, WEIGHT_GROUP_3D


class _Linear3D(torch.autograd.Function):

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
        weight = all_gather(weight, weight_dim, weight_parallel_mode)
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
            weight_grad, op = reduce_scatter(weight_grad, ctx.weight_dim, ctx.weight_parallel_mode, async_op=True)
            async_ops.append(op)

            if ctx.use_bias:
                bias_grad = torch.sum(output_grad, dim=tuple(range(len(output_grad.shape))[:-1]))
                bias_grad, op = all_reduce(bias_grad, ctx.weight_parallel_mode, async_op=True)
                async_ops.append(op)
            else:
                bias_grad = None

            for op in async_ops:
                if op is not None:
                    op.wait()

        return input_grad, weight_grad, bias_grad, None, None, None, None, None, None


def linear_3d(input_: Tensor,
              weight: Tensor,
              bias: Optional[Tensor],
              input_parallel_mode: ParallelMode,
              weight_parallel_mode: ParallelMode,
              output_parallel_mode: ParallelMode,
              input_dim: int = 0,
              weight_dim: int = -1,
              output_dim: int = 0) -> Tensor:
    """
    Linear layer for 3D parallelism

    :param input_: matrix of input
    :type input_: torch.tensor
    :param weight: matrix of weight
    :type weight: torch.tensor
    :param bias: matrix of bias
    :type bias: torch.tensor, optional
    :param input_parallel_mode: input parallel mode
    :type input_parallel_mode: colossalai.context.parallel_mode.ParallelMode
    :param weight_parallel_mode: weight parallel mode
    :type weight_parallel_mode: colossalai.context.parallel_mode.ParallelMode
    :param output_parallel_mode: output parallel mode
    :type output_parallel_mode: colossalai.context.parallel_mode.ParallelMode
    :param input_dim: dimension of input, defaults to 0
    :type input_dim: int, optional
    :param weight_dim: dimension of weight, defaults to -1
    :type weight_dim: int, optional
    :param output_dim: dimension of output, defaults to 0
    :type output_dim: int, optional
    """
    return _Linear3D.apply(input_, weight, bias, input_parallel_mode, weight_parallel_mode, output_parallel_mode,
                           input_dim, weight_dim, output_dim)


class _Classifier3D(torch.autograd.Function):

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
            else:
                bias_grad = None

            input_grad = torch.matmul(output_grad, weight)

            for op in async_ops:
                if op is not None:
                    op.wait()

        return input_grad, weight_grad, bias_grad, None, None, None, None, None, None


def classifier_3d(input_: Tensor, weight: Tensor, bias: Optional[Tensor], input_parallel_mode: ParallelMode,
                  weight_parallel_mode: ParallelMode, output_parallel_mode: ParallelMode) -> Tensor:
    """
    3D parallel classifier

    :param input_: matrix of input
    :type input_: torch.tensor
    :param weight: matrix of weight
    :type weight: torch.tensor
    :param bias: matrix of bias
    :type bias: torch.tensor, optional
    :param input_parallel_mode: input parallel mode
    :type input_parallel_mode: colossalai.context.parallel_mode.ParallelMode
    :param weight_parallel_mode: weight parallel mode
    :type weight_parallel_mode: colossalai.context.parallel_mode.ParallelMode
    :param output_parallel_mode: output parallel mode
    :type output_parallel_mode: colossalai.context.parallel_mode.ParallelMode
    """
    return _Classifier3D.apply(input_, weight, bias, input_parallel_mode, weight_parallel_mode, output_parallel_mode)


class _Layernorm3D(torch.autograd.Function):

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


def layernorm_3d(input_: Tensor, weight: Tensor, bias: Tensor, normalized_shape: int, eps: float,
                 input_parallel_mode: ParallelMode, weight_parallel_mode: ParallelMode,
                 output_parallel_mode: ParallelMode) -> Tensor:
    """
    3D parallel Layernorm

    :param input_: input maxtrix
    :type input_: torch.tensor
    :param weight: matrix of weight
    :type weight: torch.tensor
    :param bias: matrix of bias
    :type bias: torch.tensor
    :param normalized_shape: input shape from an expected input
        of size. :math:`[* \times \text{normalized_shape}[0] \times \text{normalized_shape}[1] \times \ldots \times \text{normalized_shape}[-1]]`
        If a single integer is used, it is treated as a singleton list, and this module will
        normalize over the last dimension which is expected to be of that specific size.
    :type normalized_shape: int
    :param eps: a value added to the denominator for numerical stability
    :type eps: float
    :param input_parallel_mode: input parallel mode
    :type input_parallel_mode: colossalai.context.parallel_mode.ParallelMode
    :param weight_parallel_mode: weight parallel mode
    :type weight_parallel_mode: colossalai.context.parallel_mode.ParallelMode
    :param output_parallel_mode: output parallel mode
    :type output_parallel_mode: colossalai.context.parallel_mode.ParallelMode
    """
    return _Layernorm3D.apply(input_, weight, bias, normalized_shape, eps, input_parallel_mode, weight_parallel_mode,
                              output_parallel_mode)


def split_tensor_3d(tensor: Tensor, dim: int, parallel_mode: ParallelMode) -> Tensor:
    """Splits 3D parallel tensor in specified dimension

    :param tensor: Input tensor
    :param dim: Specified dimension in which to split
    :param parallel_mode: Parallel mode
    :param weight_parallel_mode: Weight parallel mode

    :type tensor: torch.Tensor
    :type dim: int
    :type parallel_mode: colossalai.context.parallel_mode.ParallelMode
    
    :return output: Splitted tensor
    :rtype output: torch.Tensor
    """
    if tensor.size(dim) <= 1:
        return tensor
    output = torch.chunk(tensor, gpc.get_world_size(parallel_mode),
                         dim=dim)[gpc.get_local_rank(parallel_mode)].contiguous()
    return output


def split_batch_3d(input_: Tensor,
                    dim: int = 0,
                    input_parallel_mode: ParallelMode = ParallelMode.PARALLEL_3D_INPUT,
                    weight_parallel_mode: ParallelMode = ParallelMode.PARALLEL_3D_WEIGHT) -> Tensor:
    """Splits 3D tensor in batch
    :param input_: Input tensor
    :param dim: Specified dimension in which to split
    :param input_parallel_mode: Input parallel mode
    :param weight_parallel_mode: Weight parallel mode
    :type input_: torch.Tensor
    :type dim: int, optional
    :type input_parallel_mode: colossalai.context.parallel_mode.ParallelMode, optional
    :type weight_parallel_mode: colossalai.context.parallel_mode.ParallelMode, optional
    :return output: Splitted tensor
    :rtype output: torch.Tensor
    """
    if input_.size(dim) <= 1:
        return input_
    weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
    input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
    output = torch.chunk(input_, gpc.get_world_size(weight_parallel_mode),
                         dim=dim)[gpc.get_local_rank(weight_parallel_mode)].contiguous()
    output = torch.chunk(output, gpc.get_world_size(input_parallel_mode),
                         dim=dim)[gpc.get_local_rank(input_parallel_mode)].contiguous()
    return output


class _ReduceTensor3D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, parallel_mode):
        return all_reduce(input_, parallel_mode)

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad, None


def reduce_tensor_3d(tensor: Tensor, parallel_mode: ParallelMode) -> Tensor:
    """
    All-reduce the input.
    
    :param tensor: Input tensor
    :param parallel_mode: Parallel mode
    """
    return _ReduceTensor3D.apply(tensor, parallel_mode)


class _AllGatherTensor3D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, dim, parallel_mode):
        ctx.dim = dim
        ctx.parallel_mode = parallel_mode
        output = all_gather(input_, dim, parallel_mode)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        input_grad = reduce_scatter(output_grad, ctx.dim, ctx.parallel_mode)
        return input_grad, None, None


def all_gather_tensor_3d(tensor: Tensor, dim: int, parallel_mode: ParallelMode) -> Tensor:
    """
    All-reduce the gradient in backward pass.
    
    :param tensor: Input tensor
    :param parallel_mode: Parallel mode
    """
    return _AllGatherTensor3D.apply(tensor, dim, parallel_mode)


class _ReduceScatterTensor3D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, dim, parallel_mode):
        ctx.dim = dim
        ctx.parallel_mode = parallel_mode
        return reduce_scatter(input_, dim, parallel_mode)

    @staticmethod
    def backward(ctx, output_grad):
        input_grad = all_gather(output_grad, ctx.dim, ctx.parallel_mode)
        return input_grad, None, None


def reduce_scatter_tensor_3d(tensor: Tensor, dim: int, parallel_mode: ParallelMode) -> Tensor:
    """
    Reduce-scatter the input.
    
    :param tensor: Input tensor
    :param dim: Dimension to scatter
    :param parallel_mode: Parallel mode
    """
    return _ReduceScatterTensor3D.apply(tensor, dim, parallel_mode)


class _ReduceByBatch3D(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx,
                input_: Tensor,
                input_parallel_mode: ParallelMode,
                weight_parallel_mode: ParallelMode,
                reduce_mean: bool = False) -> Tensor:
        output = all_reduce(input_, input_parallel_mode)
        output = all_reduce(output, weight_parallel_mode)
        ctx.reduce_mean = reduce_mean
        if reduce_mean:
            reduce_size = gpc.get_world_size(input_parallel_mode) * gpc.get_world_size(weight_parallel_mode)
            ctx.reduce_size = reduce_size
            return output.clone() / reduce_size
        return output.clone()

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
        if ctx.reduce_mean:
            return output_grad / ctx.reduce_size, None, None, None
        else:
            return output_grad, None, None, None


def reduce_by_batch_3d(tensor: Tensor,
                       input_parallel_mode: ParallelMode,
                       weight_parallel_mode: ParallelMode,
                       reduce_mean: bool = False) -> Tensor:
    """
    All-reduce the input from the model parallel region.

    :param input_: input maxtrix
    :type input_: torch.tensor
    :param input_parallel_mode: input parallel mode
    :type input_parallel_mode: colossalai.context.parallel_mode.ParallelMode
    :param weight_parallel_mode: weight parallel mode
    :type weight_parallel_mode: colossalai.context.parallel_mode.ParallelMode
    :param reduce_mean:  If set to ``True``, it will divide the output by (input parallel size * weight parallel size), default to False
    :type reduce_mean: int, optional
    """
    return _ReduceByBatch3D.apply(tensor, input_parallel_mode, weight_parallel_mode, reduce_mean)


class _BroadcastWeight3D_FromDiagonal(torch.autograd.Function):
    """
    broadcast weight from diagonal

    :param input_: input maxtrix
    :type input_: torch.tensor
    :param input_parallel_mode: input parallel mode
    :type input_parallel_mode: colossalai.context.parallel_mode.ParallelMode
    :param weight_parallel_mode: weight parallel mode
    :type weight_parallel_mode: colossalai.context.parallel_mode.ParallelMode
    :param weight_parallel_mode: output parallel mode
    :type weight_parallel_mode: colossalai.context.parallel_mode.ParallelMode
    """

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


def broadcast_weight_3d_from_diagonal(tensor: Tensor, input_parallel_mode: ParallelMode,
                                      weight_parallel_mode: ParallelMode, output_parallel_mode: ParallelMode) -> Tensor:
    return _BroadcastWeight3D_FromDiagonal.apply(tensor, input_parallel_mode, weight_parallel_mode,
                                                 output_parallel_mode)
