#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd

from colossalai.communication import all_gather, all_reduce, broadcast, reduce, reduce_scatter
from colossalai.constants import INPUT_GROUP_3D, WEIGHT_GROUP_3D
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc

from ._utils import get_parallel_mode_from_env, push_async_grad


class _Linear3D(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx,
        input_: Tensor,
        weight: Tensor,
        weight_id: int,
        input_parallel_mode: ParallelMode,
        weight_parallel_mode: ParallelMode,
        output_parallel_mode: ParallelMode,
    ) -> Tensor:
        ctx.weight_id = weight_id
        ctx.input_parallel_mode = input_parallel_mode
        ctx.weight_parallel_mode = weight_parallel_mode
        ctx.output_parallel_mode = output_parallel_mode

        input_ = all_gather(input_, 0, input_parallel_mode)
        weight = all_gather(weight, 0, weight_parallel_mode)
        ctx.save_for_backward(input_, weight)

        output = torch.matmul(input_, weight)
        output = reduce_scatter(output, 0, output_parallel_mode)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
        input_, weight = ctx.saved_tensors
        output_grad = all_gather(output_grad, 0, ctx.output_parallel_mode)

        input_grad = torch.matmul(output_grad, weight.transpose(0, 1))
        input_grad, input_op = reduce_scatter(input_grad, 0, ctx.input_parallel_mode, async_op=True)

        weight_grad = torch.matmul(
            input_.reshape(-1, input_.shape[-1]).transpose(0, 1), output_grad.reshape(-1, output_grad.shape[-1]))
        weight_grad, op = reduce_scatter(weight_grad, 0, ctx.weight_parallel_mode, async_op=True)
        weight_grad = push_async_grad(op, weight_grad, ctx.weight_id)

        input_op.wait()

        return input_grad, weight_grad, None, None, None, None


def linear_3d(
    input_: Tensor,
    weight: Tensor,
    input_parallel_mode: ParallelMode,
    weight_parallel_mode: ParallelMode,
    output_parallel_mode: ParallelMode,
) -> Tensor:
    r"""Linear layer for 3D parallelism.

    Args:
        input_ (:class:`torch.tensor`): input matrix.
        weight (:class:`torch.tensor`): matrix of weight.
        input_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): input parallel mode.
        weight_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): weight parallel mode.
        output_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): output parallel mode.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    return _Linear3D.apply(
        input_,
        weight,
        id(weight),
        input_parallel_mode,
        weight_parallel_mode,
        output_parallel_mode,
    )


class _Classifier3D(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx,
        input_: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        weight_id: int,
        bias_id: Optional[int],
        input_parallel_mode: ParallelMode,
        weight_parallel_mode: ParallelMode,
        output_parallel_mode: ParallelMode,
    ) -> Tensor:
        ctx.use_bias = bias is not None
        ctx.weight_id = weight_id

        src_rank = gpc.get_ranks_in_group(input_parallel_mode)[gpc.get_local_rank(output_parallel_mode)]
        weight = broadcast(weight, src_rank, input_parallel_mode)
        ctx.save_for_backward(input_, weight)

        output = torch.matmul(input_, weight.transpose(0, 1))
        output = all_reduce(output, output_parallel_mode)

        if bias is not None:
            ctx.bias_id = bias_id
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
        weight_grad = torch.matmul(
            output_grad.reshape(-1, output_grad.shape[-1]).transpose(0, 1), input_.reshape(-1, input_.shape[-1]))
        weight_grad = reduce(weight_grad, ctx.src_rank, ctx.input_parallel_mode)
        if gpc.get_local_rank(ctx.input_parallel_mode) == gpc.get_local_rank(ctx.output_parallel_mode):
            weight_grad, op = all_reduce(weight_grad, ctx.weight_parallel_mode, async_op=True)
            weight_grad = push_async_grad(op, weight_grad, ctx.weight_id)
        else:
            weight_grad = None

        if ctx.use_bias:
            bias_grad = torch.sum(output_grad, dim=tuple(range(len(output_grad.shape))[:-1]))
            bias_grad = all_reduce(bias_grad, ctx.input_parallel_mode)
            bias_grad, op = all_reduce(bias_grad, ctx.weight_parallel_mode, async_op=True)
            bias_grad = push_async_grad(op, bias_grad, ctx.bias_id)
        else:
            bias_grad = None

        input_grad = torch.matmul(output_grad, weight)

        return input_grad, weight_grad, bias_grad, None, None, None, None, None


def classifier_3d(
    input_: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    input_parallel_mode: ParallelMode,
    weight_parallel_mode: ParallelMode,
    output_parallel_mode: ParallelMode,
) -> Tensor:
    r"""3D parallel classifier.

    Args:
        input_ (:class:`torch.tensor`): input matrix.
        weight (:class:`torch.tensor`): matrix of weight.
        bias (:class:`torch.tensor`): matrix of bias.
        input_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): input parallel mode.
        weight_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): weight parallel mode.
        output_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): output parallel mode.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    return _Classifier3D.apply(
        input_,
        weight,
        bias,
        id(weight),
        id(bias) if bias is not None else None,
        input_parallel_mode,
        weight_parallel_mode,
        output_parallel_mode,
    )


class _VocabParallelClassifier3D(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx,
        input_: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        weight_id: int,
        bias_id: Optional[int],
        input_parallel_mode: ParallelMode,
        weight_parallel_mode: ParallelMode,
        output_parallel_mode: ParallelMode,
    ) -> Tensor:
        ctx.use_bias = bias is not None
        ctx.weight_id = weight_id

        input_ = all_gather(input_, 0, input_parallel_mode)
        weight = all_gather(weight, 0, weight_parallel_mode).transpose(0, 1)
        ctx.save_for_backward(input_, weight)

        output = torch.matmul(input_, weight)
        output = reduce_scatter(output, 0, output_parallel_mode)

        if bias is not None:
            ctx.bias_id = bias_id
            output += bias

        ctx.input_parallel_mode = input_parallel_mode
        ctx.weight_parallel_mode = weight_parallel_mode
        ctx.output_parallel_mode = output_parallel_mode
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
        input_, weight = ctx.saved_tensors
        output_grad = all_gather(output_grad, 0, ctx.output_parallel_mode)

        input_grad = torch.matmul(output_grad, weight.transpose(0, 1))
        input_grad, input_op = reduce_scatter(input_grad, 0, ctx.input_parallel_mode, async_op=True)

        weight_grad = torch.matmul(
            input_.reshape(-1, input_.shape[-1]).transpose(0, 1), output_grad.reshape(-1, output_grad.shape[-1]))
        weight_grad, op = reduce_scatter(weight_grad.transpose(0, 1), 0, ctx.weight_parallel_mode, async_op=True)
        weight_grad = push_async_grad(op, weight_grad, ctx.weight_id)

        if ctx.use_bias:
            bias_grad = torch.sum(output_grad, dim=tuple(range(len(output_grad.shape))[:-1]))
            bias_grad, op = all_reduce(bias_grad, ctx.weight_parallel_mode, async_op=True)
            bias_grad = push_async_grad(op, bias_grad, ctx.bias_id)
        else:
            bias_grad = None

        input_op.wait()

        return input_grad, weight_grad, bias_grad, None, None, None, None, None


def vocab_parallel_classifier_3d(
    input_: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    input_parallel_mode: ParallelMode,
    weight_parallel_mode: ParallelMode,
    output_parallel_mode: ParallelMode,
) -> Tensor:
    r"""3D vocab parallel classifier.

    Args:
        input_ (:class:`torch.tensor`): input matrix.
        weight (:class:`torch.tensor`): matrix of weight.
        bias (:class:`torch.tensor`): matrix of bias.
        input_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): input parallel mode.
        weight_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): weight parallel mode.
        output_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): output parallel mode.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    return _VocabParallelClassifier3D.apply(
        input_,
        weight,
        bias,
        id(weight),
        id(bias) if bias is not None else None,
        input_parallel_mode,
        weight_parallel_mode,
        output_parallel_mode,
    )


@torch.jit.script
def norm_forward(x: Tensor, mean: Tensor, sqr_mean: Tensor, weight: Tensor, bias: Tensor, eps: float):
    mu = x - mean
    var = sqr_mean - mean**2
    sigma = torch.sqrt(var + eps)
    z = mu / sigma
    output = weight * z + bias

    return output, mu, sigma


@torch.jit.script
def norm_backward(grad: Tensor, mu: Tensor, sigma: Tensor, weight: Tensor):
    # dbias, dweight = grad, grad * mu / sigma
    dz = grad * weight
    dmu = dz / sigma
    dvar = dz * mu * (-0.5) * sigma**(-3)
    dmean = -dmu
    dvar = torch.sum(dvar, -1, keepdim=True)
    dmean = torch.sum(dmean, -1, keepdim=True)

    return dmu, dmean, dvar


class _Layernorm3D(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx,
        input_: Tensor,
        weight: Tensor,
        bias: Tensor,
        weight_id: int,
        bias_id: int,
        normalized_shape: int,
        eps: float,
        output_parallel_mode: ParallelMode,
        input_x_weight_parallel_mode: ParallelMode,
    ) -> Tensor:
        ctx.weight_id = weight_id
        ctx.bias_id = bias_id

        sum_ = torch.sum(input_, dim=-1, keepdim=True)
        sqr_sum = torch.sum(input_**2, dim=-1, keepdim=True)
        mean, sqr_mean = all_reduce(torch.stack((sum_, sqr_sum)), output_parallel_mode) / normalized_shape

        output, mu, sigma = norm_forward(input_, mean, sqr_mean, weight, bias, eps)

        ctx.save_for_backward(mu, sigma, weight)

        ctx.normalized_shape = normalized_shape
        ctx.output_parallel_mode = output_parallel_mode
        ctx.input_x_weight_parallel_mode = input_x_weight_parallel_mode

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, ...]:
        mu, sigma, weight = ctx.saved_tensors

        bias_grad, weight_grad = output_grad, output_grad * mu / sigma
        bias_grad = torch.sum(bias_grad, dim=tuple(range(len(bias_grad.shape))[:-1]))
        bias_grad, op = all_reduce(bias_grad, ctx.input_x_weight_parallel_mode, async_op=True)
        bias_grad = push_async_grad(op, bias_grad, ctx.bias_id)
        weight_grad = torch.sum(weight_grad, dim=tuple(range(len(weight_grad.shape))[:-1]))
        weight_grad, op = all_reduce(weight_grad, ctx.input_x_weight_parallel_mode, async_op=True)
        weight_grad = push_async_grad(op, weight_grad, ctx.weight_id)

        dmu, dmean, dvar = norm_backward(output_grad, mu, sigma, weight)
        dvar, dmean = all_reduce(torch.stack((dvar, dmean)), ctx.output_parallel_mode)
        input_grad = dmu + (dmean + 2 * dvar * mu) / ctx.normalized_shape

        return input_grad, weight_grad, bias_grad, None, None, None, None, None, None, None, None


def layernorm_3d(
    input_: Tensor,
    weight: Tensor,
    bias: Tensor,
    normalized_shape: int,
    eps: float,
    output_parallel_mode: ParallelMode,
    input_x_weight_parallel_mode: ParallelMode,
) -> Tensor:
    r"""3D parallel Layernorm.

    Args:
        input_ (:class:`torch.tensor`): input matrix.
        weight (:class:`torch.tensor`): matrix of weight.
        bias (:class:`torch.tensor`): matrix of bias.
        normalized_shape (int): input shape from an expected input of size.
            :math:`[* \times \text{normalized_shape}[0] \times \text{normalized_shape}[1]
            \times \ldots \times \text{normalized_shape}[-1]]`
            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps (float): a value added to the denominator for numerical stability
        output_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): output parallel mode.
        input_x_weight_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): input x weight parallel mode.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    return _Layernorm3D.apply(
        input_,
        weight,
        bias,
        id(weight),
        id(bias),
        normalized_shape,
        eps,
        output_parallel_mode,
        input_x_weight_parallel_mode,
    )


def split_tensor_3d(tensor: Tensor, dim: int, parallel_mode: ParallelMode) -> Tensor:
    r"""Splits 3D parallel tensor in specified dimension.

    Args:
        tensor (:class:`torch.tensor`): Input tensor.
        dim (int): Specified dimension in which to split.
        parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`, optional): Parallel mode.

    Returns:
        :class:`torch.tensor`: The tensor has been split.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """
    dim_size = tensor.size(dim)
    world_size = gpc.get_world_size(parallel_mode)
    assert dim_size % world_size == 0, \
        f'The dimension {dim} to split, size ({dim_size}) is not a multiple of world size ({world_size}), ' \
        f'cannot split tensor evenly'
    if tensor.size(dim) <= 1:
        return tensor
    output = torch.chunk(tensor, gpc.get_world_size(parallel_mode),
                         dim=dim)[gpc.get_local_rank(parallel_mode)].contiguous()
    return output


def split_batch_3d(input_: Tensor,
                   dim: int = 0,
                   input_parallel_mode: ParallelMode = ParallelMode.PARALLEL_3D_INPUT,
                   weight_parallel_mode: ParallelMode = ParallelMode.PARALLEL_3D_WEIGHT) -> Tensor:
    r"""Splits 3D tensor in batch.

    Args:
        input_ (:class:`torch.tensor`): Input tensor.
        dim (int): Specified dimension in which to split.
        input_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`, optional): input parallel mode.
        weight_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`, optional): weight parallel mode.

    Returns:
        :class:`torch.tensor`: The tensor has been split.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """
    if input_.size(dim) <= 1:
        return input_
    weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
    input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
    weight_world_size = gpc.get_world_size(weight_parallel_mode)
    input_world_size = gpc.get_world_size(input_parallel_mode)
    output = torch.chunk(input_, weight_world_size, dim=dim)[gpc.get_local_rank(weight_parallel_mode)].contiguous()
    output = torch.chunk(output, input_world_size, dim=dim)[gpc.get_local_rank(input_parallel_mode)].contiguous()
    return output


class _ReduceTensor3D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, parallel_mode):
        return all_reduce(input_, parallel_mode)

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad, None


def reduce_tensor_3d(tensor: Tensor, parallel_mode: ParallelMode) -> Tensor:
    r"""All-reduce the input

    Args:
        tensor (:class:`torch.tensor`): Input tensor.
        parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): Parallel mode.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
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
    r"""All-reduce the gradient in backward pass.

    Args:
        tensor (:class:`torch.tensor`): Input tensor.
        dim (int): Dimension to gather.
        parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): Parallel mode.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
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
    r"""Reduce-scatter the input.

    Args:
        tensor (:class:`torch.tensor`): Input tensor.
        dim (int): Dimension to scatter.
        parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): Parallel mode.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    dim_size = tensor.size(dim)
    world_size = gpc.get_world_size(parallel_mode)
    assert dim_size % world_size == 0, \
        f'The batch size ({dim_size}) is not a multiple of square of 3D depth ({world_size}).'

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
    r"""All-reduce the input from the model parallel region.

    Args:
        input_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): input parallel mode.
        weight_parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`): weight parallel mode.
        reduce_mean (bool, optional): If set to ``True``, it will divide the output by
            (input parallel size * weight parallel size), default to False.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    return _ReduceByBatch3D.apply(tensor, input_parallel_mode, weight_parallel_mode, reduce_mean)
