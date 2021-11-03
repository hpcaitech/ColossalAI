#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch._six import inf

try:
    import colossal_C
except:
    print('Colossalai should be built with cuda extension to use the FP16 optimizer')

from ..multi_tensor_apply import multi_tensor_applier

from colossalai.constants import IS_TENSOR_PARALLEL, TENSOR_PARALLEL_ATTRIBUTES
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc


def is_model_parallel_parameter(p):
    return hasattr(p, IS_TENSOR_PARALLEL) and getattr(p, IS_TENSOR_PARALLEL)


def _calc_l2_norm(grads):
    norm = 0.0
    if len(grads) > 0:
        dummy_overflow_buf = torch.cuda.IntTensor([0])
        norm, _ = multi_tensor_applier(
            colossal_C.multi_tensor_l2norm,
            dummy_overflow_buf,
            [grads],
            False  # no per-parameter norm
        )
    return norm


def _calc_lp(grads, norm_type):
    norm = 0.0
    for grad in grads:
        grad_norm = torch.norm(grad, norm_type)
        norm += grad_norm ** norm_type
    return norm

# ======== Gradient Clipping =========


def clip_grad_norm_fp32(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters whose gradients
       are in fp32.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters based on:
    #   - grad should not be none
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    params = []
    for param in parameters:
        if param.grad is not None:
            # Make sure the grads are in fp32
            assert param.grad.type() == 'torch.cuda.FloatTensor'
            params.append(param)
    # Norm parameters.
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    # Calculate norm.
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in params)
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        if gpc.is_initialized(ParallelMode.TENSOR):
            # Take max across all model-parallel GPUs.
            torch.distributed.all_reduce(total_norm_cuda,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=gpc.get_group(ParallelMode.TENSOR))
        total_norm = total_norm_cuda[0].item()
    else:
        tensor_parallel_grads = []
        no_tensor_parallel_grads = []
        for p in params:
            if is_model_parallel_parameter(p):
                tensor_parallel_grads.append(p.grad.data)
            else:
                no_tensor_parallel_grads.append(p.grad.data)
        if norm_type == 2.0:
            tensor_parallel_norm = _calc_l2_norm(
                tensor_parallel_grads) ** norm_type
            no_tensor_parallel_norm = _calc_l2_norm(
                no_tensor_parallel_grads) ** norm_type
        else:
            tensor_parallel_norm = _calc_lp(tensor_parallel_grads, norm_type)
            no_tensor_parallel_grads = _calc_lp(
                no_tensor_parallel_grads, norm_type)
        if gpc.is_initialized(ParallelMode.TENSOR) and len(tensor_parallel_grads) > 0:
            # Sum across all model-parallel GPUs.
            torch.distributed.all_reduce(tensor_parallel_norm,
                                         op=torch.distributed.ReduceOp.SUM,
                                         group=gpc.get_group(ParallelMode.TENSOR))
        total_norm = (tensor_parallel_norm +
                      no_tensor_parallel_norm) ** (1.0 / norm_type)
        if type(total_norm) == 'torch.cuda.FloatTensor':
            total_norm = total_norm.item()

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        grads = [p.grad.detach() for p in params]
        dummy_overflow_buf = torch.cuda.IntTensor([0])
        multi_tensor_applier(colossal_C.multi_tensor_scale,
                             dummy_overflow_buf,
                             [grads, grads],
                             clip_coeff)

    return total_norm


def count_zeros_fp32(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters based on:
    #   - grad should not be none
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    total_num_zeros = 0.0
    for param in parameters:
        grad_not_none = param.grad is not None
        is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
        if grad_not_none and is_not_tp_duplicate:
            grad = param.grad.detach()
            num_zeros = grad.numel() - torch.count_nonzero(grad)
            total_num_zeros = num_zeros + total_num_zeros

    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(total_num_zeros,
                                 op=torch.distributed.ReduceOp.SUM,
                                 group=gpc.get_group(ParallelMode.TENSOR))
    total_num_zeros = total_num_zeros.item()

    return total_num_zeros


def copy_tensor_parallel_attributes(src_tensor, dst_tensor):
    for attr in TENSOR_PARALLEL_ATTRIBUTES:
        if hasattr(src_tensor, attr):
            val = getattr(src_tensor, attr)
            setattr(dst_tensor, attr, val)


def param_is_not_tensor_parallel_duplicate(param):
    return (hasattr(param, IS_TENSOR_PARALLEL) and
            getattr(param, IS_TENSOR_PARALLEL)) or (
        gpc.get_local_rank(ParallelMode.TENSOR) == 0)
