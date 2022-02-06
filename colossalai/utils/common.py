#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import random
import socket

import torch
from torch._six import inf

try:
    import colossal_C
except:
    pass

from contextlib import contextmanager

import torch.distributed as dist
from colossalai.constants import IS_TENSOR_PARALLEL, NUM_PARTITIONS, TENSOR_PARALLEL_ATTRIBUTES
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.global_variables import moe_env
from colossalai.global_variables import tensor_parallel_env as env

from .multi_tensor_apply import multi_tensor_applier


def print_rank_0(msg: str, logger=None):
    """Print messages and save logs(optional). This is executed only if you are the rank-0 gpu.

    :param msg: A string message to output
    :type msg: str
    :param logger: Python logger object, defaults to None
    :type logger: optional
    """
    if gpc.get_global_rank() == 0:
        if logger is None:
            print(msg, flush=True)
        else:
            logger.info(msg)


def free_port():
    while True:
        try:
            sock = socket.socket()
            port = random.randint(20000, 65000)
            sock.bind(('localhost', port))
            sock.close()
            return port
        except Exception:
            continue


def sync_model_param(model, parallel_mode):
    """Make sure data parameters are consistent during Data Parallel Mode

    :param model: A pyTorch nn.model on whose parameters you check the consistency
    :param parallel_mode: Parallel mode to be checked
    :type model: torch.nn.Module
    :type parallel_mode:  colossalai.context.ParallelMode
    """
    if gpc.is_initialized(parallel_mode) and gpc.get_world_size(parallel_mode) > 1:
        for param in model.parameters():
            ranks = gpc.get_ranks_in_group(parallel_mode)
            dist.broadcast(param, src=ranks[0], group=gpc.get_group(parallel_mode))


def is_dp_rank_0():
    return not gpc.is_initialized(ParallelMode.DATA) or gpc.is_first_rank(ParallelMode.DATA)


def is_tp_rank_0():
    return not gpc.is_initialized(ParallelMode.TENSOR) or gpc.is_first_rank(ParallelMode.TENSOR)


def is_no_pp_or_last_stage():
    return not gpc.is_initialized(ParallelMode.PIPELINE) or gpc.is_last_rank(ParallelMode.PIPELINE)


def is_using_ddp():
    return gpc.is_initialized(ParallelMode.DATA) and gpc.get_world_size(ParallelMode.DATA) > 1


def is_using_pp():
    return gpc.is_initialized(ParallelMode.PIPELINE) and gpc.get_world_size(ParallelMode.PIPELINE) > 1


def is_using_sequence():
    return gpc.is_initialized(ParallelMode.SEQUENCE) and gpc.get_world_size(ParallelMode.SEQUENCE) > 1


@contextmanager
def conditional_context(context_manager, enable=True):
    if enable:
        with context_manager:
            yield
    else:
        yield


class model_branch_context(object):

    def __enter__(self):
        self.env_status = env.save()

    def __exit__(self, *exc_info):
        env.load(**self.env_status)


def is_model_parallel_parameter(p):
    return hasattr(p, IS_TENSOR_PARALLEL) and getattr(p, IS_TENSOR_PARALLEL)


def is_moe_parallel_parameter(p):
    return hasattr(p, 'moe_param') and moe_env.data_parallel_size > 1


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
        norm += grad_norm**norm_type
    return norm


# ======== Gradient Clipping =========


def clip_grad_norm_fp32(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters whose gradients are in fp32.

    This is adapted from :func:`torch.nn.utils.clip_grad.clip_grad_norm_` and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    :param parameters: An iterable of Tensors or a single Tensor that will have gradients normalized
    :type parameters: (Iterable[Tensor] or Tensor)
    :param max_norm: Max norm of the gradients
    :type max_norm: float or int
    :param norm_type: Type of the used p-norm. Can be ``'inf'`` for infinity norm.
    :type norm_type: float or int 

    :return: Total norm of the parameters (viewed as a single vector).
    :rtype: float
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
            assert param.grad.type() == 'torch.cuda.FloatTensor', \
                f'expected gradient to be dtype torch.cuda.FloatTensor, but got {param.grad.type()}'
            params.append(param)
    # Norm parameters.
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    # Calculate norm.
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in params)
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        # Take max across all model-parallel GPUs.
        if gpc.is_initialized(ParallelMode.MODEL) and gpc.get_world_size(ParallelMode.MODEL) > 1:
            dist.all_reduce(total_norm_cuda,
                            op=dist.ReduceOp.MAX,
                            group=gpc.get_group(ParallelMode.MODEL),
                            async_op=False)
        total_norm = total_norm_cuda[0].item()
    else:
        tensor_parallel_grads = []
        no_tensor_parallel_grads = []
        moe_parallel_grads = []  # used to collect moe tensor parallel gradients
        for p in params:
            if is_model_parallel_parameter(p):
                reductor = (gpc.get_world_size(ParallelMode.TENSOR) / getattr(p, NUM_PARTITIONS))**(1 / norm_type)
                tensor_parallel_grads.append(p.grad.data / reductor)
            elif is_moe_parallel_parameter(p):
                moe_parallel_grads.append(p.grad.data)
            else:
                no_tensor_parallel_grads.append(p.grad.data)

        if norm_type == 2.0:
            tensor_parallel_norm = _calc_l2_norm(tensor_parallel_grads)**norm_type
            no_tensor_parallel_norm = _calc_l2_norm(no_tensor_parallel_grads)**norm_type
            moe_parallel_norm = _calc_l2_norm(moe_parallel_grads)**norm_type
        else:
            tensor_parallel_norm = _calc_lp(tensor_parallel_grads, norm_type)
            no_tensor_parallel_norm = _calc_lp(no_tensor_parallel_grads, norm_type)
            moe_parallel_norm = _calc_lp(moe_parallel_grads, norm_type)
        # Sum across all model-parallel GPUs.
        if gpc.is_initialized(ParallelMode.TENSOR) and len(tensor_parallel_grads) > 0:
            dist.all_reduce(tensor_parallel_norm, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.TENSOR))
        # Sum across all moe-tensor-parallel GPUs
        if len(moe_parallel_grads) > 0:
            dist.all_reduce(moe_parallel_norm, group=gpc.get_group(ParallelMode.MOE_MODEL))
            no_tensor_parallel_norm += moe_parallel_norm
        total_norm = tensor_parallel_norm + no_tensor_parallel_norm
        if gpc.is_initialized(ParallelMode.PIPELINE) and gpc.get_world_size(ParallelMode.PIPELINE) > 1:
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.PIPELINE))
        total_norm = total_norm**(1.0 / norm_type)
        if type(total_norm) == 'torch.cuda.FloatTensor':
            total_norm = total_norm.item()

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        grads = [p.grad.detach() for p in params]
        dummy_overflow_buf = torch.cuda.IntTensor([0])
        multi_tensor_applier(colossal_C.multi_tensor_scale, dummy_overflow_buf, [grads, grads], clip_coeff)

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

    total_num_zeros = torch.IntTensor([int(total_num_zeros)]).cuda()

    # Sum across all model-parallel GPUs.
    ops = []
    ops.append(
        dist.all_reduce(total_num_zeros, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.TENSOR), async_op=True))
    if gpc.is_initialized(ParallelMode.PIPELINE):
        ops.append(
            dist.all_reduce(total_num_zeros,
                            op=dist.ReduceOp.SUM,
                            group=gpc.get_group(ParallelMode.PIPELINE),
                            async_op=True))

    for req in ops:
        req.wait()
    total_num_zeros = total_num_zeros.item()

    return total_num_zeros


def copy_tensor_parallel_attributes(src_tensor, dst_tensor):
    for attr in TENSOR_PARALLEL_ATTRIBUTES:
        if hasattr(src_tensor, attr):
            val = getattr(src_tensor, attr)
            setattr(dst_tensor, attr, val)


def param_is_not_tensor_parallel_duplicate(param):
    return (hasattr(param, IS_TENSOR_PARALLEL) and getattr(param, IS_TENSOR_PARALLEL)) or (gpc.get_local_rank(
        ParallelMode.TENSOR) == 0)


@contextmanager
def switch_virtual_pipeline_parallel_rank(rank):
    prev_rank = gpc.virtual_pipeline_parallel_rank
    try:
        gpc.set_virtual_pipeline_parallel_rank(rank)
        yield
    finally:
        gpc.set_virtual_pipeline_parallel_rank(prev_rank)
