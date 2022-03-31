#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import random
import socket
from pathlib import Path
from typing import List, Union

import torch
from torch._six import inf
from torch.nn.parameter import Parameter

try:
    import colossal_C
except:
    pass

from contextlib import contextmanager

import torch.distributed as dist
from colossalai.constants import (IS_TENSOR_PARALLEL, NUM_PARTITIONS, TENSOR_PARALLEL_ATTRIBUTES)
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.global_variables import tensor_parallel_env as env

from .multi_tensor_apply import multi_tensor_applier


def print_rank_0(msg: str, logger=None):
    """Print messages and save logs(optional). This is executed only if you are the rank-0 gpu.

    Args:
        msg (str): A string message to output.
        logger (:class:`colossalai.logging.DistributedLogger`, optional):
            The logger to record the message, defaults to None.
    """
    if gpc.get_global_rank() == 0:
        if logger is None:
            print(msg, flush=True)
        else:
            logger.info(msg)


def ensure_path_exists(filename: str):
    # ensure the path exists
    dirpath = os.path.dirname(filename)
    if not os.path.exists(dirpath):
        Path(dirpath).mkdir(parents=True, exist_ok=True)


def free_port():
    while True:
        try:
            sock = socket.socket()
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = random.randint(20000, 65000)
            sock.bind(('localhost', port))
            sock.close()
            return port
        except Exception:
            continue


def sync_model_param(model, parallel_mode):
    r"""Make sure data parameters are consistent during Data Parallel Mode.

    Args:
        model (:class:`torch.nn.Module`): A pyTorch model on whose parameters you check the consistency.
        parallel_mode (:class:`colossalai.context.ParallelMode`): Parallel mode to be checked.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
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


def _move_norm_to_cuda(norm: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
    if torch.is_tensor(norm) and norm.device.type != 'cuda':
        norm = norm.to(torch.cuda.current_device())
    return norm


# ======== Gradient Clipping =========


def clip_grad_norm_fp32(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters whose gradients are in fp32.

    This is adapted from :func:`torch.nn.utils.clip_grad.clip_grad_norm_` and
    added functionality to handle model parallel parameters.

    Note:
        the gradients are modified in place.

    Args:
        parameters (Iterable[:class:`torch.tensor`] or :class:`torch.tensor`):
            An iterable of Tensors or a single Tensor that will have gradients normalized.
        max_norm (Union[float, int]): Max norm of the gradients.
        norm_type (Union[float, int, 'inf']): Type of the used p-norm. Can be ``'inf'`` for infinity norm.

    Returns:
        float: Total norm of the parameters.
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters based on:
    #   - grad should not be none
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    params: List[Parameter] = []
    has_zero_shared_param: bool = False
    for param in parameters:
        if param.grad is not None:
            # Make sure the grads are in fp32
            assert param.grad.dtype == torch.float, \
                f'expected gradient to be dtype torch.float, but got {param.grad.type()}'
            if hasattr(param, 'zero_is_sharded'):
                has_zero_shared_param = True
            params.append(param)

    if len(params) == 0:
        return 0.0
    # Norm parameters.
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    # Parameters can be on CPU or CUDA
    # If parameters are on CPU, disable CUDA kernerls
    enable_cuda_kernels = params[0].grad.device.type == 'cuda'

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
        if has_zero_shared_param:
            dist.all_reduce(total_norm_cuda,
                            op=dist.ReduceOp.MAX,
                            group=gpc.get_group(ParallelMode.DATA),
                            async_op=False)
        total_norm = total_norm_cuda[0].item()
    else:
        tensor_parallel_grads = []
        no_tensor_parallel_grads = []
        zero_sharded_grads = []
        for p in params:
            if is_model_parallel_parameter(p):
                reductor = (gpc.get_world_size(ParallelMode.TENSOR) / getattr(p, NUM_PARTITIONS))**(1 / norm_type)
                tensor_parallel_grads.append(p.grad.data / reductor)
            elif hasattr(p, 'zero_is_sharded'):
                zero_sharded_grads.append(p.grad.data)
            else:
                no_tensor_parallel_grads.append(p.grad.data)

        if norm_type == 2.0 and enable_cuda_kernels:
            tensor_parallel_norm = _calc_l2_norm(tensor_parallel_grads)**norm_type
            no_tensor_parallel_norm = _calc_l2_norm(no_tensor_parallel_grads)**norm_type
            zero_sharded_norm = _calc_l2_norm(zero_sharded_grads)**norm_type
        else:
            tensor_parallel_norm = _calc_lp(tensor_parallel_grads, norm_type)
            no_tensor_parallel_norm = _calc_lp(no_tensor_parallel_grads, norm_type)
            zero_sharded_norm = _calc_lp(zero_sharded_grads, norm_type)

        # If grads are on CPU, the norms is also on CPU. Cast them to CUDA tensors
        if not enable_cuda_kernels:
            tensor_parallel_norm = _move_norm_to_cuda(tensor_parallel_norm)
            no_tensor_parallel_norm = _move_norm_to_cuda(no_tensor_parallel_norm)
            zero_sharded_norm = _move_norm_to_cuda(zero_sharded_norm)

        # Sum across all model-parallel GPUs.
        if gpc.is_initialized(ParallelMode.TENSOR) and len(tensor_parallel_grads) > 0:
            dist.all_reduce(tensor_parallel_norm, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.TENSOR))
        # Sum across all zero sharded GPUs
        if len(zero_sharded_grads) > 0:
            dist.all_reduce(zero_sharded_norm, group=gpc.get_group(ParallelMode.DATA))
            no_tensor_parallel_norm += zero_sharded_norm
        total_norm = tensor_parallel_norm + no_tensor_parallel_norm
        if gpc.is_initialized(ParallelMode.PIPELINE) and gpc.get_world_size(ParallelMode.PIPELINE) > 1:
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.PIPELINE))
        total_norm = total_norm**(1.0 / norm_type)
        if torch.is_tensor(total_norm):
            total_norm = total_norm.item()

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        if enable_cuda_kernels:
            grads = [p.grad.detach() for p in params]
            dummy_overflow_buf = torch.cuda.IntTensor([0])
            multi_tensor_applier(colossal_C.multi_tensor_scale, dummy_overflow_buf, [grads, grads], clip_coeff)
        else:
            for p in params:
                p.grad.detach().mul_(clip_coeff)
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
