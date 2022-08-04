#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import random
import socket
from pathlib import Path
from typing import Callable, List, Union
import functools
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

from colossalai.tensor import ColoTensor, ColoParameter

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
            False    # no per-parameter norm
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


def _get_tensor_norm(norm: Union[float, torch.Tensor], move_to_cuda) -> torch.Tensor:
    if isinstance(norm, float):
        norm = torch.Tensor([norm])
    if move_to_cuda:
        norm = norm.to(torch.cuda.current_device())
    return norm


# ======== Gradient Clipping =========

def _norm_tensor_parallel_fp32(parameters, norm_type=2):
    '''
    Calculate norm.
    Won't check if parameters are TP only
    expect parameter grads on cuda
    '''
    if isinstance(parameters, ColoTensor):
        parameters = [parameters]
    params: List[ColoParameter] = []
    for param in parameters:
        if param.grad is not None:
            assert param.grad.dtype == torch.float, \
                f'expected gradient to be dtype torch.float, but got {param.grad.type()}'
            assert param.grad.device.type == 'cuda', \
                f'expected gradient to be on cuda, but got {param.grad.device.type}'
            params.append(param)
    norm_type = float(norm_type)
    total_norm = 0.0
    if norm_type == inf:
        for p in params:
            local_norm = p.grad.data.abs().max()
            local_norm_cuda = torch.cuda.FloatTensor([float(local_norm)])
            if p.is_sharded():
                dist.all_reduce(local_norm_cuda,
                                op=dist.ReduceOp.MAX,
                                group=p.get_process_group().tp_process_group(),
                                async_op=False)
            total_norm = max(total_norm, local_norm_cuda[0].item())

    else:
        for p in params:
            if norm_type == 2.0:
                local_norm = _calc_l2_norm([p.grad.data])**norm_type
            else:
                local_norm = _calc_lp([p.grad.data], norm_type)
            local_norm_cuda = torch.cuda.FloatTensor([float(local_norm)])
            if p.is_sharded():
                dist.all_reduce(local_norm_cuda,
                                op=dist.ReduceOp.SUM,
                                group=p.get_process_group().tp_process_group(),
                                async_op=False)
            total_norm = total_norm + local_norm_cuda[0].item()
        total_norm = total_norm**(1.0 / norm_type)    
    if torch.torch.is_tensor(total_norm):
        total_norm = total_norm.item()
    return total_norm
    
def _norm_pipeline_parallel_fp32(total_norm, norm_type):
    '''
        TODO
        if norm_type == inf:
            MAX
        else:
            x**norm_type, SUM, x**(1/norm_type)
    '''
    return total_norm

def _norm_data_parallel_fp32(total_norm, norm_type):
    '''
        TODO
        if norm_type == inf:
            MAX
        else:
            x**norm_type, SUM, x**(1/norm_type)
    '''
    return total_norm

def _clip_grad_fp32(parameters, max_norm, total_norm):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        grads = [p.grad.detach() for p in parameters]
        dummy_overflow_buf = torch.cuda.IntTensor([0])
        multi_tensor_applier(colossal_C.multi_tensor_scale, dummy_overflow_buf, [grads, grads], clip_coeff)
    return total_norm

def clip_grad_norm_fp32(parameters, max_norm, norm_type=2):
    enable_pipline_parallel = False
    enable_zero = False
    total_norm = _norm_tensor_parallel_fp32(parameters, norm_type)
    if enable_pipline_parallel:
        total_norm = _norm_pipeline_parallel_fp32(total_norm, norm_type)
    if enable_zero:
        total_norm = _norm_data_parallel_fp32(total_norm, norm_type)
    _clip_grad_fp32(parameters, max_norm, total_norm)
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


def disposable(func: Callable) -> Callable:
    executed = False

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal executed
        if not executed:
            executed = True
            return func(*args, **kwargs)

    return wrapper
