#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import functools
import os
import random
import socket
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from torch._six import inf
from torch.nn.parameter import Parameter

from colossalai.constants import IS_TENSOR_PARALLEL, NUM_PARTITIONS, TENSOR_PARALLEL_ATTRIBUTES
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.global_variables import tensor_parallel_env as env
from colossalai.tensor import ColoParameter, ProcessGroup

from .multi_tensor_apply import multi_tensor_applier

try:
    from colossalai._C import fused_optim
except:
    fused_optim = None


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


def is_ddp_ignored(p):
    return getattr(p, '_ddp_to_ignore', False)


def _calc_l2_norm(grads):
    # we should not
    global fused_optim

    if fused_optim is None:
        from colossalai.kernel.op_builder import FusedOptimBuilder
        fused_optim = FusedOptimBuilder().load()

    norm = 0.0
    if len(grads) > 0:
        dummy_overflow_buf = torch.cuda.IntTensor([0])
        norm, _ = multi_tensor_applier(
            fused_optim.multi_tensor_l2norm,
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


def _compute_local_lp(params: List[ColoParameter], norm_type: float) -> float:
    if len(params) == 0:
        return 0.0
    grads = [p.grad for p in params]
    use_cuda_kernel = grads[0].device.type == 'cuda'
    if norm_type == inf:
        local_lp = max([g.abs().max() for g in grads])
    elif norm_type == 2.0 and use_cuda_kernel:
        local_lp = _calc_l2_norm(grads)**norm_type
    else:
        local_lp = _calc_lp(grads, norm_type)
    if isinstance(local_lp, torch.Tensor):
        return local_lp.item()
    return local_lp


def _compute_buckets_lp(params: List[ColoParameter], norm_type: float) -> float:
    if len(params) == 0:
        return 0.0
    buckets: Dict[Optional[ProcessGroup], List[ColoParameter]] = defaultdict(list)
    for p in params:
        if p.is_replicate():
            buckets[None].append(p)
        else:
            buckets[p.get_process_group().tp_process_group()].append(p)
    total_lp = 0.0
    for group, bucket in buckets.items():
        local_lp = _compute_local_lp(bucket, norm_type)
        if group is not None:
            local_lp_tensor = torch.tensor([local_lp], device=torch.cuda.current_device())
            if norm_type == inf:
                dist.all_reduce(local_lp_tensor, op=dist.ReduceOp.MAX, group=group)
            else:
                dist.all_reduce(local_lp_tensor, group=group)
            local_lp = local_lp_tensor.item()
        if norm_type == inf:
            total_lp = max(total_lp, local_lp)
        else:
            total_lp += local_lp
    return total_lp


def _compute_pp_grad_lp(total_lp: float, norm_type: float) -> float:
    if gpc.is_initialized(ParallelMode.PIPELINE) and gpc.get_world_size(ParallelMode.PIPELINE) > 1:
        total_lp_tensor = torch.tensor([total_lp], device=torch.cuda.current_device())
        if norm_type == inf:
            dist.all_reduce(total_lp_tensor, op=dist.ReduceOp.MAX, group=gpc.get_group(ParallelMode.PIPELINE))
        else:
            dist.all_reduce(total_lp_tensor, group=gpc.get_group(ParallelMode.PIPELINE))
        total_lp = total_lp_tensor.item()
    return total_lp


def _compute_grad_lp(parameters, norm_type: float = 2.0) -> float:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grad_dtype = None
    cpu_grad_params: List[ColoParameter] = []
    cuda_grad_params: List[ColoParameter] = []
    for p in parameters:
        if p.grad is None:
            continue
        assert isinstance(p, ColoParameter)
        if grad_dtype is None:
            grad_dtype = p.grad.dtype
        assert p.grad.dtype == grad_dtype, f'Expected all grads are {grad_dtype}, got {p.grad.dtype}'
        if p.grad.device.type == 'cuda':
            cuda_grad_params.append(p)
        else:
            cpu_grad_params.append(p)
    norm_type = float(norm_type)
    cpu_lp = _compute_buckets_lp(cpu_grad_params, norm_type)
    cuda_lp = _compute_buckets_lp(cuda_grad_params, norm_type)
    if norm_type == inf:
        total_lp = max(cpu_lp, cuda_lp)
    else:
        total_lp = cpu_lp + cuda_lp
    return _compute_pp_grad_lp(total_lp, norm_type)


def compute_grad_norm(parameters, norm_type: float = 2.0) -> float:
    norm_type = float(norm_type)
    total_norm = _compute_grad_lp(parameters, norm_type)
    if norm_type != inf:
        total_norm = total_norm**(1 / norm_type)
    return total_norm


def _clip_grad_norm(parameters, max_norm: float, total_norm: float) -> None:
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        cuda_grads: List[torch.Tensor] = []
        cpu_grads: List[torch.Tensor] = []
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        for p in parameters:
            if p.grad is None:
                continue
            if p.grad.device.type == 'cuda':
                cuda_grads.append(p.grad.detach())
            else:
                cpu_grads.append(p.grad.detach())
        if len(cuda_grads) > 0:
            dummy_overflow_buf = torch.cuda.IntTensor([0])
            multi_tensor_applier(fused_optim.multi_tensor_scale, dummy_overflow_buf, [cuda_grads, cuda_grads],
                                 clip_coef)
        for g in cpu_grads:
            g.mul_(clip_coef)


def clip_grad_norm(parameters, max_norm: float, norm_type: float = 2.0) -> float:
    total_norm = compute_grad_norm(parameters, norm_type)
    _clip_grad_norm(parameters, max_norm, total_norm)
    return total_norm


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
            if hasattr(param, 'colo_attr') and param.colo_attr.sharded_data_tensor.is_sharded:
                has_zero_shared_param = True
            params.append(param)

    if len(params) == 0:
        enable_cuda_kernels = False
    else:
        enable_cuda_kernels = params[0].grad.device.type == 'cuda'
    # Norm parameters.
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    # Parameters can be on CPU or CUDA
    # If parameters are on CPU, disable CUDA kernerls

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
            elif hasattr(p, 'colo_attr') and p.colo_attr.sharded_data_tensor.is_sharded:
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
        # If norm is type of float, then we convert them into torch.Tensor.
        tensor_parallel_norm = _get_tensor_norm(tensor_parallel_norm, enable_cuda_kernels)
        no_tensor_parallel_norm = _get_tensor_norm(no_tensor_parallel_norm, enable_cuda_kernels)
        zero_sharded_norm = _get_tensor_norm(zero_sharded_norm, enable_cuda_kernels)
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
            multi_tensor_applier(fused_optim.multi_tensor_scale, dummy_overflow_buf, [grads, grads], clip_coeff)
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


def disposable(func: Callable) -> Callable:
    executed = False

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal executed
        if not executed:
            executed = True
            return func(*args, **kwargs)

    return wrapper
