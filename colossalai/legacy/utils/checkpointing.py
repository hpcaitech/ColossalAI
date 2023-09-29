from collections import OrderedDict
from itertools import chain

import torch
import torch.distributed as dist

from colossalai.legacy.constants import IS_TENSOR_PARALLEL
from colossalai.legacy.context.parallel_mode import ParallelMode
from colossalai.legacy.core import global_context as gpc

try:
    from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX
except ImportError:
    _EXTRA_STATE_KEY_SUFFIX = "_extra_state"

from .common import is_using_pp

__all__ = ["save_checkpoint", "load_checkpoint"]


def broadcast_state_dict(state_dict, parallel_mode):
    state_dict = [state_dict.copy() if isinstance(state_dict, dict) else state_dict]
    src_rank = gpc.get_ranks_in_group(parallel_mode)[0]
    dist.broadcast_object_list(state_dict, src=src_rank, group=gpc.get_cpu_group(parallel_mode))
    return state_dict[0]


def partition_tensor_parallel_state_dict(
    state_dict: OrderedDict, parallel_mode: ParallelMode, dims: dict = dict(), partition_states: dict = dict()
):
    src_rank = gpc.get_ranks_in_group(parallel_mode)[0]
    depth = gpc.get_world_size(parallel_mode)
    group = gpc.get_cpu_group(parallel_mode)
    is_rank0 = gpc.get_local_rank(parallel_mode) == 0
    partition_info = [None]
    if is_rank0:
        partition_info_dict = OrderedDict()
        for key, param in state_dict.items():
            dim = dims[key]
            is_partitioned = partition_states[key]
            shape = list(param.shape)
            if is_partitioned:
                shape[dim] = shape[dim] // depth
            partition_info_dict[key] = (is_partitioned, param.dtype, shape, dim)
        partition_info[0] = partition_info_dict
    dist.broadcast_object_list(partition_info, src_rank, group=group)
    partitioned_state = OrderedDict()
    for key, (is_partitioned, dtype, shape, dim) in partition_info[0].items():
        if is_partitioned:
            output = torch.empty(shape, dtype=dtype)
            if is_rank0:
                scatter_list = [t.contiguous() for t in state_dict[key].chunk(depth, dim)]
            else:
                scatter_list = None
            dist.scatter(output, scatter_list, src_rank, group=group)
        else:
            if is_rank0:
                output = state_dict[key]
            else:
                output = torch.empty(shape, dtype=dtype)
            dist.broadcast(output, src_rank, group=group)
        partitioned_state[key] = output
    return partitioned_state


def gather_tensor_parallel_state_dict(
    state_dict: OrderedDict,
    parallel_mode: ParallelMode,
    dims: dict = dict(),
    partition_states: dict = dict(),
    keep_vars: bool = False,
):
    dst_rank = gpc.get_ranks_in_group(parallel_mode)[0]
    depth = gpc.get_world_size(parallel_mode)

    for key in list(state_dict.keys()):
        param = state_dict.pop(key)
        param = param if keep_vars else param.detach()
        dim = dims.get(key, 0)
        do_partition = partition_states.get(key, True)
        if do_partition:
            temp = param.transpose(0, dim).contiguous()
            gather_list = None
            if gpc.get_local_rank(parallel_mode) == 0:
                shape = list(param.shape)
                shape[0], shape[dim] = shape[dim], shape[0]
                shape[0] *= depth
                param = torch.empty(shape, dtype=param.dtype, device=param.device)
                gather_list = list(torch.chunk(param, depth, dim=0))
            dist.gather(temp, gather_list, dst=dst_rank, group=gpc.get_cpu_group(parallel_mode))
            param = torch.transpose(param, 0, dim)
        # update params in state_dict only on local rank 0
        if gpc.get_local_rank(parallel_mode) == 0:
            state_dict[key] = param

    return state_dict


def _send_state_dict(state_dict, dst, parallel_mode):
    state_tensor, state_size = dist.distributed_c10d._object_to_tensor(state_dict)
    dist.send(state_size, dst, group=gpc.get_cpu_group(parallel_mode))
    dist.send(state_tensor, dst, group=gpc.get_cpu_group(parallel_mode))


def _recv_state_dict(src, parallel_mode):
    state_size = torch.tensor([0], dtype=torch.long)
    dist.recv(state_size, src, group=gpc.get_cpu_group(parallel_mode))
    state_tensor = torch.empty(state_size.item(), dtype=torch.uint8)
    dist.recv(state_tensor, src, group=gpc.get_cpu_group(parallel_mode))
    state_dict = dist.distributed_c10d._tensor_to_object(state_tensor, state_size)
    return state_dict


def partition_pipeline_parallel_state_dict(model, state_dict):
    pipeline_state = OrderedDict()

    if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
        # receive all states from prev stage
        if not gpc.is_first_rank(ParallelMode.PIPELINE):
            state_dict = _recv_state_dict(gpc.get_prev_global_rank(ParallelMode.PIPELINE), ParallelMode.PIPELINE)
        # move states to output
        for name, _ in model.named_parameters(recurse=True):
            if name in state_dict:
                pipeline_state[name] = state_dict.pop(name)
        for name, _ in model.named_buffers(recurse=True):
            if name in state_dict:
                pipeline_state[name] = state_dict.pop(name)
        for name, _ in model.named_modules():
            extra_state_key = name + "." + _EXTRA_STATE_KEY_SUFFIX
            if extra_state_key in state_dict:
                pipeline_state[extra_state_key] = state_dict.pop(extra_state_key)
        # send rest states to next stage
        if not gpc.is_last_rank(ParallelMode.PIPELINE):
            _send_state_dict(state_dict, gpc.get_next_global_rank(ParallelMode.PIPELINE), ParallelMode.PIPELINE)

    return pipeline_state


def gather_pipeline_parallel_state_dict(state_dict):
    gathered_states = (
        [None for _ in range(gpc.get_world_size(ParallelMode.PIPELINE))]
        if gpc.get_local_rank(ParallelMode.PIPELINE) == 0
        else None
    )
    dist.gather_object(
        state_dict,
        gathered_states,
        dst=gpc.get_ranks_in_group(ParallelMode.PIPELINE)[0],
        group=gpc.get_cpu_group(ParallelMode.PIPELINE),
    )

    state_dict = (
        OrderedDict(chain.from_iterable(state.items() for state in gathered_states))
        if gpc.get_local_rank(ParallelMode.PIPELINE) == 0
        else OrderedDict()
    )

    return state_dict


def save_checkpoint(
    file,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    **kwargs,
):
    """Stores the checkpoint to disk. Saves all the training components' parameters or buffers, such as model, optimizer,
    lr_scheduler etc. into a checkpoint dictionary.

    Args:
        file: a file-like object (has to implement write and flush) or a string or os.PathLike object containing a
            file name.
        epoch (int): Epoch number (indicates how many epochs have you trained this model).
        model (:class:`torch.nn.Module`): Model to be saved.
        optimizer (Union[:class:`torch.optim.Optimizer`, :class:`colossalai.nn.optimizer`]): Optimizer to be saved.
        lr_scheduler (Union[:class:`torch.optim.lr_scheduler`, :class:`colossalai.nn.lr_scheduler`], optional):
            lr_scheduler to be saved, defaults to None.
        pickle_module: module used for pickling metadata and objects
        pickle_protocol: can be specified to override the default protocol
    """
    # ckpt container
    checkpoint = {"epoch": epoch}

    model_state = model.state_dict()
    if is_using_pp() and gpc.get_local_rank(ParallelMode.TENSOR) == 0:
        model_state = gather_pipeline_parallel_state_dict(model_state)

    if gpc.get_global_rank() == 0:
        checkpoint["model"] = model_state

        # if optimizer is not None:
        #     checkpoint['optimizer'] = optimizer.state_dict()

        # if lr_scheduler is not None:
        #     checkpoint['lr_scheduler'] = lr_scheduler.state_dict()

        torch.save(checkpoint, file, **kwargs)


def broadcast_model(model: torch.nn.Module):
    src_rank = gpc.get_ranks_in_group(ParallelMode.TENSOR)[0]
    for p in model.parameters():
        if not getattr(p, IS_TENSOR_PARALLEL, False) and p.storage().size() > 0:
            group = (
                gpc.get_group(ParallelMode.TENSOR)
                if p.device.type == "cuda"
                else gpc.get_cpu_group(ParallelMode.TENSOR)
            )
            dist.broadcast(p, src_rank, group=group)


def load_checkpoint(
    file,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    strict: bool = True,
):
    """Loads training states from a checkpoint file.

    Args:
        file: a file-like object (has to implement read(), readline(), tell(), and seek()), or a string or os.PathLike
            object containing a file name.
        model (:class:`torch.nn.Module`): Model to load saved weights and buffers.
        optimizer (Union[:class:`torch.optim.Optimizer`, :class:`colossalai.nn.optimizer`]): Optimizer to recuperate.
        lr_scheduler (:class:`torch.optim.lr_scheduler._LRScheduler`, optional):
            lr_scheduler to recuperate, defaults to None.
        strict (bool, optional): Whether to strictly enforce that the keys in :attr:`state_dict`
            of the checkpoint match the names of parameters and buffers in model, defaults to True.

    Returns:
        int: The saved epoch number.

    Raises:
        RuntimeError: Raise error if the model/optimizer cannot successfully be recuperated
    """
    state_dict = (
        torch.load(file, map_location=torch.device("cpu")) if gpc.get_local_rank(ParallelMode.MODEL) == 0 else None
    )

    # model states
    model_state = state_dict.pop("model") if state_dict is not None else dict()
    # pipeline
    if is_using_pp():
        model_state = partition_pipeline_parallel_state_dict(model, model_state)
    try:
        model.load_state_dict(model_state, strict=strict)
        broadcast_model(model)
    except RuntimeError as e:
        error_msgs = str(e)
        if error_msgs.startswith("Error(s) in loading state_dict for "):
            error_msgs = error_msgs.split("\n\t")[1:]
            dst_rank = gpc.get_ranks_in_group(ParallelMode.MODEL)[0]
            all_error_msgs = [None for _ in range(gpc.get_world_size(ParallelMode.MODEL))]
            dist.gather_object(error_msgs, all_error_msgs, dst=dst_rank, group=gpc.get_cpu_group(ParallelMode.MODEL))
            if gpc.get_global_rank() == 0:
                all_error_msgs = list(chain.from_iterable(all_error_msgs))
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        model.__class__.__name__, "\n\t".join(all_error_msgs)
                    )
                )
        else:
            raise e

    # broadcast the rest states
    state_dict = broadcast_state_dict(state_dict, ParallelMode.MODEL)

    # # optimizer states
    # if optimizer is not None and 'optimizer' in state_dict:
    #     optimizer.load_state_dict(state_dict['optimizer'])

    # # lr scheduler states
    # if lr_scheduler is not None and 'lr_scheduler' in state_dict:
    #     lr_scheduler.load_state_dict(state_dict['lr_scheduler'])

    # last epoch
    last_epoch = state_dict.pop("epoch", -1)

    return last_epoch
