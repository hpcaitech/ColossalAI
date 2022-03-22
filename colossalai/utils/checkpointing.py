from collections import OrderedDict
from itertools import chain

import torch
import torch.distributed as dist
from colossalai.communication.collective import scatter_object_list
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX

from .common import is_using_pp

__all__ = ['save_checkpoint', 'load_checkpoint']


def broadcast_state_dict(state_dict, parallel_mode):
    state_dict = [state_dict.copy()]
    src_rank = gpc.get_ranks_in_group(parallel_mode)[0]
    dist.broadcast_object_list(state_dict, src=src_rank, group=gpc.get_cpu_group(parallel_mode))
    return state_dict[0]


def partition_tensor_parallel_state_dict(state_dict: OrderedDict,
                                         parallel_mode: ParallelMode,
                                         dims: dict = dict(),
                                         partition_states: dict = dict()):
    src_rank = gpc.get_ranks_in_group(parallel_mode)[0]
    depth = gpc.get_world_size(parallel_mode)

    if gpc.get_local_rank(parallel_mode) == 0:

        partitioned_state_list = [dict() for _ in range(depth)]

        for key in list(state_dict.keys()):
            param = state_dict.pop(key)
            dim = dims.get(key, 0)
            do_partition = partition_states.get(key, True)
            if do_partition:
                param = torch.chunk(param, depth, dim=dim)
            for i, p in enumerate(partitioned_state_list):
                p[key] = param[i] if do_partition else param

    else:
        partitioned_state_list = [None for _ in range(depth)]

    partitioned_state = [None]
    scatter_object_list(partitioned_state, partitioned_state_list, src=src_rank, group=gpc.get_cpu_group(parallel_mode))
    return partitioned_state[0]


def gather_tensor_parallel_state_dict(state_dict: OrderedDict,
                                      parallel_mode: ParallelMode,
                                      dims: dict = dict(),
                                      partition_states: dict = dict(),
                                      keep_vars: bool = False):
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
            extra_state_key = name + '.' + _EXTRA_STATE_KEY_SUFFIX
            if extra_state_key in state_dict:
                pipeline_state[extra_state_key] = state_dict.pop(extra_state_key)
        # send rest states to next stage
        if not gpc.is_last_rank(ParallelMode.PIPELINE):
            _send_state_dict(state_dict, gpc.get_next_global_rank(ParallelMode.PIPELINE), ParallelMode.PIPELINE)

    return pipeline_state


def gather_pipeline_parallel_state_dict(state_dict):
    gathered_states = [None for _ in range(gpc.get_world_size(ParallelMode.PIPELINE))] \
        if gpc.get_local_rank(ParallelMode.PIPELINE) == 0 else None
    dist.gather_object(state_dict,
                       gathered_states,
                       dst=gpc.get_ranks_in_group(ParallelMode.PIPELINE)[0],
                       group=gpc.get_cpu_group(ParallelMode.PIPELINE))

    state_dict = OrderedDict(chain.from_iterable(state.items() for state in gathered_states)) \
        if gpc.get_local_rank(ParallelMode.PIPELINE) == 0 else OrderedDict()

    return state_dict


def save_checkpoint(file,
                    epoch: int,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer = None,
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                    **kwargs):
    if gpc.get_local_rank(ParallelMode.DATA) == 0:
        # ckpt container
        checkpoint = {'epoch': epoch}

        model_state = model.state_dict()
        if is_using_pp() and gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            model_state = gather_pipeline_parallel_state_dict(model_state)
        checkpoint['model'] = model_state

        # if optimizer is not None:
        #     checkpoint['optimizer'] = optimizer.state_dict()

        # if lr_scheduler is not None:
        #     checkpoint['lr_scheduler'] = lr_scheduler.state_dict()

        if gpc.get_global_rank() == 0:
            torch.save(checkpoint, file, **kwargs)


def load_checkpoint(file,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer = None,
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                    strict: bool = True):
    state_dict = torch.load(file, map_location=torch.device('cpu')) \
        if gpc.get_local_rank(ParallelMode.MODEL) == 0 else None

    # model states
    model_state = state_dict.pop('model') if state_dict is not None else dict()
    # pipeline
    if is_using_pp():
        model_state = partition_pipeline_parallel_state_dict(model, model_state)
    try:
        model.load_state_dict(model_state, strict=strict)
    except RuntimeError as e:
        error_msgs = str(e)
        if error_msgs.startswith('Error(s) in loading state_dict for '):
            error_msgs = error_msgs.split("\n\t")[1:]
            dst_rank = gpc.get_ranks_in_group(ParallelMode.MODEL)[0]
            all_error_msgs = [None for _ in range(gpc.get_world_size(ParallelMode.MODEL))]
            dist.gather_object(error_msgs, all_error_msgs, dst=dst_rank, group=gpc.get_cpu_group(ParallelMode.MODEL))
            if gpc.get_global_rank() == 0:
                all_error_msgs = list(chain.from_iterable(all_error_msgs))
                raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                    model.__class__.__name__, "\n\t".join(all_error_msgs)))
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
    last_epoch = state_dict.pop('epoch', -1)

    return last_epoch


# def save_checkpoint(checkpoint_path: str,
#                     epoch: int,
#                     model: torch.nn.Module,
#                     optimizer: torch.optim.Optimizer,
#                     lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
#                     **kwargs):
#     """Given a directory to store the checkpoints, saves all the training components' parameters or buffers, such as model,
#      optimizer, lr_scheduler and etc. into a checkpoint dictionary.

#     This method can be used for both colosalai nn.BaseModel and normal pytorch nn.Module.

#     :param checkpoint_path: Set up a directory for saving checkpoints
#     :type checkpoint_path: str
#     :param epoch: Epoch number (indicate how many epochs have you trained this model)
#     :type epoch: int
#     :param model: Model to be registered
#     :type model: torch.nn.Module
#     :param optimizer: Optimizer to be registered
#     :type optimizer: torch.optim.Optimizer
#     :param lr_scheduler: lr_scheduler to be registered, defaults to None
#     :type lr_scheduler: torch.optim.lr_scheduler._LRScheduler, optional
#     """
#     # for compatibility with normal pytorch nn.Module
#     if hasattr(model, 'state_dict_for_save_checkpoint'):
#         model_sd = model.state_dict_for_save_checkpoint()
#     else:
#         model_sd = model.state_dict()
#     # ckpt container
#     checkpoint = {'epoch': epoch, 'model': model_sd, 'optimizer': optimizer.state_dict(), **kwargs}
#     if lr_scheduler is not None:
#         checkpoint['lr_scheduler'] = lr_scheduler.state_dict()

#     _ensure_directory_exists(checkpoint_path)
#     torch.save(checkpoint, checkpoint_path)

# def load_checkpoint(checkpoint_path: str,
#                     old_tp_size: int,
#                     old_pp_size: int,
#                     epoch: int,
#                     model: torch.nn.Module,
#                     optimizer: torch.optim.Optimizer = None,
#                     lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
#                     suffix: str = '',
#                     finetune: bool = False,
#                     strict: bool = True) -> Tuple:
#     """Loads the checkpoint file.
#     If finetune is False, then we intend to continue/resume the training process from the checkpoint given.
#     So we copy parameters and buffers from state_dict into these modules(model, optimizer,lr_scheduler)
#      and its descendants.
#     If finetune is True, then only the weights and buffers of model should be reload.
#     If strict is True, then the keys of state_dict must exactly match the keys returned by this module's
#      state_dict() function.

#     :param checkpoint_path: The exact and matched checkpoint_path directory to retrieve appropriate state_dict
#     :type checkpoint_path: str
#     :param model: Model to reload parameters and buffers
#     :type model: torch.nn.Module
#     :param optimizer: Optimizer to recuperate
#     :type optimizer: torch.optim.Optimizer
#     :param lr_scheduler: lr_scheduler to recuperate, defaults to None
#     :type lr_scheduler: torch.optim.lr_scheduler._LRScheduler, optional
#     :param finetune: Whether to finetune the model with new dataset or continue the pre-training, defaults to False
#     :type finetune: bool, optional
#     :param strict: Whether to strictly enforce that the keys in
#         :attr:`state_dict` of the checkpoint match the names of
#         parameters and buffers in model., defaults to True
#     :type strict: bool, optional
#     :raises ValueError: Raise error if the model/optimizer cannot successfully be recuperated
#     :return: (the epoch number of the checkpoint retrieved, the checkpoint retrieved)
#     :rtype: Tuple

#     """
#     # Load the checkpoint.
#     checkpoint = colossalai_load(checkpoint_path, old_tp_size, old_pp_size, epoch, suffix)  # load
#     try:
#         last_epoch = checkpoint.pop('epoch') if not finetune else 0
#         model.load_state_dict(checkpoint.pop('model'), strict=strict)
#     except KeyError:
#         raise ValueError('Checkpoint is corrupted')

#     # if not finetune:
#     #     try:
#     #         optimizer.load_state_dict(checkpoint.pop('optimizer'))
#     #     except KeyError:
#     #         raise ValueError('Checkpoint is corrupted')

#     #     if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
#     #         lr_scheduler.load_state_dict(checkpoint.pop('lr_scheduler'))
#     # else:
#     #     lr_scheduler = None
#     #     optimizer = None
#     #     assert lr_scheduler is None, "Optimizer and lr_scheduler should be None when finetune is true"
#     #     assert optimizer is None , "Optimizer and lr_scheduler should be None when finetune is true"
#     lr_scheduler = None
#     optimizer = None
#     if optimizer is not None:
#         try:
#             optimizer.load_state_dict(checkpoint.pop('optimizer'))
#         except ValueError:
#             raise ValueError(
#                 'Optimizer should be None to load parameters with a different setting, finetune or inference')
#     if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
#         try:
#             lr_scheduler.load_state_dict(checkpoint.pop('lr_scheduler'))
#         except ValueError:
#             raise ValueError(
#                 'LR_scheduler should be None to load parameters with a different setting, finetune or inference')

#     return last_epoch, checkpoint
