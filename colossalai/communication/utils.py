import torch
import torch.distributed as dist

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import get_current_device
from typing import Union, List, Tuple

TensorShape = Union[torch.Size, List[int], Tuple[int]]


def send_meta_helper(obj, next_rank, tensor_kwargs):
    send_shape = torch.tensor(obj.size(), **tensor_kwargs)
    send_ndims = torch.tensor(len(obj.size()), **tensor_kwargs)
    dist.send(send_ndims, next_rank)
    dist.send(send_shape, next_rank)


def send_obj_meta(obj, need_meta=True, next_rank=None) -> bool:
    """Sends obj meta information before sending a specific obj.
    Since the recipient must know the shape of the obj in p2p communications,
    meta information of the obj should be sent before communications. This function
    synchronizes with :func:`recv_obj_meta`.

    Args:
        obj (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): obj to be sent.
        need_meta (bool, optional): If False, meta information won't be sent.
        next_rank (int): The rank of the next member in pipeline parallel group.

    Returns:
        bool: False
    """
    if need_meta:
        if next_rank is None:
            next_rank = gpc.get_next_global_rank(ParallelMode.PIPELINE)

        tensor_kwargs = {'dtype': torch.long, 'device': get_current_device()}
        if isinstance(obj, torch.Tensor):
            send_obj_nums = torch.tensor(1, **tensor_kwargs)
            dist.send(send_obj_nums, next_rank)
            send_meta_helper(obj, next_rank, tensor_kwargs)
        else:
            send_obj_nums = torch.tensor(len(obj), **tensor_kwargs)
            dist.send(send_obj_nums, next_rank)
            for tensor_to_send in obj:
                send_meta_helper(tensor_to_send, next_rank, tensor_kwargs)

    return False


def recv_meta_helper(prev_rank, tensor_kwargs):
    recv_ndims = torch.empty((), **tensor_kwargs)
    dist.recv(recv_ndims, prev_rank)
    recv_shape = torch.empty(recv_ndims, **tensor_kwargs)
    dist.recv(recv_shape, prev_rank)
    return recv_shape


def recv_obj_meta(obj_shape, prev_rank=None) -> torch.Size:
    """Receives obj meta information before receiving a specific obj.
    Since the recipient must know the shape of the obj in p2p communications,
    meta information of the obj should be received before communications. This function
    synchronizes with :func:`send_obj_meta`.

    Args:
        obj_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the obj to be received.
        prev_rank (int): The rank of the source of the obj.

    Returns:
        Union[:class:`torch.Size`, List[:class:`torch.Size`]]: The shape of the obj to be received.
    """
    if obj_shape is None:
        if prev_rank is None:
            prev_rank = gpc.get_prev_global_rank(ParallelMode.PIPELINE)

        tensor_kwargs = {'dtype': torch.long, 'device': get_current_device()}
        recv_obj_nums = torch.empty((), **tensor_kwargs)
        dist.recv(recv_obj_nums, prev_rank)
        if recv_obj_nums.item() == 1:
            recv_shape = recv_meta_helper(prev_rank, tensor_kwargs)
            obj_shape = torch.Size(recv_shape)
        else:
            obj_shape = []
            for i in range(recv_obj_nums.item()):
                recv_shape = recv_meta_helper(prev_rank, tensor_kwargs)
                obj_shape.append(torch.Size(recv_shape))

    return obj_shape


def split_tensor_into_1d_equal_chunks(tensor: torch.Tensor, new_buffer=False) -> torch.Tensor:
    """Break a tensor into equal 1D chunks.

    Args:
        tensor (:class:`torch.Tensor`): Tensor to be split before communication.
        new_buffer (bool, optional): Whether to use a new buffer to store sliced tensor.

    Returns:
        :class:`torch.Tensor`: The split tensor
    """
    partition_size = torch.numel(tensor) // gpc.get_world_size(ParallelMode.PARALLEL_1D)
    start_index = partition_size * gpc.get_local_rank(ParallelMode.PARALLEL_1D)
    end_index = start_index + partition_size
    if new_buffer:
        data = torch.empty(partition_size, dtype=tensor.dtype, device=torch.cuda.current_device(), requires_grad=False)
        data.copy_(tensor.view(-1)[start_index:end_index])
    else:
        data = tensor.view(-1)[start_index:end_index]
    return data


def gather_split_1d_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Opposite of above function, gather values from model parallel ranks.

    Args:
        tensor (:class:`torch.Tensor`): Tensor to be gathered after communication.
    Returns:
        :class:`torch.Tensor`: The gathered tensor.
    """
    world_size = gpc.get_world_size(ParallelMode.PARALLEL_1D)
    numel = torch.numel(tensor)
    numel_gathered = world_size * numel
    gathered = torch.empty(numel_gathered, dtype=tensor.dtype, device=torch.cuda.current_device(), requires_grad=False)
    chunks = [gathered[i * numel:(i + 1) * numel] for i in range(world_size)]
    dist.all_gather(chunks, tensor, group=gpc.get_group(ParallelMode.PARALLEL_1D))
    return gathered
