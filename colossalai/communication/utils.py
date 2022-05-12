import torch
import torch.distributed as dist

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import get_current_device
from typing import Union, List, Tuple

TensorShape = Union[torch.Size, List[int], Tuple[int]]


def send_tensor_meta(tensor, need_meta=True, next_rank=None) -> bool:
    """Sends tensor meta information before sending a specific tensor.
    Since the recipient must know the shape of the tensor in p2p communications,
    meta information of the tensor should be sent before communications. This function
    synchronizes with :func:`recv_tensor_meta`.

    Args:
        tensor (:class:`torch.Tensor`): Tensor to be sent.
        need_meta (bool, optional): If False, meta information won't be sent.
        next_rank (int): The rank of the next member in pipeline parallel group.

    Returns:
        bool: False
    """
    if need_meta:
        if next_rank is None:
            next_rank = gpc.get_next_global_rank(ParallelMode.PIPELINE)

        tensor_kwargs = {'dtype': torch.long, 'device': get_current_device()}

        send_shape = torch.tensor(tensor.size(), **tensor_kwargs)
        send_ndims = torch.tensor(len(tensor.size()), **tensor_kwargs)
        dist.send(send_ndims, next_rank)
        dist.send(send_shape, next_rank)

    return False


def recv_tensor_meta(tensor_shape: TensorShape, prev_rank=None) -> torch.Size:
    """Receives tensor meta information before receiving a specific tensor.
    Since the recipient must know the shape of the tensor in p2p communications,
    meta information of the tensor should be received before communications. This function
    synchronizes with :func:`send_tensor_meta`.

    Args:
        tensor_shape (:class:`torch.Size`): The shape of the tensor to be received.
        prev_rank (int): The rank of the source of the tensor.

    Returns:
        :class:`torch.Size`: The shape of the tensor to be received.
    """
    if tensor_shape is None:
        if prev_rank is None:
            prev_rank = gpc.get_prev_global_rank(ParallelMode.PIPELINE)

        tensor_kwargs = {'dtype': torch.long, 'device': get_current_device()}

        recv_ndims = torch.empty((), **tensor_kwargs)
        dist.recv(recv_ndims, prev_rank)
        recv_shape = torch.empty(recv_ndims, **tensor_kwargs)
        dist.recv(recv_shape, prev_rank)

        tensor_shape = torch.Size(recv_shape)

    return tensor_shape


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
