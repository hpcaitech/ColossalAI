import torch
import torch.distributed as dist

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import get_current_device


def send_tensor_meta(tensor, need_meta=True, down_group=None):
    """Sends tensor meta information before sending a specific tensor. 
    Since the recipient must know the shape of the tensor in p2p communications,
    meta information of the tensor should be sent before communications. This function
    synchronizes with :func:`recv_tensor_meta`.

    :param tensor: Tensor to be sent
    :param need_meta: If False, meta information won't be sent
    :param down_group: Communication group including the next member in pipeline parallel group
    :type tensor: Tensor
    :type need_meta: bool, optional
    :type down_group: ProcessGroup, optional
    :return: False
    :rtype: bool
    """
    if need_meta:
        rank = gpc.get_global_rank()

        if down_group is None:
            down_group = gpc.get_group(ParallelMode.PIPELINE_NEXT)

        tensor_kwargs = {'dtype': torch.long, 'device': get_current_device()}

        send_shape = torch.tensor(tensor.size(), **tensor_kwargs)
        send_ndims = torch.tensor(len(tensor.size()), **tensor_kwargs)

        dist.broadcast(send_ndims, src=rank, group=down_group)
        dist.broadcast(send_shape, src=rank, group=down_group)

    return False


def recv_tensor_meta(tensor_shape, prev_rank=None, up_group=None):
    """Recieves tensor meta information before recieving a specific tensor. 
    Since the recipient must know the shape of the tensor in p2p communications,
    meta information of the tensor should be recieved before communications. This function
    synchronizes with :func:`send_tensor_meta`.

    :param tensor_shape: The shape of the tensor to be recieved
    :param prev_rank: The rank of the source of the tensor
    :param up_group: Communication group including the previous member in pipeline parallel group
    :type tensor_shape: torch.Size
    :type prev_rank: int, optional
    :type up_group: ProcessGroup, optional
    :return: The shape of the tensor to be recieved
    :rtype: torch.Size
    """
    if tensor_shape is None:
        if prev_rank is None:
            prev_rank = gpc.get_prev_global_rank(
                ParallelMode.PIPELINE)
        if up_group is None:
            up_group = gpc.get_group(ParallelMode.PIPELINE_PREV)

        tensor_kwargs = {'dtype': torch.long, 'device': get_current_device()}

        recv_ndims = torch.empty((), **tensor_kwargs)
        dist.broadcast(recv_ndims, src=prev_rank, group=up_group)

        recv_shape = torch.empty(recv_ndims, **tensor_kwargs)
        dist.broadcast(recv_shape, src=prev_rank, group=up_group)

        tensor_shape = torch.Size(recv_shape)

    return tensor_shape
