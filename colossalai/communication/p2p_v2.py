#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import List, Tuple, Union, Any, Dict
import pickle
import io

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d
from torch.distributed import ProcessGroupNCCL
# TODO remove it when release
from colorama import Back, Style

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc


TensorShape = Union[torch.Size, List[int], Tuple[int]]
_pg_manager = {}

def init_process_group():
    """
    intialise process group by dist.new_group in the adjacent stages
    """
    world_size = gpc.get_world_size(ParallelMode.PIPELINE)
    for i in range(world_size - 1):
        _pg_manager[(i, i + 1)] = dist.new_group([i, i + 1])
    

def _acquire_pair_group_handle(first_rank: int, second_rank: int) -> ProcessGroupNCCL:
    """
    get the group handle of two given ranks
    """
    if len(_pg_manager) == 0:
        init_process_group()
    if first_rank > second_rank:
        first_rank, second_rank = second_rank, first_rank
    pair_key = (first_rank, second_rank)
    return _pg_manager[pair_key]

def _cuda_safe_tensor_to_object(tensor: torch.Tensor, tensor_size: torch.Size) -> object:
    buf = tensor.numpy().tobytes()[:tensor_size]
    if b'cuda' in buf:
        buf_array = bytearray(buf)
        device_index = torch.cuda.current_device()
        buf_array[buf_array.find(b'cuda') + 5] = 48 + device_index 
        buf = bytes(buf_array)

    io_bytes = io.BytesIO(buf)
    byte_pickler = pickle.Unpickler(io_bytes)
    unpickle = byte_pickler.load()
    
    return unpickle


def _broadcast_object_list(object_list, src, dst, device=None, async_op=False):
    """
    This is the adjust to the broadcast_object_list in torch.distribution
    The only difference is that object will be move to correct device after unpickled
    """
    group = _acquire_pair_group_handle(src, dst)

    if c10d._rank_not_in_group(group):
        # c10d._warn_not_in_group("broadcast_object_list")
        print(Back.RED, "ERROR", Style.RESET_ALL, "{} and {} has abnormal reflection, broadcast failed!".format(src, dst))
        return
    
    local_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    # Serialize object_list elements to tensors on src rank.
    if local_rank == src:
        tensor_list, size_list = zip(*[c10d._object_to_tensor(obj) for obj in object_list])
        object_sizes_tensor = torch.cat(size_list)
    else:
        object_sizes_tensor = torch.empty(len(object_list), dtype=torch.long)
    
    is_nccl_backend = c10d._check_for_nccl_backend(group)
    current_device = None

    if device is not None:
        if is_nccl_backend and device.type != "cuda":
            raise ValueError("device type must be cuda for nccl backend")
        current_device = device
    else:
        current_device = torch.device("cpu")
        if is_nccl_backend:
            # See note about using torch.cuda.current_device() here in
            # docstring. We cannot simply use my_rank since rank == device is
            # not necessarily true.
            current_device = torch.device("cuda", torch.cuda.current_device())
    if is_nccl_backend:
        object_sizes_tensor = object_sizes_tensor.to(current_device)

    # Broadcast object sizes
    c10d.broadcast(object_sizes_tensor, src=src, group=group, async_op=async_op)
    # print(Back.CYAN, "inner broadcast length", Style.RESET_ALL, "{} finish {} {}".format(local_rank, local_rank, src))
            
    # Concatenate and broadcast serialized object tensors
    if local_rank == src:
        object_tensor = torch.cat(tensor_list)
    else:
        object_tensor = torch.empty(  # type: ignore[call-overload]
            torch.sum(object_sizes_tensor).item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
        )
    
    if is_nccl_backend:
        object_tensor = object_tensor.to(current_device)

    c10d.broadcast(object_tensor, src=src, group=group, async_op=async_op)
    # print(Back.CYAN, "inner broadcast content", Style.RESET_ALL, "rank_{} finish {} {}".format(local_rank, local_rank, src))

    # Deserialize objects using their stored sizes.
    offset = 0
    
    if local_rank != src:
        for i, obj_size in enumerate(object_sizes_tensor):
            obj_view = object_tensor[offset : offset + obj_size]
            obj_view = obj_view.type(torch.uint8)
            if obj_view.device != torch.device("cpu"):
                obj_view = obj_view.cpu()
            offset += obj_size
            # unpickle
            unpickle_object = _cuda_safe_tensor_to_object(obj_view, obj_size)
            
            # unconsistence in device
            if isinstance(unpickle_object, torch.Tensor) and unpickle_object.device.index != torch.cuda.current_device():
                unpickle_object = unpickle_object.cuda()

            object_list[i] = unpickle_object
        # print(Back.BLUE, "this is rank_{}".format(gpc.get_local_rank(ParallelMode.PIPELINE)), Style.RESET_ALL, object_list)
        
    # print(Back.GREEN, "rank_{} finish _broadcast_object_list".format(local_rank), Style.RESET_ALL)


def _send_object(object: Any, dst: int) -> None:
    """
    send anything to dst (nccl backend only)
    """
    local_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    # handler = _acquire_pair_group_handle(local_rank, dst)

    # transform to list if not
    if isinstance(object, torch.Tensor):
        object = [object]

    # broadcast length first
    # TODO : more elegant ? P.S. reduce a _broadcast_object_list
    _broadcast_object_list([len(object)], local_rank, dst)
    # print(Back.LIGHTMAGENTA_EX, "[_send_object]", Style.RESET_ALL, "rank_{} send length {} to rank_{}".format(local_rank, [len(object)], dst))
    # then broadcast safely
    _broadcast_object_list(object, local_rank, dst)

    # print(Back.LIGHTGREEN_EX, "[_send_object]", Style.RESET_ALL, "rank_{} send {} to rank_{}".format(local_rank, type(object), dst))


def _recv_object(src: int) -> Any:
    """
    recv anything from src (nccl backend only)
    """
    local_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    # handler = _acquire_pair_group_handle(local_rank, src)
    # recv length first
    length = [0]
    # print(Back.LIGHTYELLOW_EX, "[_recv_object]", Style.RESET_ALL, "rank_{} waiting for msg from rank_{}".format(local_rank, src))
    _broadcast_object_list(length, src, local_rank)

    # then create recv buff from length[0] and broadcast
    object = [None] * length[0]
    # print(Back.MAGENTA, "[_recv_object]", Style.RESET_ALL, "rank_{} recv length {} from rank_{}".format(local_rank, length, src))
    _broadcast_object_list(object, src, local_rank)


    if length[0] == 1:
        object = object[0]
    # print(Back.GREEN, "[_recv_object]", Style.RESET_ALL, "rank_{} recv {} from rank_{}".format(local_rank, type(object), src))

    return object


def _communicate(object_send_next: Union[torch.Tensor, List[torch.Tensor]] = None,
                 object_send_prev: Union[torch.Tensor, List[torch.Tensor]] = None,
                 recv_prev: bool = False,
                 recv_next: bool = False,
                 recv_prev_shape: Union[torch.Size, List[torch.Size]] = None,
                 recv_next_shape: Union[torch.Size, List[torch.Size]] = None,
                 prev_rank: int = None,
                 next_rank: int = None,
                 dtype: torch.dtype = None,
                 scatter_gather_tensors: bool = False) -> Tuple[Union[torch.Tensor, List[torch.Tensor]]]:
    """
    Adapted from megatron.p2p_communication.
    Communicate tensors between stages. Used as helper method in other
    communication methods that are used in pipeline schedule.
    Takes the following arguments:
        object_send_next (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): tensor to send to next rank (no tensor sent if
                          set to None).
        object_send_prev (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev (bool): boolean for whether tensor should be received from
                   previous rank.
        recv_next (bool): boolean for whether tensor should be received from
                   next rank.
        recv_prev_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): shape of the tensor to be received from the previous stage, defualts to None.
        recv_next_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): shape of the tensor to be received from the next stage, defualts to None.
        prev_rank (int): the rank of the previous pipeline stage, defualts to None,
        next_rank (int): the rank of the next pipeline stage, defualts to None,
        dtype (torch.dtype): data type of intermediate buffers, defaults to None
        scatter_gather_tensors (bool): whether to scatter and gather tensor between pipeline stages, defaults to False

    Returns:
        Tuple[Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]]: returns tensor_recv_prev, tensor_recv_next
    """

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None
    

    if object_send_prev is not None or recv_prev:
        if prev_rank is None:
            prev_rank = gpc.get_prev_global_rank(ParallelMode.PIPELINE)

    if object_send_next is not None or recv_next:
        if next_rank is None:
            next_rank = gpc.get_next_global_rank(ParallelMode.PIPELINE)

    
    if object_send_prev is not None:
        _send_object(object_send_prev, prev_rank)

    if object_send_next is not None:
        _send_object(object_send_next, next_rank)

    if tensor_recv_prev is not None:
        tensor_recv_prev = _recv_object(prev_rank)

    if tensor_recv_next is not None:
        tensor_recv_next = _recv_object(next_rank)

    return tensor_recv_prev, tensor_recv_next


def recv_forward(prev_rank: int = None) -> Any:
    """Copy the forward output from the previous stage in pipeline as the input tensor of this stage.

    Args:
        input_tensor_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor to be received.
        prev_rank (int, optional): The rank of the source of the tensor.

    Returns:
        Any: The input tensor or input tensor list.
    """
    if gpc.is_pipeline_first_stage():
        input_tensor = None
    else:
        if prev_rank is None:
            prev_rank = gpc.get_prev_global_rank(ParallelMode.PIPELINE)
        input_tensor = _recv_object(prev_rank)

    return input_tensor


def recv_backward(next_rank: int = None) -> Any:
    """Copy the gradient tensor from the next stage in pipeline as the input gradient of this stage.

    Args:
        output_grad_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor to be received.
        next_rank (int, optional): The rank of the source of the tensor.

    Returns:
        Any: The input gradient tensor or gradident tensor list.
    """
    if gpc.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if next_rank is None:
            next_rank = gpc.get_next_global_rank(ParallelMode.PIPELINE)
        output_tensor_grad = _recv_object(next_rank)
    
    return output_tensor_grad


def send_forward(output_object : Any, next_rank: int = None) -> None:
    """Sends the input tensor to the next stage in pipeline.

    Args:
        output_object Any: Object to be sent.
        next_rank (int, optional): The rank of the recipient of the tensor.
    """
    if not gpc.is_pipeline_last_stage():
        if next_rank is None:
            next_rank = gpc.get_next_global_rank(ParallelMode.PIPELINE)
        _send_object(output_object, next_rank)


def send_backward(input_object: Any, prev_rank: int=None) -> None:
    """Sends the gradient tensor to the previous stage in pipeline.

    Args:
        input_tensor_grad (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Tensor to be sent
        prev_rank (int, optional): The rank of the recipient of the tensor
    """
    if not gpc.is_pipeline_first_stage():
        if prev_rank is None:
            prev_rank = gpc.get_prev_global_rank(ParallelMode.PIPELINE)
        _send_object(input_object, prev_rank)


def send_forward_recv_backward():
    """reserve
    """
    pass


def send_backward_recv_forward():
    """reserve
    """
    pass


def send_forward_recv_forward():
    """reserve
    """
    pass


def send_backward_recv_backward():
    """reserve
    """
    pass


def send_forward_backward_recv_forward_backward(
        output_tensor,
        input_tensor_grad,
        input_tensor_shape,
        output_grad_shape,
        recv_prev=True,
        recv_next=True,
        prev_rank=None,
        next_rank=None,
        dtype=torch.float,
        scatter_gather_tensors=False) -> Tuple[Union[torch.Tensor, List[torch.Tensor]]]:
    """Batched communication operation. Sends the input tensor to the next stage in pipeline and
    the gradient tensor to the previous stage, while receives the input gradient tensor from the
    next stage and the input tensor from the previous stage.

    Args:
        output_tensor (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Tensor sent to the next.
        input_tensor_grad (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Tensor sent to the previous.
        input_tensor_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor received from the previous.
        output_grad_shape (Union[:class:`torch.Size`, List[:class:`torch.Size`]]): The shape of the tensor received from the next.

    Returns:
        Tuple(Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]], Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): (the input tensor, the input gradient tensor)
    """
    input_tensor, output_tensor_grad = _communicate(object_send_next=output_tensor,
                                                    object_send_prev=input_tensor_grad,
                                                    recv_prev=recv_prev,
                                                    recv_next=recv_next,
                                                    recv_prev_shape=input_tensor_shape,
                                                    recv_next_shape=output_grad_shape,
                                                    prev_rank=prev_rank,
                                                    next_rank=next_rank,
                                                    dtype=dtype,
                                                    scatter_gather_tensors=scatter_gather_tensors)
    return input_tensor, output_tensor_grad
