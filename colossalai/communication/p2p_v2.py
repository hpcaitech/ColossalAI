#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import io
import pickle
from typing import Any, List, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroupNCCL
from torch.distributed import distributed_c10d as c10d

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc

TensorShape = Union[torch.Size, List[int], Tuple[int]]
_pg_manager = {}
_unpickler = pickle.Unpickler


def init_process_group():
    """intialise process group by dist.new_group in the adjacent stages

    Args:
        None

    Returns:
        None
    """
    world_size = gpc.get_world_size(ParallelMode.PIPELINE)
    for i in range(world_size - 1):
        _pg_manager[(i, i + 1)] = dist.new_group([i, i + 1])


def _acquire_pair_group_handle(first_rank: int, second_rank: int) -> ProcessGroupNCCL:
    """get the group handle of two given ranks

    Args:
        first_rank (int): first rank in the pair
        second_rank (int): second rank in the pair

    Returns:
        :class:`ProcessGroupNCCL`: the handle of the group consisting of the given two ranks
    """
    if len(_pg_manager) == 0:
        init_process_group()
    if first_rank > second_rank:
        first_rank, second_rank = second_rank, first_rank
    pair_key = (first_rank, second_rank)
    return _pg_manager[pair_key]


def _cuda_safe_tensor_to_object(tensor: torch.Tensor, tensor_size: torch.Size) -> object:
    """transform tensor to object with unpickle.
    Info of the device in bytes stream will be modified into current device before unpickling

    Args:
        tensor (:class:`torch.tensor`): tensor to be unpickled
        tensor_size (:class:`torch.Size`): Size of the real info in bytes

    Returns:
        Any: object after unpickled
    """
    buf = tensor.numpy().tobytes()[:tensor_size]
    if b'cuda' in buf:
        buf_array = bytearray(buf)
        device_index = torch.cuda.current_device()
        buf_array[buf_array.find(b'cuda') + 5] = 48 + device_index
        buf = bytes(buf_array)

    io_bytes = io.BytesIO(buf)
    byte_pickler = _unpickler(io_bytes)
    unpickle = byte_pickler.load()

    return unpickle


def _broadcast_object_list(object_list: List[Any], src: int, dst: int, device=None):
    """This is a modified version of the broadcast_object_list in torch.distribution
    The only difference is that object will be move to correct device after unpickled.
    If local_rank = src, then object list will be sent to rank src. Otherwise, object list will
    be updated with data sent from rank src.

    Args:
        object_list (List[Any]): list of object to broadcast
        src (int): source rank to broadcast
        dst (int): dst rank to broadcast
        device (:class:`torch.device`): device to do broadcast. current device in default

    """
    group = _acquire_pair_group_handle(src, dst)

    if c10d._rank_not_in_group(group):
        c10d._warn_not_in_group("broadcast_object_list")
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
            current_device = torch.device("cuda", torch.cuda.current_device())
    if is_nccl_backend:
        object_sizes_tensor = object_sizes_tensor.to(current_device)

    # Broadcast object sizes
    c10d.broadcast(object_sizes_tensor, src=src, group=group, async_op=False)

    # Concatenate and broadcast serialized object tensors
    if local_rank == src:
        object_tensor = torch.cat(tensor_list)
    else:
        object_tensor = torch.empty(    # type: ignore[call-overload]
            torch.sum(object_sizes_tensor).item(),    # type: ignore[arg-type]
            dtype=torch.uint8,
        )

    if is_nccl_backend:
        object_tensor = object_tensor.to(current_device)

    c10d.broadcast(object_tensor, src=src, group=group, async_op=False)

    # Deserialize objects using their stored sizes.
    offset = 0

    if local_rank != src:
        for i, obj_size in enumerate(object_sizes_tensor):
            obj_view = object_tensor[offset:offset + obj_size]
            obj_view = obj_view.type(torch.uint8)
            if obj_view.device != torch.device("cpu"):
                obj_view = obj_view.cpu()
            offset += obj_size
            # unpickle
            unpickle_object = _cuda_safe_tensor_to_object(obj_view, obj_size)

            # unconsistence in device
            if isinstance(unpickle_object,
                          torch.Tensor) and unpickle_object.device.index != torch.cuda.current_device():
                unpickle_object = unpickle_object.cuda()

            object_list[i] = unpickle_object


def _send_object(object: Any, dst: int) -> None:
    """send anything to dst rank
    Args:
        object (Any): object needed to be sent
        dst (int): rank of the destination

    Returns:
        None
    """
    local_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    # handler = _acquire_pair_group_handle(local_rank, dst)

    # transform to list if not
    if isinstance(object, torch.Tensor):
        object = [object]

    # broadcast length first
    # TODO : more elegant ? P.S. reduce a _broadcast_object_list
    _broadcast_object_list([len(object)], local_rank, dst)
    # then broadcast safely
    _broadcast_object_list(object, local_rank, dst)


def _recv_object(src: int) -> Any:
    """recv anything from src

    Args:
        src (int): source rank of data. local rank will receive data from src rank.

    Returns:
        Any: Object received from src.
    """
    local_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    # handler = _acquire_pair_group_handle(local_rank, src)
    # recv length first
    length = [0]
    _broadcast_object_list(length, src, local_rank)

    # then create recv buff from length[0] and broadcast
    object = [None] * length[0]
    _broadcast_object_list(object, src, local_rank)

    if length[0] == 1:
        object = object[0]

    return object


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


def send_forward(output_object: Any, next_rank: int = None) -> None:
    """Sends the input tensor to the next stage in pipeline.

    Args:
        output_object Any: Object to be sent.
        next_rank (int, optional): The rank of the recipient of the tensor.
    """
    if not gpc.is_pipeline_last_stage():
        if next_rank is None:
            next_rank = gpc.get_next_global_rank(ParallelMode.PIPELINE)
        _send_object(output_object, next_rank)


def send_backward(input_object: Any, prev_rank: int = None) -> None:
    """Sends the gradient tensor to the previous stage in pipeline.

    Args:
        input_tensor_grad (Union[:class:`torch.Tensor`, List[:class:`torch.Tensor`]]): Tensor to be sent
        prev_rank (int, optional): The rank of the recipient of the tensor
    """
    if not gpc.is_pipeline_first_stage():
        if prev_rank is None:
            prev_rank = gpc.get_prev_global_rank(ParallelMode.PIPELINE)
        _send_object(input_object, prev_rank)
