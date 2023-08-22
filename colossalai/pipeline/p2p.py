#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import io
import pickle
import re
from typing import Any, List, Optional, Union

import torch
import torch.distributed as dist
from packaging.version import Version
from torch.distributed import ProcessGroup
from torch.distributed import distributed_c10d as c10d

from .stage_manager import PipelineStageManager

_unpickler = pickle.Unpickler


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
        # There might be more than one output tensors during forward
        for cuda_str in re.finditer(b'cuda', buf_array):
            pos = cuda_str.start()
            buf_array[pos + 5] = 48 + device_index
        buf = bytes(buf_array)

    io_bytes = io.BytesIO(buf)
    byte_pickler = _unpickler(io_bytes)
    unpickle = byte_pickler.load()

    return unpickle


def _broadcast_object_list(object_list: List[Any],
                           src: int,
                           group: ProcessGroup,
                           device: Optional[Union[torch.device, str, int]] = None):
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

    if c10d._rank_not_in_group(group):
        c10d._warn_not_in_group("broadcast_object_list")
        return

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

    my_rank = dist.get_rank()
    # Serialize object_list elements to tensors on src rank.
    if my_rank == src:
        if Version(torch.__version__) >= Version("1.13.0"):
            tensor_list, size_list = zip(*[c10d._object_to_tensor(obj, device=current_device) for obj in object_list])
        else:
            tensor_list, size_list = zip(*[c10d._object_to_tensor(obj) for obj in object_list])
        object_sizes_tensor = torch.cat(size_list)
    else:
        object_sizes_tensor = torch.empty(len(object_list), dtype=torch.long)

    if is_nccl_backend:
        object_sizes_tensor = object_sizes_tensor.to(current_device)

    # Broadcast object sizes
    c10d.broadcast(object_sizes_tensor, src=src, group=group, async_op=False)

    # Concatenate and broadcast serialized object tensors
    if my_rank == src:
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

    if my_rank != src:
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


def _send_object(object: Any, src: int, dst: int, group: ProcessGroup) -> None:
    """send anything to dst rank

    Args:
        object (Any): object needed to be sent
        dst (int): rank of the destination

    Returns:
        None
    """
    # then broadcast safely
    _broadcast_object_list([object], src, group)


def _recv_object(src: int, dst: int, group: ProcessGroup) -> Any:
    """recv anything from src

    Args:
        src (int): source rank of data. local rank will receive data from src rank.

    Returns:
        Any: Object received from src.
    """
    object_list = [None]
    _broadcast_object_list(object_list, src, group)

    return object_list[0]


class PipelineP2PCommunication:

    def __init__(self, stage_manager: PipelineStageManager) -> None:
        self.stage_manager = stage_manager

    def recv_forward(self, prev_rank: int = None) -> Any:
        """Copy the forward output from the previous stage in pipeline as the input tensor of this stage.

        Args:
            prev_rank (int, optional): The rank of the source of the tensor.

        Returns:
            Any: The input tensor or input tensor list.
        """
        if prev_rank is None:
            prev_rank = self.stage_manager.get_prev_rank()
        cur_rank = self.stage_manager.get_rank()
        input_tensor = _recv_object(prev_rank, cur_rank, self.stage_manager.get_p2p_process_group(prev_rank, cur_rank))

        return input_tensor

    def recv_backward(self, next_rank: int = None) -> Any:
        """Copy the gradient tensor from the next stage in pipeline as the input gradient of this stage.

        Args:
            next_rank (int, optional): The rank of the source of the tensor.

        Returns:
            Any: The input gradient tensor or gradient tensor list.
        """
        if next_rank is None:
            next_rank = self.stage_manager.get_next_rank()
        cur_rank = self.stage_manager.get_rank()
        output_tensor_grad = _recv_object(next_rank, cur_rank,
                                          self.stage_manager.get_p2p_process_group(next_rank, cur_rank))

        return output_tensor_grad

    def send_forward(self, output_object: Any, next_rank: int = None) -> None:
        """Sends the input tensor to the next stage in pipeline.

        Args:
            output_object (Any): Object to be sent.
            next_rank (int, optional): The rank of the recipient of the tensor.
        """
        if next_rank is None:
            next_rank = self.stage_manager.get_next_rank()
        cur_rank = self.stage_manager.get_rank()
        _send_object(output_object, cur_rank, next_rank, self.stage_manager.get_p2p_process_group(cur_rank, next_rank))

    def send_backward(self, input_object: Any, prev_rank: int = None) -> None:
        """Sends the gradient tensor to the previous stage in pipeline.

        Args:
            input_object (Any): Object to be sent.
            prev_rank (int, optional): The rank of the recipient of the tensor
        """
        if prev_rank is None:
            prev_rank = self.stage_manager.get_prev_rank()
        cur_rank = self.stage_manager.get_rank()
        _send_object(input_object, cur_rank, prev_rank, self.stage_manager.get_p2p_process_group(cur_rank, prev_rank))
