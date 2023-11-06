#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import io
import pickle
import re
from typing import Any, List, Optional, Union, Dict

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
    if b"cuda" in buf:
        buf_array = bytearray(buf)
        device_index = torch.cuda.current_device()
        # There might be more than one output tensors during forward
        for cuda_str in re.finditer(b"cuda", buf_array):
            pos = cuda_str.start()
            buf_array[pos + 5] = 48 + device_index
        buf = bytes(buf_array)

    io_bytes = io.BytesIO(buf)
    byte_pickler = _unpickler(io_bytes)
    unpickle = byte_pickler.load()

    return unpickle


def check_for_nccl_backend(group):
    pg = group or c10d._get_default_group()
    # Gate PG wrapper check on Gloo availability.
    if c10d._GLOO_AVAILABLE:
        # It is not expected for PG to be wrapped many times, but support it just
        # in case
        while isinstance(pg, c10d._ProcessGroupWrapper):
            pg = pg.wrapped_pg

    return (
        c10d.is_nccl_available() and
        pg.name() == c10d.Backend.NCCL
    )


def _broadcast_object_list(
    object_list: List[Any], src: int, group: ProcessGroup, device: Optional[Union[torch.device, str, int]] = None
):
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

    is_nccl_backend = check_for_nccl_backend(group)
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
        object_tensor = torch.empty(  # type: ignore[call-overload]
            torch.sum(object_sizes_tensor).item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
        )

    if is_nccl_backend:
        object_tensor = object_tensor.to(current_device)

    c10d.broadcast(object_tensor, src=src, group=group, async_op=False)

    # Deserialize objects using their stored sizes.
    offset = 0

    if my_rank != src:
        for i, obj_size in enumerate(object_sizes_tensor):
            obj_view = object_tensor[offset: offset + obj_size]
            obj_view = obj_view.type(torch.uint8)
            if obj_view.device != torch.device("cpu"):
                obj_view = obj_view.cpu()
            offset += obj_size
            # unpickle
            unpickle_object = _cuda_safe_tensor_to_object(obj_view, obj_size)

            # unconsistence in device
            if (
                isinstance(unpickle_object, torch.Tensor)
                and unpickle_object.device.index != torch.cuda.current_device()
            ):
                unpickle_object = unpickle_object.cuda()

            object_list[i] = unpickle_object


def check_device(group):
    is_nccl_backend = check_for_nccl_backend(group)
    current_device = None

    current_device = torch.device("cpu")
    if is_nccl_backend:
        current_device = torch.device("cuda", torch.cuda.current_device())
    return current_device, is_nccl_backend


def filling_ops_queue(obj, comm_op, comm_rank, ops_queue, group):
    if isinstance(obj, torch.Tensor):
        op_to_add = dist.P2POp(comm_op, obj, comm_rank, group)
        ops_queue.append(op_to_add)
    else:
        for tensor_to_comm in obj:
            op_to_add = dist.P2POp(comm_op, tensor_to_comm, comm_rank, group)
            ops_queue.append(op_to_add)


def create_recv_buffer(tensor_metadata, current_device):
    if tensor_metadata[0] == 0:
        tensor_shape, tensor_dtype, tensor_requires_grad = tensor_metadata[1]
        tensor_recv = torch.empty(
            tensor_shape, requires_grad=tensor_requires_grad, device=current_device, dtype=tensor_dtype)
        return tensor_recv
    elif tensor_metadata[0] == 1 or tensor_metadata[0] == 2:
        buffer_recv = []
        for tensor_data in tensor_metadata[1:]:
            tensor_shape, tensor_dtype, tensor_requires_grad = tensor_data[-3:]
            tensor_recv = torch.empty(
                tensor_shape, requires_grad=tensor_requires_grad, device=current_device, dtype=tensor_dtype)
            buffer_recv.append(tensor_recv)
        return buffer_recv


def _batch_send_recv_tensor(send_tensor_list, recv_tensor_metadata, send_dst, recv_src, group, current_device):
    buffer_recv = None
    if recv_tensor_metadata is not None:
        buffer_recv = create_recv_buffer(recv_tensor_metadata, current_device)

    ops = []

    if send_dst is not None:
        filling_ops_queue(send_tensor_list, dist.isend, send_dst, ops, group)

    if recv_src is not None:
        assert buffer_recv is not None
        filling_ops_queue(buffer_recv, dist.irecv, recv_src, ops, group)

    if len(ops) > 0:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
    torch.cuda.synchronize()
    return buffer_recv


def _send_object_with_serialization(object: Any, dst: int, group: ProcessGroup, current_device, is_nccl_backend):
    if Version(torch.__version__) >= Version("1.13.0"):
        object_tensor, object_size_tensor = c10d._object_to_tensor(
            object, device=current_device)
    else:
        object_tensor, object_size_tensor = c10d._object_to_tensor(object)

    if is_nccl_backend:
        object_size_tensor = object_size_tensor.to(current_device)
        object_tensor = object_tensor.to(current_device)

    c10d.send(object_size_tensor, dst=dst, group=group)
    c10d.send(object_tensor, dst=dst, group=group)


def _recv_object_with_serialization(src: int, group: ProcessGroup, current_device, is_nccl_backend):
    object_size_tensor = torch.empty(1, dtype=torch.long)
    if is_nccl_backend:
        object_size_tensor = object_size_tensor.to(current_device)

    c10d.recv(object_size_tensor, src=src, group=group)

    object_tensor = torch.empty(object_size_tensor.item(), dtype=torch.uint8)
    if is_nccl_backend:
        object_tensor = object_tensor.to(current_device)

    c10d.recv(object_tensor, src=src, group=group)

    object_tensor = object_tensor.type(torch.uint8)
    if object_tensor.device != torch.device("cpu"):
        object_tensor = object_tensor.cpu()

    unpickle_object = _cuda_safe_tensor_to_object(object_tensor, object_size_tensor.item())

    if (
        isinstance(unpickle_object, torch.Tensor)
        and unpickle_object.device.index != torch.cuda.current_device()
    ):
        unpickle_object = unpickle_object.cuda()

    return unpickle_object


def _send_recv_serialization_object(object: Any, send_dst: Optional[int], recv_src: Optional[int], group: ProcessGroup, current_device, is_nccl_backend):
    ops = []
    send_object_tensor = None
    if object is not None and send_dst is not None:
        if Version(torch.__version__) >= Version("1.13.0"):
            send_object_tensor, send_object_size_tensor = c10d._object_to_tensor(object, device=current_device)
        else:
            send_object_tensor, send_object_size_tensor = c10d._object_to_tensor(object)

        if is_nccl_backend:
            send_object_size_tensor = send_object_size_tensor.to(current_device)
            send_object_tensor = send_object_tensor.to(current_device)

        filling_ops_queue(send_object_size_tensor, dist.isend, send_dst, ops, group)

    recv_object_size_tensor = None
    if recv_src is not None:
        recv_object_size_tensor = torch.empty(1, dtype=torch.long)
        if is_nccl_backend:
            recv_object_size_tensor = recv_object_size_tensor.to(current_device)
        filling_ops_queue(recv_object_size_tensor, dist.irecv, recv_src, ops, group)

    if len(ops) > 0:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
    torch.cuda.synchronize()

    ops = []

    if send_dst is not None and send_object_tensor is not None:
        filling_ops_queue(send_object_tensor, dist.isend, send_dst, ops, group)

    recv_object_tensor = None
    if recv_src is not None and recv_object_size_tensor is not None:
        recv_object_tensor = torch.empty(recv_object_size_tensor.item(), dtype=torch.uint8)
        if is_nccl_backend:
            recv_object_tensor = recv_object_tensor.to(current_device)
        filling_ops_queue(recv_object_tensor, dist.irecv, recv_src, ops, group)

    if len(ops) > 0:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
    torch.cuda.synchronize()

    if recv_object_tensor is not None and recv_object_size_tensor is not None:
        recv_object_tensor = recv_object_tensor.type(torch.uint8)
        if recv_object_tensor.device != torch.device("cpu"):
            recv_object_tensor = recv_object_tensor.cpu()

        unpickle_object = _cuda_safe_tensor_to_object(
            recv_object_tensor, recv_object_size_tensor.item())

        if (
            isinstance(unpickle_object, torch.Tensor)
            and unpickle_object.device.index != torch.cuda.current_device()
        ):
            unpickle_object = unpickle_object.cuda()

        return unpickle_object


def _check_if_fast_send_available(object):
    if type(object) is torch.Tensor:
        return True
    elif type(object) is list:
        is_list_of_tensor = all([type(v) is torch.Tensor for v in object])
        return is_list_of_tensor
    elif type(object) is dict:
        is_dict_of_tensor = all([type(k) is str and type(
            v) is torch.Tensor for k, v in object.items()])

        return is_dict_of_tensor
    return False


def _send_recv(
    object,
    send_dst: Optional[int],
    recv_src: Optional[int],
    group: ProcessGroup
) -> Any:
    if c10d._rank_not_in_group(group):
        c10d._warn_not_in_group("_send_recv")
        return

    current_device, is_nccl_backend = check_device(group)

    assert (send_dst is not None) or (recv_src is not None)

    can_fast_send = False
    send_metadata = None
    if send_dst is not None:
        can_fast_send = _check_if_fast_send_available(object) and is_nccl_backend
        if not can_fast_send:
            send_metadata = ['__serialization__', object]
        else:
            send_metadata = []
            send_metadata.append('__tensor__')
            if type(object) is torch.Tensor:
                send_metadata.append(0)
                send_metadata.append((object.shape, object.dtype, object.requires_grad))
            elif type(object) is list:
                send_metadata.append(1)
                for v in object:
                    send_metadata.append((v.shape, v.dtype, v.requires_grad))
            elif type(object) is dict:
                send_metadata.append(2)
                for k, v in object.items():
                    send_metadata.append((k, v.shape, v.dtype, v.requires_grad))

    recv_metadata = _send_recv_serialization_object(send_metadata, send_dst, recv_src, group, current_device, is_nccl_backend)
    if recv_metadata is not None:
        assert type(recv_metadata) is list and len(recv_metadata) >= 2
        if recv_metadata[0] == '__serialization__':
            return recv_metadata[1]
        else:
            recv_metadata.pop(0)
    if not can_fast_send and send_dst is not None:
        return

    send_tensor_list = None
    if type(object) is torch.Tensor:
        send_tensor_list = object
    elif type(object) is list:
        send_tensor_list = object
    elif type(object) is dict:
        send_tensor_list = list(object.values())

    recv_buffer = _batch_send_recv_tensor(send_tensor_list, recv_metadata, send_dst, recv_src, group, current_device)

    if recv_metadata is not None:
        assert recv_buffer is not None
        if recv_metadata[0] == 0:
            return recv_buffer
        elif recv_metadata[0] == 1:
            return recv_buffer
        else:
            return {
                k: v
                for k, v in zip(
                    [m[0] for m in recv_metadata[1:]],
                    recv_buffer,
                )
            }


def _send_object(object: Any, src: int, dst: int, group: ProcessGroup) -> None:
    """send anything to dst rank

    Args:
        object (Any): object needed to be sent
        dst (int): rank of the destination

    Returns:
        None
    """
    _send_recv(object, dst, None, group)


def _recv_object(src: int, dst: int, group: ProcessGroup) -> Any:
    """recv anything from src

    Args:
        src (int): source rank of data. local rank will receive data from src rank.

    Returns:
        Any: Object received from src.
    """
    return _send_recv(None, None, src, group)


def _p2p_comm(
    tensor_send_next: torch.Tensor,
    recv_prev: bool,
    peer: int,
    group: ProcessGroup,
    comm_dtype: torch.dtype = torch.float16,
):
    """
    Send and recv tensor using P2P communication, used when pipeline size is 2 to solve the race communication.

    Agrs:
        tensor_send_next (torch.Tensor): tensor to be sent to next stage
        recv_prev (bool): whether to receive tensor from previous stage
        peer (int): rank of the peer
        group (ProcessGroup): process group
        comm_dtype (torch.dtype): dtype of the tensor to be sent

    Returns:
        torch.Tensor: tensor received from previous stage
    """
    # send and recv shape
    send_next_shape = None
    recv_prev_shape = None

    if tensor_send_next is not None:
        send_next_shape = torch.tensor(tensor_send_next.size(), device=torch.cuda.current_device(), dtype=torch.int64)
    if recv_prev:
        recv_prev_shape = torch.empty((3), device=torch.cuda.current_device(), dtype=torch.int64)

    ops = []
    if send_next_shape is not None:
        send_next_op = dist.P2POp(dist.isend, send_next_shape, peer=peer, group=group)
        ops.append(send_next_op)
    if recv_prev_shape is not None:
        recv_prev_op = dist.P2POp(
            dist.irecv,
            recv_prev_shape,
            peer=peer,
            group=group,
        )
        ops.append(recv_prev_op)

    if len(ops) > 0:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    if recv_prev_shape is not None:
        recv_prev_shape = recv_prev_shape.tolist()

    # send and recv data
    tensor_recv_prev = None
    if recv_prev:
        tensor_recv_prev = torch.empty(recv_prev_shape, device=torch.cuda.current_device(), dtype=comm_dtype)

    ops = []
    if tensor_send_next is not None:
        send_next_op = dist.P2POp(
            dist.isend,
            tensor_send_next,
            peer=peer,
            group=group,
        )
        ops.append(send_next_op)

    if tensor_recv_prev is not None:
        recv_prev_op = dist.P2POp(
            dist.irecv,
            tensor_recv_prev,
            peer=peer,
            group=group,
        )
        ops.append(recv_prev_op)
    if len(ops) > 0:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
    return tensor_recv_prev


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
        output_tensor_grad = _recv_object(
            next_rank, cur_rank, self.stage_manager.get_p2p_process_group(next_rank, cur_rank)
        )

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

    def p2p_communicate(
        self, output_object: Any, recv_pre: bool, peer: int = None, comm_dtype: torch.dtype = torch.float16
    ) -> None:
        """
        Sends the input tensor to the next stage in pipeline, using `P2Pop` in torch.

        Args:
            output_object (Any): Object to be sent.
            next_rank (int, optional): The rank of the recipient of the tensor.
        """
        if peer is None:
            peer = self.stage_manager.get_next_rank()
        cur_rank = self.stage_manager.get_rank()
        recv_tensor = _p2p_comm(
            output_object, recv_pre, peer, self.stage_manager.get_p2p_process_group(cur_rank, peer), comm_dtype
        )
        return recv_tensor
