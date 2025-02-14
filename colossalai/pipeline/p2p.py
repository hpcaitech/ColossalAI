#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import io
import pickle
import re
from collections import namedtuple
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from packaging.version import Version
from torch.distributed import ProcessGroup
from torch.distributed import distributed_c10d as c10d
from torch.utils._pytree import tree_flatten, tree_unflatten

from colossalai.accelerator import get_accelerator

from .stage_manager import PipelineStageManager


def _cuda_safe_tensor_to_object(tensor: torch.Tensor, tensor_size: torch.Size) -> Any:
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
        device_index = get_accelerator().current_device()
        # There might be more than one output tensors during forward
        for cuda_str in re.finditer(b"cuda", buf_array):
            pos = cuda_str.start()
            buf_array[pos + 5] = 48 + device_index
        buf = bytes(buf_array)

    io_bytes = io.BytesIO(buf)
    byte_pickler = pickle.Unpickler(io_bytes)
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

    return c10d.is_nccl_available() and pg.name() == c10d.Backend.NCCL


# NOTE: FIXME: NPU DOES NOT support isend nor irecv, so broadcast is kept for future use
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

    is_nccl_backend = _check_for_nccl_backend(group)
    current_device = None

    if device is not None:
        if is_nccl_backend and device.type != "cuda":
            raise ValueError("device type must be cuda for nccl backend")
        current_device = device
    else:
        current_device = torch.device("cpu")
        if is_nccl_backend:
            current_device = torch.device("cuda", get_accelerator().current_device())

    my_rank = dist.get_rank()
    # Serialize object_list elements to tensors on src rank.
    if my_rank == src:
        if Version(torch.__version__) >= Version("2.3.0"):
            tensor_list, size_list = zip(
                *[c10d._object_to_tensor(obj, device=current_device, group=group) for obj in object_list]
            )
        elif Version(torch.__version__) >= Version("1.13.0"):
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
            obj_view = object_tensor[offset : offset + obj_size]
            obj_view = obj_view.type(torch.uint8)
            if obj_view.device != torch.device("cpu"):
                obj_view = obj_view.cpu()
            offset += obj_size
            # unpickle
            unpickle_object = _cuda_safe_tensor_to_object(obj_view, obj_size)

            # unconsistence in device
            if (
                isinstance(unpickle_object, torch.Tensor)
                and unpickle_object.device.index != get_accelerator().current_device()
            ):
                unpickle_object = unpickle_object.to(get_accelerator().current_device())

            object_list[i] = unpickle_object


def _check_for_nccl_hccl_backend(group):
    pg = group or c10d._get_default_group()
    # Gate PG wrapper check on Gloo availability.
    if c10d._GLOO_AVAILABLE:
        # It is not expected for PG to be wrapped many times, but support it just in case
        while isinstance(pg, c10d._ProcessGroupWrapper):
            pg = pg.wrapped_pg

    return (c10d.is_nccl_available() or torch.distributed.is_hccl_available()) and (
        pg.name() == c10d.Backend.NCCL or pg.name() == c10d.Backend.HCCL
    )


def _check_device(group):
    is_nccl_backend = _check_for_nccl_hccl_backend(group)
    current_device = torch.device("cpu")
    if is_nccl_backend:
        current_device = torch.device(get_accelerator().current_device())
    return current_device, is_nccl_backend


TensorMetadata = namedtuple("TensorMetadata", ["shape", "dtype", "requires_grad"])
P2PMetadata = namedtuple("P2PMetadata", ["tree_spec", "tensor_metadata", "non_tensor_obj_idx", "non_tensor_objs"])


def create_send_metadata(
    object: Any, strict: bool = True, return_tensor: bool = False
) -> Union[P2PMetadata, Tuple[P2PMetadata, List[torch.Tensor]]]:
    """
    Args:
        object (Any): object needed to be sent
        strict (bool, optional): whether to check if the object is supported for fast send
        return_tensor (bool, optional): whether to return tensor objects
    """
    objs, tree_spec = tree_flatten(object)
    tensor_metadata, tensor_objs = [], []
    non_tensor_obj_idx, non_tensor_objs = [], []
    for idx, obj in enumerate(objs):
        if isinstance(obj, torch.Tensor):
            tensor_objs.append(obj)
            tensor_metadata.append(TensorMetadata(obj.shape, obj.dtype, obj.requires_grad))
        else:
            non_tensor_obj_idx.append(idx)
            non_tensor_objs.append(obj)

    assert not strict or len(non_tensor_objs) == 0, "Only support tensor for fast send"
    metadata = P2PMetadata(tree_spec, tensor_metadata, non_tensor_obj_idx, non_tensor_objs)
    return metadata if not return_tensor else (metadata, tensor_objs)


def _filling_ops_queue(
    obj: Union[torch.Tensor, List[torch.Tensor]],
    comm_op: Callable,
    comm_rank: int,
    ops_queue: List,
    group: ProcessGroup,
):
    if isinstance(obj, torch.Tensor):
        obj = obj.contiguous()
        op_to_add = dist.P2POp(comm_op, obj, comm_rank, group)
        ops_queue.append(op_to_add)
    else:
        for tensor_to_comm in obj:
            assert isinstance(tensor_to_comm, torch.Tensor)
            _filling_ops_queue(tensor_to_comm, comm_op, comm_rank, ops_queue, group)


def _create_recv_buffer(tensor_metadata: List[TensorMetadata], current_device) -> List[torch.Tensor]:
    buffer_recv = []
    for metadata in tensor_metadata:
        tensor_recv = torch.empty(
            metadata.shape, requires_grad=metadata.requires_grad, device=current_device, dtype=metadata.dtype
        )
        buffer_recv.append(tensor_recv)
    return buffer_recv


def _batch_send_recv_tensor(
    send_tensor_list: Optional[List[torch.Tensor]],
    recv_tensor_metadata: Optional[List[TensorMetadata]],
    send_dst: Optional[int],
    recv_src: Optional[int],
    send_group: Optional[ProcessGroup],
    recv_group: Optional[ProcessGroup],
    current_device: Any,
    overlap_p2p: bool = True,
    send_first: bool = True,
) -> Optional[Union[torch.Tensor, List[torch.Tensor]]]:
    buffer_recv = None
    if recv_tensor_metadata is not None:
        buffer_recv = _create_recv_buffer(recv_tensor_metadata, current_device)

    ops = []
    is_send = send_dst is not None and send_tensor_list is not None
    is_recv = recv_src is not None and buffer_recv is not None

    if send_first:
        if is_send:
            assert send_group is not None
            _filling_ops_queue(send_tensor_list, dist.isend, send_dst, ops, send_group)
        if is_recv:
            assert recv_group is not None
            _filling_ops_queue(buffer_recv, dist.irecv, recv_src, ops, recv_group)
    else:
        if is_recv:
            assert recv_group is not None
            _filling_ops_queue(buffer_recv, dist.irecv, recv_src, ops, recv_group)
        if is_send:
            assert send_group is not None
            _filling_ops_queue(send_tensor_list, dist.isend, send_dst, ops, send_group)

    if len(ops) > 0:
        reqs = dist.batch_isend_irecv(ops)
        if not overlap_p2p:
            for req in reqs:
                req.wait()
            return buffer_recv, []
        else:
            return buffer_recv, reqs
    return None, []


def _send_recv_serialization_object(
    object: Optional[P2PMetadata],
    send_dst: Optional[int],
    recv_src: Optional[int],
    send_group: Optional[ProcessGroup],
    recv_group: Optional[ProcessGroup],
    current_device: Any,
    is_nccl_backend: bool,
    send_first: bool = True,
) -> Optional[P2PMetadata]:
    ops = []
    send_object_tensor = None
    send_object_size_tensor = None
    if object is not None and send_dst is not None:
        if Version(torch.__version__) >= Version("2.3.0"):
            send_object_tensor, send_object_size_tensor = c10d._object_to_tensor(
                object, device=current_device, group=send_group
            )
        elif Version(torch.__version__) >= Version("1.13.0"):
            send_object_tensor, send_object_size_tensor = c10d._object_to_tensor(object, device=current_device)
        else:
            send_object_tensor, send_object_size_tensor = c10d._object_to_tensor(object)

        if is_nccl_backend:
            send_object_size_tensor = send_object_size_tensor.to(current_device)
            send_object_tensor = send_object_tensor.to(current_device)

    recv_object_size_tensor = None
    if recv_src is not None:
        recv_object_size_tensor = torch.empty(1, dtype=torch.long)
        if is_nccl_backend:
            recv_object_size_tensor = recv_object_size_tensor.to(current_device)

    if send_first:
        if send_object_size_tensor is not None:
            _filling_ops_queue(send_object_size_tensor, dist.isend, send_dst, ops, send_group)
        if recv_src is not None:
            _filling_ops_queue(recv_object_size_tensor, dist.irecv, recv_src, ops, recv_group)
    else:
        if recv_src is not None:
            _filling_ops_queue(recv_object_size_tensor, dist.irecv, recv_src, ops, recv_group)
        if send_object_size_tensor is not None:
            _filling_ops_queue(send_object_size_tensor, dist.isend, send_dst, ops, send_group)

    if len(ops) > 0:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()  # This blocks the compute stream in torch

    ops = []
    is_send = send_dst is not None and send_object_tensor is not None
    is_recv = recv_src is not None and recv_object_size_tensor is not None

    recv_object_tensor = None
    if is_recv:
        recv_object_tensor = torch.empty(recv_object_size_tensor.item(), dtype=torch.uint8)
        if is_nccl_backend:
            recv_object_tensor = recv_object_tensor.to(current_device)

    if send_first:
        if is_send:
            _filling_ops_queue(send_object_tensor, dist.isend, send_dst, ops, send_group)
        if is_recv:
            _filling_ops_queue(recv_object_tensor, dist.irecv, recv_src, ops, recv_group)
    else:
        if is_recv:
            _filling_ops_queue(recv_object_tensor, dist.irecv, recv_src, ops, recv_group)
        if is_send:
            _filling_ops_queue(send_object_tensor, dist.isend, send_dst, ops, send_group)

    if len(ops) > 0:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    if recv_object_tensor is not None and recv_object_size_tensor is not None:
        recv_object_tensor = recv_object_tensor.type(torch.uint8)
        if recv_object_tensor.device != torch.device("cpu"):
            recv_object_tensor = recv_object_tensor.cpu()

        unpickle_object = _cuda_safe_tensor_to_object(recv_object_tensor, recv_object_size_tensor.item())

        if (
            isinstance(unpickle_object, torch.Tensor)
            and unpickle_object.device.index != get_accelerator().current_device()
        ):
            unpickle_object = unpickle_object.to(get_accelerator().current_device())

        return unpickle_object


def _communicate(
    object: Any,
    send_dst: Optional[int],
    recv_src: Optional[int],
    overlap_p2p: bool,
    send_group: Optional[ProcessGroup] = None,
    recv_group: Optional[ProcessGroup] = None,
    send_metadata: bool = True,
    metadata_recv: Optional[P2PMetadata] = None,
    send_first: Optional[bool] = None,
) -> Any:
    """
    Send and receive object from send_dst and recv_src respectively

    Args:
        object (Any): object needed to be sent
        send_dst (int): rank of the destination
        recv_src (int): rank of the source
        overlap_p2p (bool): whether to overlap p2p communication with computation
        send_group (ProcessGroup, optional): process group of sender
        recv_group (ProcessGroup, optional): process group of receiver
        send_metadata (bool, optional): whether to send metadata
        metadata_recv (P2PMetadata, optional): metadata of the object to be received
    """
    assert send_dst is not None or recv_src is not None, "send_dst and recv_src cannot be both None"
    assert send_dst is None or send_group is not None, "send_group must be specified when send_dst is not None"
    assert recv_src is None or recv_group is not None, "recv_group must be specified when recv_src is not None"
    assert (
        metadata_recv is None or len(metadata_recv.non_tensor_obj_idx) == 0
    ), "metadata_recv should not contain non-tensor objects"

    metadata_send, tensor_objs = None, None
    if object is not None:
        # NOTE: if object contains non-tensor objects, we have to send metadata
        metadata_send, tensor_objs = create_send_metadata(object, strict=False, return_tensor=True)
        send_metadata = send_metadata or len(metadata_send.non_tensor_obj_idx) > 0
    else:
        send_metadata = False

    assert not c10d._rank_not_in_group(send_group) and not c10d._rank_not_in_group(recv_group)
    current_send_device, is_send_nccl_backend = _check_device(send_group)
    current_recv_device, is_recv_nccl_backend = _check_device(recv_group)

    is_nccl_backend = is_send_nccl_backend and is_recv_nccl_backend

    assert current_send_device == current_recv_device
    current_device = current_send_device

    if (send_dst is not None and send_metadata) or (recv_src is not None and metadata_recv is None):
        # Send and receive metadata
        _metadata_recv = _send_recv_serialization_object(
            object=metadata_send,
            send_dst=send_dst if send_metadata else None,
            recv_src=recv_src if metadata_recv is None else None,
            send_group=send_group if send_metadata else None,
            recv_group=recv_group if metadata_recv is None else None,
            current_device=current_device,
            is_nccl_backend=is_nccl_backend,
            send_first=send_first if send_first != None else True,
        )
        assert (
            metadata_recv is None or _metadata_recv is None
        ), "You shouldn't receive metadata when using the cached metadata"
        metadata_recv = _metadata_recv if metadata_recv is None else metadata_recv

    # Send and receive data
    recv_tensor_metadata = None if metadata_recv is None else metadata_recv.tensor_metadata
    recv_tensor_objs, wait_handles = _batch_send_recv_tensor(
        tensor_objs,
        recv_tensor_metadata,
        send_dst,
        recv_src,
        send_group,
        recv_group,
        current_device,
        overlap_p2p=overlap_p2p,
        send_first=send_first if send_first != None else True,
    )
    if metadata_recv is not None:
        assert isinstance(metadata_recv, P2PMetadata)
        tree_spec = metadata_recv.tree_spec
        non_tensor_obj_idx = metadata_recv.non_tensor_obj_idx
        non_tensor_objs = metadata_recv.non_tensor_objs

        if recv_tensor_objs is None:
            recv_tensor_objs = []

        for idx in non_tensor_obj_idx:
            recv_tensor_objs.insert(idx, non_tensor_objs.pop(0))
        recv_object = tree_unflatten(recv_tensor_objs, tree_spec)
        return recv_object, wait_handles

    return None, wait_handles


def _p2p_comm(
    tensor_send_next: torch.Tensor,
    recv_prev: bool,
    peer: int,
    group: ProcessGroup,
    comm_dtype: torch.dtype = torch.float16,
):
    """
    Send and recv tensor using P2P communication, used when pipeline size is 2 to solve the race communication.

    Args:
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
        send_next_shape = torch.tensor(
            tensor_send_next.size(), device=get_accelerator().current_device(), dtype=torch.int64
        )
    if recv_prev:
        recv_prev_shape = torch.empty((3), device=get_accelerator().current_device(), dtype=torch.int64)

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
        tensor_recv_prev = torch.empty(recv_prev_shape, device=get_accelerator().current_device(), dtype=comm_dtype)

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
    def __init__(self, stage_manager: PipelineStageManager, overlap_p2p: bool = True) -> None:
        self.stage_manager = stage_manager
        self.overlap_p2p = overlap_p2p

    def recv_forward(
        self, prev_rank: Optional[int] = None, metadata_recv: Optional[P2PMetadata] = None
    ) -> Tuple[Any, List]:
        """Copy the forward output from the previous stage in pipeline as the input tensor of this stage.

        Args:
            prev_rank (int, optional): The rank of the source of the tensor.

        Returns:
            Any: The input tensor or input tensor list.
            List: List of handles for the communication requests, if overlap is enabled.
        """
        if prev_rank is None:
            prev_rank = self.stage_manager.get_prev_rank()
        input_tensor, wait_handles = _communicate(
            object=None,
            recv_src=prev_rank,
            send_dst=None,
            recv_group=self.stage_manager.get_p2p_process_group(),
            metadata_recv=metadata_recv,
            overlap_p2p=self.overlap_p2p,
        )

        return input_tensor, wait_handles

    def recv_backward(
        self, next_rank: Optional[int] = None, metadata_recv: Optional[P2PMetadata] = None
    ) -> Tuple[Any, List]:
        """Copy the gradient tensor from the next stage in pipeline as the input gradient of this stage.
        Args:
            next_rank (int, optional): The rank of the source of the tensor.

        Returns:
            Any: The input tensor or input tensor list.
            List: List of handles for the communication requests, if overlap is enabled.
        """
        if next_rank is None:
            next_rank = self.stage_manager.get_next_rank()

        output_tensor_grad, wait_handles = _communicate(
            object=None,
            recv_src=next_rank,
            send_dst=None,
            recv_group=self.stage_manager.get_p2p_process_group(),
            metadata_recv=metadata_recv,
            overlap_p2p=self.overlap_p2p,
        )

        return output_tensor_grad, wait_handles

    def send_forward(self, output_object: Any, next_rank: Optional[int] = None, send_metadata: bool = True) -> List:
        """Sends the input tensor to the next stage in pipeline.

        Args:
            output_object (Any): Object to be sent.
            next_rank (int, optional): The rank of the recipient of the tensor.

        Returns:
            List: List of handles for the communication requests, if overlap is enabled.
        """
        if next_rank is None:
            next_rank = self.stage_manager.get_next_rank()
        _, handles = _communicate(
            output_object,
            recv_src=None,
            send_dst=next_rank,
            send_group=self.stage_manager.get_p2p_process_group(),
            send_metadata=send_metadata,
            overlap_p2p=self.overlap_p2p,
        )
        return handles

    def send_backward(self, input_object: Any, prev_rank: Optional[int] = None, send_metadata: bool = True) -> List:
        """Sends the gradient tensor to the previous stage in pipeline.

        Args:
            input_object (Any): Object to be sent.
            prev_rank (int, optional): The rank of the recipient of the tensor

        Returns:
            List: List of handles for the communication requests, if overlap is enabled.
        """
        if prev_rank is None:
            prev_rank = self.stage_manager.get_prev_rank()
        _, handles = _communicate(
            input_object,
            recv_src=None,
            send_dst=prev_rank,
            send_group=self.stage_manager.get_p2p_process_group(),
            send_metadata=send_metadata,
            overlap_p2p=self.overlap_p2p,
        )
        return handles

    def send_forward_recv_forward(
        self,
        output_object: Any,
        is_send: bool,
        is_recv: bool,
        send_first: bool,
        send_metadata: bool = True,
        metadata_recv: Optional[P2PMetadata] = None,
    ) -> Tuple[Any, List]:
        """Sends the input tensor to the next pipeline stage and copy the output tensor from the next pipeline stage

        Args:
            output_object (Any): Object to be sent.
            is_send (bool): Whether to send the input tensor to the next pipeline stage.
            is_recv (bool): Whether to copy the output tensor from the next pipeline stage.
            send_first (bool): Whether to send before receive.
            send_metadata (bool, optional): Whether to send metadata.
            metadata_recv (P2PMetadata, optional): The cached metadata(size, type) of the object to be received.

        Returns:
            Any: The input tensor or input tensor list.
            List: List of handles for the communication requests, if overlap is enabled.
        """
        next_rank = self.stage_manager.get_next_rank() if is_send else None
        prev_rank = self.stage_manager.get_prev_rank() if is_recv else None
        group = self.stage_manager.get_p2p_process_group()
        return _communicate(
            output_object,
            send_dst=next_rank,
            recv_src=prev_rank,
            send_group=group if is_send else None,
            recv_group=group if is_recv else None,
            send_metadata=send_metadata if is_send else False,
            metadata_recv=metadata_recv if is_recv else None,
            send_first=send_first,
            overlap_p2p=self.overlap_p2p,
        )

    def send_backward_recv_backward(
        self,
        input_object: Any,
        is_send: bool,
        is_recv: bool,
        send_first: bool,
        send_metadata: bool = True,
        metadata_recv: Optional[P2PMetadata] = None,
    ) -> Tuple[Any, List]:
        """Sends the gradient tensor to the previous pipeline stage and copy the gradient tensor from the previous pipeline stage

        Args:
            input_object (Any): Object to be sent.
            is_send (bool): Whether to send the gradient tensor to the previous pipeline stage.
            is_recv (bool): Whether to copy the gradient tensor from the previous pipeline stage.
            send_first (bool): Whether to send before receive.
            send_metadata (bool, optional): Whether to send metadata.
            metadata_recv (P2PMetadata, optional): The cached metadata(size, type) of the object to be received.

        Returns:
            Any: The input tensor or input tensor list.
            List: List of handles for the communication requests, if overlap is enabled.
        """
        prev_rank = self.stage_manager.get_prev_rank() if is_send else None
        next_rank = self.stage_manager.get_next_rank() if is_recv else None

        group = self.stage_manager.get_p2p_process_group()

        return _communicate(
            input_object,
            send_dst=prev_rank,
            recv_src=next_rank,
            send_group=group if is_send else None,
            recv_group=group if is_recv else None,
            send_metadata=send_metadata if is_send else False,
            metadata_recv=metadata_recv if is_recv else None,
            send_first=send_first,
            overlap_p2p=self.overlap_p2p,
        )

    def send_forward_recv_backward(
        self,
        input_object: Any,
        send_metadata: bool = True,
        metadata_recv: Optional[P2PMetadata] = None,
        send_first: Optional[bool] = None,
    ) -> Tuple[Any, List]:
        """Sends the gradient tensor to and copy the gradient tensor from the next pipeline stage

        Args:
            input_object (Any): Object to be sent.

        Returns:
            Any: The input tensor or input tensor list.
            List: List of handles for the communication requests, if overlap is enabled.
        """
        next_rank = self.stage_manager.get_next_rank()
        group = self.stage_manager.get_p2p_process_group()
        return _communicate(
            input_object,
            next_rank,
            next_rank,
            send_group=group,
            recv_group=group,
            send_metadata=send_metadata,
            metadata_recv=metadata_recv,
            send_first=send_first,
            overlap_p2p=False,
        )

    def send_backward_recv_forward(
        self,
        input_object: Any,
        send_metadata: bool = True,
        metadata_recv: Optional[P2PMetadata] = None,
        send_first: Optional[bool] = None,
    ) -> Tuple[Any, List]:
        """Sends the gradient tensor to and copy the gradient tensor from the previous stage in pipeline

        Args:
            input_object (Any): Object to be sent.

        Returns:
            Any: The input tensor or input tensor list.
            List: List of handles for the communication requests, if overlap is enabled.
        """
        prev_rank = self.stage_manager.get_prev_rank()
        group = self.stage_manager.get_p2p_process_group()
        return _communicate(
            input_object,
            prev_rank,
            prev_rank,
            send_group=group,
            recv_group=group,
            send_metadata=send_metadata,
            metadata_recv=metadata_recv,
            send_first=send_first,
            overlap_p2p=False,
        )

    def p2p_communicate(
        self,
        output_object: Any,
        recv_pre: bool,
        next_rank: Optional[int] = None,
        comm_dtype: torch.dtype = torch.float16,
    ) -> Any:
        """
        Sends the input tensor to the next stage in pipeline, using `P2Pop` in torch.

        Args:
            output_object (Any): Object to be sent.
            next_rank (int, optional): The rank of the recipient of the tensor.
        """
        if next_rank is None:
            next_rank = self.stage_manager.get_next_rank()
        recv_tensor = _p2p_comm(
            output_object,
            recv_pre,
            next_rank,
            self.stage_manager.get_p2p_process_group(),
            comm_dtype,
        )
        return recv_tensor
