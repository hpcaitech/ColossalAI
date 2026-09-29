from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.distributed.device_mesh import DeviceMesh

from colossalai.quantization.fp8 import all_gather_fp8
from colossalai.zero.low_level._utils import all_gather_into_flat_tensor_nd


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield list(lst)[i : i + n]


class TensorBucket:
    def __init__(self, size):
        self._max_size = size
        self._current_size = 0
        self._bucket = []
        self._write_back_pairs = {}
        self._allgather_handle = None
        self._allgather_buffer = None
        world_size = dist.get_world_size()
        self.mesh = DeviceMesh(
            device_type="cuda",
            mesh=list(chunks(range(world_size), torch.cuda.device_count())),
            mesh_dim_names=["internode", "intranode"],
        )
        self.internode_pg = self.mesh["internode"].get_group()
        self.intranode_pg = self.mesh["intranode"].get_group()

    @property
    def max_size(self):
        return self._max_size

    @property
    def current_size(self):
        return self._current_size

    def is_full_or_oversized(self):
        return self._current_size >= self._max_size

    def is_empty(self):
        return len(self._bucket) == 0

    def add_to_bucket(self, tensor, allow_oversize=False, write_back_tensor: Optional[torch.Tensor] = None):
        tensor_size = tensor.numel()

        if not allow_oversize and self.will_exceed_max_size(tensor_size):
            msg = f"The param bucket max size {self._max_size} is exceeded" + f"by tensor (size {tensor_size})"
            raise RuntimeError(msg)

        self._bucket.append(tensor)
        self._current_size += tensor_size
        write_back_tensor = write_back_tensor if write_back_tensor is not None else tensor
        self._write_back_pairs[tensor] = write_back_tensor

    def will_exceed_max_size(self, tensor_size):
        expected_size = self._current_size + tensor_size
        return expected_size > self._max_size

    def get_bucket(self):
        return self._bucket

    def empty(self):
        self._bucket = []
        self._current_size = 0
        self._write_back_pairs = {}
        # del self._allgather_buffer
        self._allgather_buffer = None
        self._allgather_handle = None

    def flatten(self):
        return _flatten_dense_tensors(self._bucket)

    def unflatten(self, flat_tensor):
        return _unflatten_dense_tensors(flat_tensor, self._bucket)

    def unflatten_and_copy(self, flat_tensor):
        unflattened_tensor_list = self.unflatten(flat_tensor)
        for old, new in zip(self._bucket, unflattened_tensor_list):
            old.copy_(new)

    def internode_all_gather_async(self, group=None, fp8_communication: bool = False):
        assert fp8_communication is False, "fp8 communication is not supported yet"
        # assert group is None, "internode_all_gather_async only support default group"
        flat = self.flatten()

        if isinstance(group, tuple):
            world_size = np.prod([dist.get_world_size(pg) for pg in group])
        else:
            world_size = dist.get_world_size(group)

        n_gpus = torch.cuda.device_count()
        # print("==debug== flat", flat.dtype)
        self._allgather_buffer = torch.empty(flat.numel() * world_size // n_gpus, device=flat.device, dtype=flat.dtype)
        self._allgather_handle = all_gather_into_flat_tensor_nd(
            self._allgather_buffer, flat, group=self.internode_pg, async_op=True
        )

    def intranode_allgather_and_write_back(self, group=None):
        assert self._allgather_buffer is not None, "all_gather_async must be called before write_back_and_empty"
        assert self._allgather_handle is not None, "all_gather_async must be called before write_back_and_empty"
        self._allgather_handle.wait()
        dist.get_world_size(group)
        n_gpus = torch.cuda.device_count()

        local_chunk = self._allgather_buffer
        # print("==debug== local_chunk", local_chunk.dtype)
        self._allgather_buffer = torch.empty(
            local_chunk.numel() * n_gpus, device=local_chunk.device, dtype=local_chunk.dtype
        )
        all_gather_into_flat_tensor_nd(self._allgather_buffer, local_chunk, group=self.intranode_pg, async_op=False)
        del local_chunk
        self.write_back_and_empty()

    def all_gather_async(self, group=None, fp8_communication: bool = False):
        assert fp8_communication is False, "fp8 communication is not supported yet"

        flat = self.flatten()
        if isinstance(group, tuple):
            world_size = np.prod([dist.get_world_size(pg) for pg in group])
        else:
            world_size = dist.get_world_size(group)
        self._allgather_buffer = torch.empty(flat.numel() * world_size, device=flat.device, dtype=flat.dtype)
        self._allgather_handle = all_gather_into_flat_tensor_nd(
            self._allgather_buffer, flat, group=group, async_op=True
        )

    def write_back_and_empty(self, group=None):
        assert self._allgather_buffer is not None, "all_gather_async must be called before write_back_and_empty"
        assert self._allgather_handle is not None, "all_gather_async must be called before write_back_and_empty"

        if isinstance(group, tuple):
            world_size = np.prod([dist.get_world_size(pg) for pg in group])
        else:
            world_size = dist.get_world_size(group)
        self._allgather_handle.wait()
        unflat_buffers = [self.unflatten(buffer) for buffer in self._allgather_buffer.chunk(world_size)]
        # transpose the list of list
        unflat_buffers = list(map(list, zip(*unflat_buffers)))
        for unflat_shards, tensor in zip(unflat_buffers, self._bucket):
            write_back_tensor = self._write_back_pairs[tensor]
            rec_tensor = _flatten_dense_tensors(unflat_shards)[: write_back_tensor.numel()]
            if write_back_tensor.is_contiguous():
                rec_tensor = rec_tensor.view_as(write_back_tensor)
            else:
                rec_tensor = rec_tensor.reshape_as(write_back_tensor)
            write_back_tensor.data.copy_(rec_tensor)
        self.empty()

    def all_gather(self, group=None, fp8_communication: bool = False):
        flat = self.flatten()
        if isinstance(group, tuple):
            world_size = np.prod([dist.get_world_size(pg) for pg in group])
        else:
            world_size = dist.get_world_size(group)
        buffer = torch.empty(flat.numel() * world_size, device=flat.device, dtype=flat.dtype)
        if fp8_communication:
            # TODO: fit fp8
            all_gather_fp8(list(buffer.chunk(dist.get_world_size(group))), flat, group=group, fp8_format="e4m3")
        else:
            # dist.all_gather_into_tensor(buffer, flat, group=group)
            all_gather_into_flat_tensor_nd(buffer, flat, group=group)
        unflat_buffers = [self.unflatten(buffer) for buffer in buffer.chunk(world_size)]
        # transpose the list of list
        unflat_buffers = list(map(list, zip(*unflat_buffers)))
        for unflat_shards, tensor in zip(unflat_buffers, self._bucket):
            write_back_tensor = self._write_back_pairs[tensor]
            rec_tensor = _flatten_dense_tensors(unflat_shards)[: write_back_tensor.numel()]
            if write_back_tensor.is_contiguous():
                rec_tensor = rec_tensor.view_as(write_back_tensor)
            else:
                rec_tensor = rec_tensor.reshape_as(write_back_tensor)
            write_back_tensor.data.copy_(rec_tensor)

        self.empty()
