from typing import Optional

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class TensorBucket:
    def __init__(self, size):
        self._max_size = size
        self._current_size = 0
        self._bucket = []
        self._write_back_pairs = {}

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

    def flatten(self):
        return _flatten_dense_tensors(self._bucket)

    def unflatten(self, flat_tensor):
        return _unflatten_dense_tensors(flat_tensor, self._bucket)

    def unflatten_and_copy(self, flat_tensor):
        unflattened_tensor_list = self.unflatten(flat_tensor)
        for old, new in zip(self._bucket, unflattened_tensor_list):
            old.copy_(new)

    def all_gather(self, group=None):
        flat = self.flatten()
        buffers = [torch.empty_like(flat) for _ in range(dist.get_world_size(group))]
        dist.all_gather(buffers, flat, group=group)
        unflat_buffers = [self.unflatten(buffer) for buffer in buffers]
        # transpose the list of list
        unflat_buffers = list(map(list, zip(*unflat_buffers)))
        for unflat_shards, tensor in zip(unflat_buffers, self._bucket):
            write_back_tensor = self._write_back_pairs[tensor]
            write_back_tensor.data.copy_(
                _flatten_dense_tensors(unflat_shards)[: write_back_tensor.numel()].reshape_as(write_back_tensor)
            )
        self.empty()
