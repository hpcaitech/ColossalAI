# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup


class Bucket:
    def __init__(self, size: int, dtype: torch.dtype, device: torch.device, group: ProcessGroup):
        self.buffer = torch.zeros(size, dtype=dtype, device=device)
        self.group = group
        self.offset = 0
        self.callbacks: List[Callable] = []

    def flush(self) -> None:
        """Flush content of the bucket."""
        if self.offset == 0:
            assert len(self.callbacks) == 0
            return
        # reduce-scatter bucket
        dist.all_reduce(self.buffer[: self.offset], group=self.group)

        # execute post-reduction callbacks
        for callback_fn in self.callbacks:
            callback_fn()
        # reuse input bucket but allocate a fresh output shard
        self.offset = 0
        self.callbacks.clear()
        self.buffer = torch.zeros_like(self.buffer)

    def alloc(self) -> None:
        if self.buffer.storage().size() == 0:
            self.buffer.storage().resize_(self.buffer.numel())

    def free(self) -> None:
        assert self.offset == 0 and self.callbacks == [], "Incorrect call of teardown"
        self.buffer.storage().resize_(0)

    def append(self, tensor: Tensor, callback_fn: Callable):
        tensor_size = tensor.numel()
        offset = self.offset
        self.buffer[offset : offset + tensor_size].copy_(tensor.flatten())
        self.offset += tensor_size

        # callback will be given the reduced result
        if callback_fn is not None:
            result_view = self.buffer[offset : offset + tensor_size].view(tensor.shape)
            self.callbacks.append(functools.partial(callback_fn, result_view))

    @property
    def avail_size(self) -> int:
        return self.buffer.size(0) - self.offset


class Reducer:
    def __init__(self, bucket_size_mb: int = 25):
        self.bucket_size_mb = bucket_size_mb
        self.buckets: Dict[Tuple[torch.dtype, torch.device, ProcessGroup], Bucket] = {}

    @torch.no_grad()
    def all_reduce_async(
        self,
        tensor: Tensor,
        group: ProcessGroup,
        callback_fn: Optional[Callable] = None,
    ) -> None:
        bucket_size = self._get_bucket_size(tensor.element_size())

        if tensor.numel() >= bucket_size:
            dist.all_reduce(tensor, group=group)
            if callback_fn is not None:
                callback_fn(tensor)
            return

        bucket = self._get_bucket(tensor, group)
        if tensor.numel() > bucket.avail_size:
            # not enough space remaining in bucket, flush it now
            bucket.flush()
        bucket.append(tensor, callback_fn)

    @torch.no_grad()
    def flush(self) -> None:
        for bucket in self.buckets.values():
            bucket.flush()

    @torch.no_grad()
    def free(self) -> None:
        for bucket in self.buckets.values():
            bucket.free()

    @functools.lru_cache()
    def _get_bucket_size(self, element_size: int) -> int:
        if self.bucket_size_mb <= 0:  # Values <= 0 disable bucketing.
            return 0
        MB = 1024 * 1024
        bucket_size = self.bucket_size_mb * MB / element_size
        return int(bucket_size)

    def _get_bucket(self, tensor: Tensor, group: ProcessGroup) -> Bucket:
        key = (tensor.dtype, tensor.device, group)
        if key not in self.buckets:
            bucket_size = self._get_bucket_size(tensor.element_size())
            self.buckets[key] = Bucket(bucket_size, tensor.dtype, tensor.device, group)
        self.buckets[key].alloc()
        return self.buckets[key]
