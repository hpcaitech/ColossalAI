# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

# TODO: Remove the toggle-enable_nccl_base_collectives when github open issue #801 is resolved.
if os.getenv("ENABLE_NCCL_BASE_COLLECTIVES", "1") == "0":
    enable_nccl_base_collectives = False
else:
    enable_nccl_base_collectives = True


class Bucket:

    def __init__(self, shard_size: int, dtype: torch.dtype, device: torch.device, group: ProcessGroup):
        self.buffer = torch.zeros((group.size(), shard_size), dtype=dtype, device=device)
        self.group = group
        self.offset = 0
        self.callbacks: List[Callable] = []
        self.output_shard = torch.zeros_like(self.buffer[0])

    def flush(self) -> None:
        """Flush content of the bucket."""
        if self.offset == 0:
            assert len(self.callbacks) == 0
            return
        # reduce-scatter bucket
        if hasattr(dist, "_reduce_scatter_base") and enable_nccl_base_collectives:
            dist._reduce_scatter_base(self.output_shard[:self.offset],
                                      self.buffer[:, :self.offset].contiguous(),
                                      group=self.group)
        else:
            dist.reduce_scatter(self.output_shard[:self.offset],
                                list(self.buffer[:, :self.offset].unbind(0)),
                                group=self.group)
        # execute post-reduction callbacks
        for callback_fn in self.callbacks:
            callback_fn()
        # reuse input bucket but allocate a fresh output shard
        self.buffer[:, :self.offset].zero_()
        self.offset = 0
        self.callbacks.clear()
        self.output_shard = torch.zeros_like(self.buffer[0])

    def alloc(self) -> None:
        """Setup the buffers if they are not allocated.

        Using ``setup`` and ``teardown``, we can ensure that the bucket
        buffers are only allocated during the backward pass, hence saving more
        memory to other parts of the training process, such as the forward pass
        for activation memory.
        """
        for tensor in [self.buffer, self.output_shard]:
            if tensor.storage().size() == 0:
                tensor.storage().resize_(tensor.size().numel())

    def free(self) -> None:
        """Tear down the bucket by freeing the memory"""
        assert self.offset == 0 and self.callbacks == [], "Incorrect call of teardown"
        for tensor in [self.buffer, self.output_shard]:
            tensor.storage().resize_(0)

    def append(self, tensor_list: List[Tensor], callback_fn: Callable):
        # copy data from input_list into bucket
        tensor_size = tensor_list[0].numel()
        stacked_input = torch.stack(tensor_list).view(self.group.size(), tensor_size)
        offset = self.offset
        self.buffer[:, offset:offset + tensor_size].copy_(stacked_input)
        self.offset += tensor_size

        # callback will be given the reduced result
        if callback_fn is not None:
            result_view = self.output_shard[offset:offset + tensor_size].view_as(tensor_list[0])
            self.callbacks.append(functools.partial(callback_fn, result_view))


class ReduceScatterBucketer:
    """
    Helper for bucketing multiple reduce-scatter operations on small tensors
    into larger reduce-scatter ops to improve communication efficiency.

    Usage::

        bucketer = ReduceScatterBucketer()
        bucketer.reduce_scatter_async(
            small_tensors, callback_fn=lambda result: print("small")
        )
        bucketer.reduce_scatter_async(
            big_tensors, callback_fn=lambda result: print("big")
        )
        bucketer.reduce_scatter_async(
            more_small_tensors, callback_fn=lambda result: print("small2")
        )
        bucketer.flush()  # callbacks only guaranteed to be called after flush()
        # Example output (note that it is out of order, due to bucketing):
        # big
        # small
        # small2

    Args:
        bucket_size_mb (int, Optional): bucket size for communicating. Buckets
            are sub-divided based on world_size. Values <= 0 disable bucketing.
    """

    def __init__(self, bucket_size_mb: int = 25):
        self.bucket_size_mb = bucket_size_mb
        self.buckets: Dict[Tuple[torch.dtype, torch.device, ProcessGroup], Bucket] = {}

    @torch.no_grad()
    def reduce_scatter_async(
        self,
        input_list: List[Tensor],
        group: ProcessGroup,
        callback_fn: Optional[Callable] = None,
    ) -> None:
        """
        Reduce-scatter a list of tensors asynchronously, so smaller reductions
        can be bucketed together. The given callback (``callback_fn``) will be
        called with the reduced result at some later time. Call ``flush()`` to
        force all queued ops and callbacks to be executed.

        Note that large inputs will be reduced immediately, and this function
        may also flush the relevant bucket to make room for ``input_list``.

        Args:
            input_list (List[Tensor]): list of tensors to reduce-scatter. List
                should contain ``group.size()`` tensors and each tensor should
                have identical shape, dtype and device.
            group (ProcessGroup): process group for reduction
            callback_fn (Callable, Optional): callback function to call after
                the reduction executes. Function will be called with a single
                argument corresponding to the reduced result.
        """
        world_size = group.size()

        assert (len(input_list) == world_size
               ), f"reduce_scatter received {len(input_list)} inputs, expected group.size() ({world_size})"

        first_input = input_list[0]
        first_input_size = first_input.numel()

        bucket_shard_size = self._get_shard_size(first_input.element_size(), world_size)
        if first_input_size > bucket_shard_size:
            # TODO: investigate how to avoid using torch.cat (because it seems to be slow for CPU tensors)
            # input is too big to fit in the bucket, reduce-scatter directly
            output = torch.zeros_like(input_list[0])
            if hasattr(dist, "_reduce_scatter_base") and enable_nccl_base_collectives:
                input_flattened = torch.cat(input_list)
                dist._reduce_scatter_base(output, input_flattened, group=group)
            else:
                # fallback
                dist.reduce_scatter(output, input_list, group=group)
            if callback_fn is not None:
                callback_fn(output)
            return

        bucket = self._get_bucket(first_input, group)
        if first_input_size > bucket.buffer.size(1) - bucket.offset:
            # not enough space remaining in bucket, flush it now
            bucket.flush()
        bucket.append(input_list, callback_fn)

    @torch.no_grad()
    def flush(self) -> None:
        """Reduce-scatter any partial buckets."""
        for bucket in self.buckets.values():
            bucket.flush()

    @torch.no_grad()
    def free(self) -> None:
        """Free buffers from all buckets."""
        for bucket in self.buckets.values():
            bucket.free()

    @functools.lru_cache()
    def _get_shard_size(self, element_size: int, num_shards: int) -> int:
        if self.bucket_size_mb <= 0:    # Values <= 0 disable bucketing.
            return 0
        MB = 1024 * 1024
        bucket_size = self.bucket_size_mb * MB / element_size
        return int(bucket_size // num_shards)

    def _get_bucket(self, tensor: Tensor, group: ProcessGroup) -> Bucket:
        key = (tensor.dtype, tensor.device, group)
        if key not in self.buckets:
            # buckets are divided into world_size pieces, bucket.data shaped (world_size, shard_size)
            world_size = group.size()
            shard_size = self._get_shard_size(tensor.element_size(), world_size)
            self.buckets[key] = Bucket(shard_size, tensor.dtype, tensor.device, group)
        self.buckets[key].alloc()
        return self.buckets[key]
