import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from typing import Iterable


def bucket_allreduce(param_list: Iterable[nn.Parameter], group=None):
    # get communication world size
    comm_size = dist.get_world_size(group)
    # bucketize and all-reduce
    buckets = {}
    # Pack the buckets.
    for param in param_list:
        if param.requires_grad and param.grad is not None:
            tp = param.data.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(param)

    # For each bucket, all-reduce and copy all-reduced grads.
    for tp in buckets:
        bucket = buckets[tp]
        grads = [param.grad.data for param in bucket]
        coalesced = _flatten_dense_tensors(grads)
        coalesced /= comm_size

        dist.all_reduce(coalesced, group=group)
        for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
            buf.copy_(synced)
