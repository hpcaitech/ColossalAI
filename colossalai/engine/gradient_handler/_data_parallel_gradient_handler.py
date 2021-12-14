#!/usr/bin/env python

import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from colossalai.core import global_context as gpc
from colossalai.registry import GRADIENT_HANDLER
from ._base_gradient_handler import BaseGradientHandler
from ...context.parallel_mode import ParallelMode


@GRADIENT_HANDLER.register_module
class DataParallelGradientHandler(BaseGradientHandler):
    """A helper class to handle all-reduce operations in a data parallel group.
    A all-reduce collective communication will be operated in 
    :func:`handle_gradient` among a data parallel group.
    For better performance, it bucketizes the gradients of all parameters that are 
    the same type to improve the efficiency of communication.
    """

    def handle_gradient(self):
        """A method running a all-reduce operation in a data parallel group.
        """
        # TODO: add memory buffer
        if gpc.data_parallel_size > 1:
            # bucketize and all-reduce
            buckets = {}
            # Pack the buckets.
            for param in self._model.parameters():
                if param.requires_grad and param.grad is not None:
                    tp = param.data.type()
                    if tp not in buckets:
                        buckets[tp] = []
                    buckets[tp].append(param)
                    # param.main_grad = param.grad

            # For each bucket, all-reduce and copy all-reduced grads.
            for tp in buckets:
                bucket = buckets[tp]
                grads = [param.grad.data for param in bucket]
                coalesced = _flatten_dense_tensors(grads)
                coalesced /= gpc.get_world_size(ParallelMode.DATA)

                dist.all_reduce(
                    coalesced, group=gpc.get_group(ParallelMode.DATA))
                for buf, synced in zip(grads, _unflatten_dense_tensors(
                        coalesced, grads)):
                    buf.copy_(synced)
