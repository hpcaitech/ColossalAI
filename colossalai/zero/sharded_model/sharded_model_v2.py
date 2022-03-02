
import functools
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.engine.ophooks import (ShardGradHook, ShardParamHook,
                                       register_ophooks_recursively)
from colossalai.engine.paramhooks import BaseParamHookMgr
from colossalai.logging import get_dist_logger
from colossalai.zero.shard_param import ShardParam
from colossalai.zero.sharded_model.reduce_scatter import ReduceScatterBucketer
from colossalai.zero.sharded_model.sharded_grad import ShardedGradient
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter

from ._zero3_utils import chunk_and_pad, get_gradient_predivide_factor


class ShardedModelV2(nn.Module):
    def __init__(self,
                 module: nn.Module,
                 process_group: Optional[ProcessGroup] = None,
                 reduce_scatter_process_group: Optional[ProcessGroup] = None,
                 reduce_scatter_bucket_size_mb: int = 25,
                 reshard_after_forward: bool = True,
                 mixed_precision: bool = False,
                 fp32_reduce_scatter: bool = False,
                 offload_config: Optional[dict] = None,
                 gradient_predivide_factor: Optional[float] = 1.0,
                 ):
        r"""
        A demo to reconfigure zero1 shared_model.
        Currently do not consider the Optimizer States.
        """
        super().__init__()
        self.logger = get_dist_logger()

        self.process_group = process_group or gpc.get_group(ParallelMode.DATA)
        self.reduce_scatter_process_group = reduce_scatter_process_group or self.process_group
        self.world_size = dist.get_world_size(self.process_group)
        self.rank = dist.get_rank(self.process_group)

        # The module has to be placed on GPU
        self.module = module.cuda()

        # Shard the parameters at first
        for _, param in self.module.named_parameters():
            param.ca_attr = ShardParam(param)
            param.ca_attr.shard()
            param._sharded_grad = ShardedGradient(param, self, offload_config)

        # Register hooks
        register_ophooks_recursively(self.module, [ShardParamHook(), ShardGradHook()])
        self.param_hook_mgr = BaseParamHookMgr(list(self.module.parameters()))
        self.param_hook_mgr.register_backward_hooks(self._grad_post_backward_hook)

        self.reshard_after_forward = reshard_after_forward
        self.mixed_precision = mixed_precision
        self.fp32_reduce_scatter = fp32_reduce_scatter
        self._cpu_offload: bool = offload_config.get('device', None) == 'cpu' if offload_config else False
        # We find if gradient_predivide_factor != 1.0, there may be wrong precision problem
        # So we use 1.0 as the default gradient_predivide_factor
        # However, if you set gradient_predivide_factor to None, we will set gradient_predivide_factor to a value >= 1.0 automatically
        self.gradient_predivide_factor: float = gradient_predivide_factor if gradient_predivide_factor is not None else \
            get_gradient_predivide_factor(self.world_size)
        self.gradient_postdivide_factor: float = self.world_size / self.gradient_predivide_factor

        self.comm_stream: torch.cuda.Stream = torch.cuda.Stream()
        self.reducer = ReduceScatterBucketer(reduce_scatter_bucket_size_mb)
        self._require_backward_grad_sync: bool = True

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        outputs = self.module(*args, **kwargs)
        return outputs

    def backward(self, loss):
        loss.backward()
        self._final_backward_hook()

    @torch.no_grad()
    def _final_backward_hook(self) -> None:
        if self._require_backward_grad_sync:
            # Flush any unreduced buckets in the post_backward stream.
            with torch.cuda.stream(self.comm_stream):
                self.reducer.flush()
            torch.cuda.current_stream().wait_stream(self.comm_stream)
            if self._cpu_offload:
                # Wait for the non-blocking GPU -> CPU grad transfers to finish.
                torch.cuda.current_stream().synchronize()
        self.reducer.free()
        for p in self.module.parameters():
            if not p.requires_grad:
                continue
            # Leave the gradient accumulation state as-is if not synchronizing this pass. This ensures p.grad
            # remains the unsharded gradient accumulated from prior no-sync passes, and _saved_grad_shard
            # remains the sharded gradient from the last synchronized pass. This also allows interleaved no-sync and
            # sync passes, if desired.
            if not self._require_backward_grad_sync:
                continue
            p._sharded_grad.write_back()

    @torch.no_grad()
    def _grad_post_backward_hook(self, param: Parameter, grad: torch.Tensor) -> Optional[torch.Tensor]:
        """
        At the start of :func:`_grad_post_backward_hook`, ``param.grad`` contains the
        full gradient for the local batch. The reduce-scatter op will save a single shard of the summed gradient across all
        GPUs to param._sharded_grad. This shard will align with the current GPU rank. For example::

            before reduce_scatter:
                param.grad (GPU #0): [1, 2, 3, 4]
                param.grad (GPU #1): [5, 6, 7, 8]

            after reduce_scatter:
                param.grad (GPU #0): [6, 8]    # 1+5, 2+6
                param.grad (GPU #1): [10, 12]  # 3+7, 4+8

        The local GPU's ``optim.step`` is responsible for updating a single
        shard of params, also corresponding to the current GPU's rank. This
        alignment is created by `param._sharded_grad`, which ensures that
        the local optimizer only sees the relevant parameter shard.
        """
        if grad is None:
            return
        assert not grad.requires_grad, 'ShardedModel only works with gradients that don\'t require gradients'
        if not self._require_backward_grad_sync:
            return
        self.comm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.comm_stream):
            new_grad = grad.clone()
            if self.mixed_precision and self.fp32_reduce_scatter:
                new_grad.data = new_grad.data.to(param.dtype)
            if self.gradient_predivide_factor > 1.0:
                # Average grad by world_size for consistency with PyTorch DDP.
                new_grad.data.div_(self.gradient_predivide_factor)
            orig_grad_data = new_grad.data
            if self.world_size > 1:
                grad_chunks = chunk_and_pad(orig_grad_data, self.reduce_scatter_process_group.size())
                self.reducer.reduce_scatter_async(
                    grad_chunks, group=self.reduce_scatter_process_group, callback_fn=functools.partial(self._reduce_scatter_callback, param))
            else:
                self._reduce_scatter_callback(param, new_grad)
            orig_grad_data.record_stream(self.comm_stream)

    def _reduce_scatter_callback(self, param: Parameter, reduced_grad: torch.Tensor) -> None:
        if self.gradient_postdivide_factor > 1:
            # Average grad by world_size for consistency with PyTorch DDP.
            reduced_grad.data.div_(self.gradient_postdivide_factor)
        # Cast grad to param's dtype (typically FP32). Note: we do this
        # before the cpu offload step so that this entire hook remains
        # non-blocking. The downside is a bit more D2H transfer in that case.
        if self.mixed_precision:
            orig_param_grad_data = reduced_grad.data
            reduced_grad.data = reduced_grad.data.to(dtype=param.ca_attr.origin_dtype)
            # Don't let this memory get reused until after the transfer.
            orig_param_grad_data.record_stream(torch.cuda.current_stream())

        param._sharded_grad.reduce_scatter_callback(reduced_grad)
