from ast import Try
import functools
from collections import OrderedDict
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.engine.ophooks import register_ophooks_recursively
from colossalai.engine.ophooks.zero_hook import ZeroHook
from colossalai.engine.paramhooks import BaseParamHookMgr
from colossalai.logging import get_dist_logger
from colossalai.zero.shard_utils import BaseShardStrategy
from colossalai.zero.sharded_model.reduce_scatter import ReduceScatterBucketer
from colossalai.zero.sharded_param import ShardedParamV2
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter
from colossalai.utils.memory_tracer.memstats_collector import MemStatsCollector
from colossalai.utils.memory_tracer.allocator import col_move_to_cpu
from ._zero3_utils import (cast_float_arguments, cast_tensor_to_fp16, cast_tensor_to_fp32, chunk_and_pad,
                           get_gradient_predivide_factor)


class ShardedModelV2(nn.Module):

    def __init__(self,
                 module: nn.Module,
                 shard_strategy: BaseShardStrategy,
                 process_group: Optional[ProcessGroup] = None,
                 reduce_scatter_process_group: Optional[ProcessGroup] = None,
                 reduce_scatter_bucket_size_mb: int = 25,
                 fp32_reduce_scatter: bool = False,
                 offload_config: Optional[dict] = None,
                 gradient_predivide_factor: Optional[float] = 1.0,
                 shard_param: bool = True,
                 use_memory_tracer: bool = False):
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

        # Cast module to fp16 and cuda, in case user didn't use ZeroInitContext
        self.module = module.half().cuda()

        self.shard_strategy = shard_strategy
        self.shard_param = shard_param

        # In case user didn't use ZeroInitContext
        for param in self.module.parameters():
            if not hasattr(param, 'col_attr'):
                param.col_attr = ShardedParamV2(param, process_group, rm_torch_payload=True)
                if self.shard_param:
                    self.shard_strategy.shard([param.col_attr.data])

        # Init Memory Statistics Collector
        self._use_memory_tracer = use_memory_tracer
        if self._use_memory_tracer:
            self._memstats_collector = MemStatsCollector()
        else:
            self._memstats_collector = None
        self._iter_cnter = 0

        # Register hooks
        register_ophooks_recursively(self.module, [ZeroHook(self.shard_strategy, self._memstats_collector)])
        self.param_hook_mgr = BaseParamHookMgr(list(self.module.parameters()))
        self.param_hook_mgr.register_backward_hooks(self._grad_post_backward_hook)

        self.fp32_reduce_scatter = fp32_reduce_scatter
        self._cpu_offload: bool = offload_config.get('device', None) == 'cpu' if offload_config else False
        # We find if gradient_predivide_factor != 1.0, there may be wrong precision problem
        # So we use 1.0 as the default gradient_predivide_factor
        # However, if you set gradient_predivide_factor to None, we will set
        # gradient_predivide_factor to a value >= 1.0 automatically
        self.gradient_predivide_factor: float = gradient_predivide_factor if \
            gradient_predivide_factor is not None else \
            get_gradient_predivide_factor(self.world_size)
        self.gradient_postdivide_factor: float = self.world_size / self.gradient_predivide_factor

        self.comm_stream: torch.cuda.Stream = torch.cuda.Stream()
        self.reducer = ReduceScatterBucketer(reduce_scatter_bucket_size_mb)
        self._require_backward_grad_sync: bool = True

    @property
    def cpu_offload(self):
        return self._cpu_offload

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self._iter_cnter == 0 and self._memstats_collector:
            # the opeartion will affect the flag in ZeroHook
            self._memstats_collector.start_collection()
        args, kwargs = cast_float_arguments(cast_tensor_to_fp16, *args, **kwargs)
        outputs = self.module(*args, **kwargs)
        return outputs

    def backward(self, loss):
        loss.backward()
        self._final_backward_hook()

    def backward_by_grad(self, tensor, grad):
        torch.autograd.backward(tensors=tensor, grad_tensors=grad)
        self._final_backward_hook()

    @torch.no_grad()
    def _final_backward_hook(self) -> None:
        if self._iter_cnter == 0 and self._memstats_collector:
            self._memstats_collector.finish_collection()
        if self._memstats_collector:
            self._memstats_collector.reset_sampling_cnter()
        self._iter_cnter += 1

        if self._require_backward_grad_sync:
            # Flush any unreduced buckets in the post_backward stream.
            with torch.cuda.stream(self.comm_stream):
                self.reducer.flush()
            torch.cuda.current_stream().wait_stream(self.comm_stream)
            if self._cpu_offload:
                # Wait for the non-blocking GPU -> CPU grad transfers to finish.
                torch.cuda.current_stream().synchronize()
        self.reducer.free()
        # In case some post bwd hook is not fired
        if self.shard_param:
            for p in self.module.parameters():
                if not p.col_attr.param_is_sharded:
                    self.shard_strategy.shard([p.col_attr.data])
        for p in self.module.parameters():
            p.col_attr.bwd_count = 0
            if not p.requires_grad:
                continue
            # Leave the gradient accumulation state as-is if not synchronizing this pass. This ensures p.grad
            # remains the unsharded gradient accumulated from prior no-sync passes, and _saved_grad_shard
            # remains the sharded gradient from the last synchronized pass. This also allows interleaved no-sync and
            # sync passes, if desired.
            if not self._require_backward_grad_sync:
                continue
            # Write grad back to p.grad and set p.col_attr.grad to None
            # As sharded optimizer only update a shard of param,
            # no matter whether we shard param in sharded model
            # We have to make sure the grad is a flat tensor shard
            # If world size == 1 and sharded param,
            # the shape `grad` is the same as unsharded param
            # So we can just use `view(-1)` to ensure grad is a flat tensor shard
            p.grad.data = p.col_attr.grad.view(-1)
            p.col_attr.grad = None

    @torch.no_grad()
    def _grad_post_backward_hook(self, param: Parameter, grad: torch.Tensor) -> Optional[torch.Tensor]:
        """
        At the start of :func:`_grad_post_backward_hook`, ``param.grad`` contains the
        full gradient for the local batch. The reduce-scatter op will save
        a single shard of the summed gradient across all
        GPUs to param.col_attr.grad. This shard will align with the current GPU rank. For example::

            before reduce_scatter:
                param.grad (GPU #0): [1, 2, 3, 4]
                param.grad (GPU #1): [5, 6, 7, 8]

            after reduce_scatter:
                param.grad (GPU #0): [6, 8]    # 1+5, 2+6
                param.grad (GPU #1): [10, 12]  # 3+7, 4+8

        The local GPU's ``optim.step`` is responsible for updating a single
        shard of params, also corresponding to the current GPU's rank. This
        alignment is created by `param.col_attr.grad`, which ensures that
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
            if self.fp32_reduce_scatter:
                new_grad.data = new_grad.data.to(param.dtype)
            if self.gradient_predivide_factor > 1.0:
                # Average grad by world_size for consistency with PyTorch DDP.
                new_grad.data.div_(self.gradient_predivide_factor)
            orig_grad_data = new_grad.data
            if self.world_size > 1:
                grad_chunks = chunk_and_pad(orig_grad_data, self.reduce_scatter_process_group.size())
                self.reducer.reduce_scatter_async(grad_chunks,
                                                  group=self.reduce_scatter_process_group,
                                                  callback_fn=functools.partial(self._reduce_scatter_callback, param))
            else:
                self._reduce_scatter_callback(param, new_grad)
            orig_grad_data.record_stream(self.comm_stream)

    def _reduce_scatter_callback(self, param: Parameter, reduced_grad: torch.Tensor) -> None:
        if self.gradient_postdivide_factor > 1:
            # Average grad by world_size for consistency with PyTorch DDP.
            reduced_grad.data.div_(self.gradient_postdivide_factor)

        # Make sure we store fp32 grad
        reduced_grad.data = cast_tensor_to_fp32(reduced_grad.data)

        # Maybe offload
        # TODO() optimize GPU->CPU bandwidth utilization
        if self._cpu_offload:
            col_move_to_cpu(reduced_grad)
            # reduced_grad.data = reduced_grad.data.cpu()

        if param.col_attr.grad is None:
            param.col_attr.grad = reduced_grad.data
        else:
            # When dp size = 1
            # param.col_attr.grad is local accumulated grad shard (full but flatten)
            # But reduced_grad here is full grad
            # We should call `view_as`
            param.col_attr.grad.add_(reduced_grad.data.view_as(param.col_attr.grad))

    def state_dict(self, destination=None, prefix='', keep_vars=False) -> 'OrderedDict[str, torch.Tensor]':
        self.shard_strategy.gather([p.col_attr.data for p in self.module.parameters()])
        prev_params = {}
        for p in self.module.parameters():
            prev_params[p] = p.data
            p.data = p.col_attr.data.payload
        gathered_state_dict = self.module.state_dict(destination, prefix, keep_vars)
        self.shard_strategy.shard([p.col_attr.data for p in self.module.parameters()])
        for p in self.module.parameters():
            p.data = prev_params[p]
        return gathered_state_dict

    def load_state_dict(self, state_dict: 'OrderedDict[str, torch.Tensor]', strict: bool = True):
        raise NotImplementedError
