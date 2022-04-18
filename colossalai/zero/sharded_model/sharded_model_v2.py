import functools
from collections import OrderedDict
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.engine.ophooks import register_ophooks_recursively
from colossalai.zero.utils import ZeroHook
from colossalai.engine.paramhooks import BaseParamHookMgr
from colossalai.logging import get_dist_logger
from colossalai.utils import get_current_device, disposable
from colossalai.utils.memory_tracer.memstats_collector import MemStatsCollector
from colossalai.utils.memory_tracer.model_data_memtracer import \
    GLOBAL_MODEL_DATA_TRACER
from colossalai.utils.memory import colo_device_memory_capacity
from colossalai.zero.shard_utils import BaseShardStrategy
from colossalai.zero.sharded_param.tensor_utils import colo_model_data_move_to_cpu
from colossalai.zero.sharded_model.reduce_scatter import ReduceScatterBucketer
from colossalai.zero.sharded_param.tensorful_state import TensorState
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter
from colossalai.gemini.stateful_tensor_mgr import StatefulTensorMgr
from colossalai.gemini.tensor_placement_policy import TensorPlacementPolicyFactory, TensorPlacementPolicy

from ._utils import (cast_float_arguments, cast_tensor_to_fp16, cast_tensor_to_fp32, chunk_and_pad, free_storage,
                     get_gradient_predivide_factor)


class ShardedModelV2(nn.Module):
    """
    A wrapper for the PyTorch module shards the model parameters among multiple GPU memory.
    Only `1/#nproc` of parameters, gradients are stored in local CUDA memory, so forward and backward
    passes can be executed with limited CUDA memory budget.

    Note:
        You must use ``ShardedModelV2`` with ``ShardedOptimizerV2``.
    Note:
        Make sure you don't use gradient accumulation and your optimizer can work with fp16 gradient and fp32 parameter,
        if you enable ``reuse_fp16_shard``.

    Args:
        module (nn.Module): A sharded module, which must be initialized by `ZeroInitContext`.
        shard_strategy (BaseShardStrategy): A shard strategy to manage shard behavior.
        process_group (Optional[ProcessGroup], optional): Data parallel process group. Defaults to None.
        reduce_scatter_process_group (Optional[ProcessGroup], optional): Reduce-scatter process group. 
            Generally, it should be `None`, and it's the same as `process_group`. Defaults to None.
        reduce_scatter_bucket_size_mb (int, optional): Reduce-scatter bucket size in *MB*. Defaults to 25.
        fp32_reduce_scatter (bool, optional): If set to `True`, gradients are forced to FP32 before reduce-scatter. Defaults to False.
        tensor_placement_policy (str): Which device to place *held* tensors. It can be 'cpu', 'cuda' and 'auto'.
            If it's 'cpu', parameters, gradients and optimizer states will be offloaded to CPU, which means min CUDA memory will be used.
            If it's 'cuda', they won't be offloaded, which means max CUDA memory will be used.
            If it's 'auto', they are moving dynamically based on CPU and CUDA memory usage. It will utilize heterogeneous memory space evenly and well.
            Note that 'auto' policy can only work well when no other processes use CUDA during your training.
            Defaults to 'cuda'.
        gradient_predivide_factor (Optional[float], optional): Gradient is divived by this value before reduce-scatter. Defaults to 1.0.
        reuse_fp16_shard (bool, optional): Whether to reuse fp16 shard for param and grad. 
            Enabling this can reduce GPU memory usage, but you have to make sure you disable it when using gradient accumulation. 
            In this mode, grad will be fp16. Make sure your optimizer supports mixed precision (fp32 param and fp16 grad). 
            We find that PyTorch's optimizers don't support mixed precision, 
            so we recommend you enable this only when using our CPUAdam with CPU offload. Defaults to False.
    """

    def __init__(self,
                 module: nn.Module,
                 shard_strategy: BaseShardStrategy,
                 process_group: Optional[ProcessGroup] = None,
                 reduce_scatter_process_group: Optional[ProcessGroup] = None,
                 reduce_scatter_bucket_size_mb: int = 25,
                 fp32_reduce_scatter: bool = False,
                 tensor_placement_policy: str = 'cuda',
                 gradient_predivide_factor: Optional[float] = 1.0,
                 reuse_fp16_shard: bool = False):
        super().__init__()
        self.logger = get_dist_logger()

        # We force users to use ZeroInitContext
        for submodule in module.modules():
            sharded_cnt = 0
            unshard_cnt = 0
            for param in submodule.parameters(recurse=False):
                assert hasattr(param, 'colo_attr'), 'You must use ZeroInitContext to init your module first.'
                if param.colo_attr.param_is_sharded:
                    sharded_cnt += 1
                else:
                    unshard_cnt += 1
            assert (not sharded_cnt) or (not unshard_cnt), 'nn.Module can not both have shard param and unshard param'
            submodule.param_is_sharded = (sharded_cnt > 0)

        self.sharded_params = []
        self.unshard_params = []
        for param in module.parameters():
            if param.colo_attr.param_is_sharded:
                self.sharded_params.append(param)
            else:
                self.unshard_params.append(param)

        self.module = module
        self.process_group = process_group or gpc.get_group(ParallelMode.DATA)
        self.reduce_scatter_process_group = reduce_scatter_process_group or self.process_group
        self.world_size = dist.get_world_size(self.process_group)
        self.rank = dist.get_rank(self.process_group)
        self.shard_strategy = shard_strategy

        self._use_memory_tracer = tensor_placement_policy == 'auto'
        if self._use_memory_tracer:
            GLOBAL_MODEL_DATA_TRACER.register_model(self)
            self._memstats_collector = MemStatsCollector()
            self._start_collect_memstats = disposable(self._memstats_collector.start_collection)
            self._finish_collect_memstats = disposable(self._memstats_collector.finish_collection)
        else:
            self._memstats_collector = None
        self._tensor_placement_policy: TensorPlacementPolicy = TensorPlacementPolicyFactory.create(
            tensor_placement_policy)(mem_stats_collector=self._memstats_collector)
        self._stateful_tensor_mgr = StatefulTensorMgr(self._tensor_placement_policy)
        for param in module.parameters():
            if hasattr(param, 'colo_attr'):
                self._stateful_tensor_mgr.register_stateful_param(param.colo_attr)

        # Register hooks
        self._ophook_list = [
            ZeroHook(self.shard_strategy, self._memstats_collector, self._stateful_tensor_mgr, self.process_group)
        ]
        register_ophooks_recursively(self.module, self._ophook_list)
        self.param_hook_mgr = BaseParamHookMgr(list(self.module.parameters()))
        self.param_hook_mgr.register_backward_hooks(self._grad_post_backward_hook)

        self.fp32_reduce_scatter = fp32_reduce_scatter
        self._cpu_offload: bool = tensor_placement_policy != 'cuda'
        for param in module.parameters():
            # Init `offload_grad`
            param.colo_attr.offload_grad = self._cpu_offload

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

        self._cuda_margin_space = 0
        self.reuse_fp16_shard = reuse_fp16_shard

        # record whether gradients have inf or nan
        self.overflow_counter = 0

    def adjust_stateful_tensor_layout(self) -> None:
        self._stateful_tensor_mgr.adjust_layout()

    @property
    def use_memory_tracer(self):
        return self._use_memory_tracer

    @property
    def cuda_margin_space(self):
        return self._cuda_margin_space

    @property
    def cpu_offload(self):
        return self._cpu_offload

    def dump_memory_stats(self, filename: Optional[str] = 'dump_mem_stats.log') -> None:
        """
        dummy memory tracer collected infomation to a file.
        try:
            # forward: model(inputs)
            # backward: optimizer.backward()
        except Exception as e:
            model.dump_memory_stats()
            exit(0)
        """
        if self._use_memory_tracer:
            self.logger.error(f'dump memort tracer collected infomation to a {filename}', ranks=[0])
            if gpc.get_global_rank() == 0:
                with open(filename, 'w+') as f:
                    f.write(f'cuda reserved {torch.cuda.memory_reserved(get_current_device()) / 1e9} GB\n')
                    f.write(f'cuda max allocated {torch.cuda.max_memory_allocated(get_current_device()) / 1e9} GB\n')
                    f.write('CUDA model data (GB)\n')
                    f.write(str(self._memstats_collector.model_data_list('cuda', 'GB')))
                    f.write('\n')
                    f.write('CUDA non model data (GB)\n')
                    f.write(str(self._memstats_collector.non_model_data_list('cuda', 'GB')))
                    f.write('CPU non model data (GB)\n')
                    f.write(str(self._memstats_collector.non_model_data_list('cpu', 'GB')))
                    f.write('\n')

    def _pre_forward_operations(self):
        # the operation will affect the memory tracer behavior in ZeroHook
        if self._memstats_collector:
            self._start_collect_memstats()

        for p in self.module.parameters():
            if hasattr(p, 'colo_attr'):
                p.colo_attr.sharded_data_tensor.trans_state(TensorState.HOLD)

    def _post_forward_operations(self):
        for p in self.module.parameters():
            if hasattr(p, 'colo_attr'):
                p.colo_attr.sharded_data_tensor.trans_state(TensorState.HOLD)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._pre_forward_operations()
        args, kwargs = cast_float_arguments(cast_tensor_to_fp16, *args, **kwargs)
        outputs = self.module(*args, **kwargs)
        self._post_forward_operations()
        return outputs

    def backward(self, loss):
        loss.backward()
        self._post_backward_operations()
        for ophook in self._ophook_list:
            ophook.post_iter()

    def backward_by_grad(self, tensor, grad):
        torch.autograd.backward(tensors=tensor, grad_tensors=grad)
        self._post_backward_operations()
        for ophook in self._ophook_list:
            ophook.post_iter()

    def _update_memstats(self):
        if self._memstats_collector:
            self._finish_collect_memstats()
            # cuda margin space = cuda mem capacity - max fwd/bwd cuda mem used.
            # the way to calculate margin space is based on the assumption that
            # model data is fixed in cuda during training.
            # cuda margin space can be used to store OS.
            self._cuda_margin_space = colo_device_memory_capacity(get_current_device()) - max(
                self._memstats_collector.overall_mem_stats('cuda'))

    @torch.no_grad()
    def _post_backward_operations(self) -> None:
        """
        The method includes operations required to be processed after backward
        1. update memory tracer.
        2. flush the gradient in buckets. Reducing partial gradients in each process.
        3. shard tensors not dealed in the zero hook
        4. move sharded param grad payload to param.grad
        """
        # 1. update memory tracer.
        self._update_memstats()

        # 2. flush the gradient in buckets. Reducing partial gradients in each process.
        if self._require_backward_grad_sync:
            # Flush any unreduced buckets in the post_backward stream.
            with torch.cuda.stream(self.comm_stream):
                self.reducer.flush()
            torch.cuda.current_stream().wait_stream(self.comm_stream)
        self.reducer.free()

        # 3. shard tensors not dealed in the zero hook
        tensor_list = []
        for p in self.sharded_params:
            if not p.colo_attr.param_is_sharded:
                tensor_list.append(p.colo_attr.sharded_data_tensor)
                p.colo_attr.sharded_data_tensor.trans_state(TensorState.HOLD_AFTER_BWD)
                p.colo_attr.set_data_none()
        self.shard_strategy.shard(tensor_list, self.process_group)

        # 4. set all parameters' grad to None
        for p in self.module.parameters():
            if not p.requires_grad:
                continue
            # Leave the gradient accumulation state (_require_backward_grad_sync) as-is if not synchronizing this pass.
            # NOTE() (no-sync)/sync pass: (not conduct)/conduct gradient allreducing between process group.
            # If _require_backward_grad_sync is True,
            # p.grad remains the accumulated unsharded gradient from prior no-sync passes.
            # We also allows to interleave no-sync pass with sync passes, if desired.
            if not self._require_backward_grad_sync:
                continue

            p.grad = None

    @torch.no_grad()
    def _grad_post_backward_hook(self, param: Parameter, grad: torch.Tensor) -> Optional[torch.Tensor]:
        """
        At the start of :func:`_grad_post_backward_hook`, ``param.grad`` contains the
        full gradient for the local batch. The reduce-scatter op will save
        a single shard of the summed gradient across all
        GPUs to param.colo_attr.grad. This shard will align with the current GPU rank. For example::

            before reduce_scatter:
                param.grad (GPU #0): [1, 2, 3, 4]
                param.grad (GPU #1): [5, 6, 7, 8]

            after reduce_scatter:
                param.grad (GPU #0): [6, 8]    # 1+5, 2+6
                param.grad (GPU #1): [10, 12]  # 3+7, 4+8

        The local GPU's ``optim.step`` is responsible for updating a single
        shard of params, also corresponding to the current GPU's rank. This
        alignment is created by `param.colo_attr.grad`, which ensures that
        the local optimizer only sees the relevant parameter shard.
        """
        if grad is None:
            return
        assert not grad.requires_grad, 'ShardedModel only works with gradients that don\'t require gradients'
        if not self._require_backward_grad_sync:
            return
        # used to cheat Pytorch, since we can't return None
        empty_grad = torch.empty_like(grad)
        free_storage(empty_grad)
        # As torch didn't allow modifying grad in hook, we make a copy
        grad = grad.clone()
        if param.colo_attr.is_replicated:
            self._reduce_scatter_handler(param, grad)
        else:
            self._save_grad(param, grad)
        return empty_grad

    def _reduce_scatter_handler(self, param: Parameter, grad: torch.Tensor) -> None:
        self.comm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.comm_stream):
            if self.fp32_reduce_scatter:
                grad.data = grad.data.to(param.dtype)
            if self.gradient_predivide_factor > 1.0:
                # Average grad by world_size for consistency with PyTorch DDP.
                grad.data.div_(self.gradient_predivide_factor)
            if self.world_size > 1:
                grad_chunks = chunk_and_pad(grad, self.reduce_scatter_process_group.size())
                self.reducer.reduce_scatter_async(grad_chunks,
                                                  group=self.reduce_scatter_process_group,
                                                  callback_fn=functools.partial(self._reduce_scatter_callback, param))
            else:
                self._reduce_scatter_callback(param, grad)
        torch.cuda.current_stream().wait_stream(self.comm_stream)

    def _reduce_scatter_callback(self, param: Parameter, reduced_grad: torch.Tensor) -> None:
        assert isinstance(reduced_grad,
                          torch.Tensor), f"_reduce_scatter_callback accept reduced_grad as {type(reduced_grad)}"
        reduced_grad.data = reduced_grad.data.contiguous().view(-1)
        if self.gradient_postdivide_factor > 1:
            # Average grad by world_size for consistency with PyTorch DDP.
            reduced_grad.data.div_(self.gradient_postdivide_factor)
        self._save_grad(param, reduced_grad)

    # FIXME(ver217): refactor the below line when impl eviction policy
    def _save_grad(self, param: Parameter, grad: torch.Tensor):

        # record whether we have overflow
        self.overflow_counter += torch.isinf(grad).any().item()
        self.overflow_counter += torch.isnan(grad).any().item()

        # move gradient to cpu
        if param.colo_attr.offload_grad:
            colo_model_data_move_to_cpu(grad)

        if self.reuse_fp16_shard:
            # make parameters point to gradient

            assert param.colo_attr.saved_grad.is_null(
            ), 'Gradien accumulation is not supported when reuse_fp16_shard=True'

            param.colo_attr.reset_grad_payload(grad.data)
            param.colo_attr.reset_data_payload(grad.data)    # release the memory of param

            if param.colo_attr.is_replicated:
                param.colo_attr.sharded_data_tensor.is_sharded = True
        else:

            fp32_grad = cast_tensor_to_fp32(grad)

            if param.colo_attr.saved_grad.is_null():
                param.colo_attr.reset_grad_payload(fp32_grad)
            else:
                param.colo_attr.grad_payload.add_(fp32_grad.view_as(param.colo_attr.grad_payload))

        # keep saved_grad in HOLD state
        param.colo_attr.saved_grad.trans_state(TensorState.HOLD)

    def state_dict(self, destination=None, prefix='', keep_vars=False) -> 'OrderedDict[str, torch.Tensor]':
        self.shard_strategy.gather([p.colo_attr.sharded_data_tensor for p in self.sharded_params], self.process_group)
        for p in self.sharded_params:
            p.data = p.colo_attr.data_payload
        gathered_state_dict = self.module.state_dict(destination, prefix, keep_vars)
        self.shard_strategy.shard([p.colo_attr.sharded_data_tensor for p in self.sharded_params], self.process_group)
        for p in self.sharded_params:
            p.colo_attr.set_data_none()
        return gathered_state_dict

    def load_state_dict(self, state_dict: 'OrderedDict[str, torch.Tensor]', strict: bool = True):
        raise NotImplementedError

    def __getitem__(self, idx: int):
        assert isinstance(self.module, nn.ModuleList)
        return self.module[idx]

    def __len__(self):
        assert isinstance(self.module, nn.ModuleList)
        return len(self.module)

    def __iter__(self):
        assert isinstance(self.module, nn.ModuleList)
        return iter(self.module)
