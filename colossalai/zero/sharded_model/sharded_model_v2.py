import functools
import itertools
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Iterator, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.gemini.memory_tracer import MemStatsCollector, StaticMemStatsCollector
from colossalai.gemini.ophooks import register_ophooks_recursively
from colossalai.gemini.paramhooks import BaseParamHookMgr
from colossalai.gemini.stateful_tensor import TensorState
from colossalai.gemini.stateful_tensor_mgr import StatefulTensorMgr
from colossalai.gemini.tensor_placement_policy import TensorPlacementPolicy, TensorPlacementPolicyFactory
from colossalai.gemini.tensor_utils import colo_model_data_move_to_cpu
from colossalai.logging import get_dist_logger
from colossalai.utils import disposable, get_current_device
from colossalai.utils.memory import colo_device_memory_capacity
from colossalai.zero.shard_utils import BaseShardStrategy
from colossalai.zero.sharded_model.reduce_scatter import ReduceScatterBucketer
from colossalai.zero.utils import ZeroHook

from ._utils import (
    cast_float_arguments,
    cast_tensor_to_fp16,
    cast_tensor_to_fp32,
    chunk_and_pad,
    free_storage,
    get_gradient_predivide_factor,
)

try:
    from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX
except ImportError:
    _EXTRA_STATE_KEY_SUFFIX = '_extra_state'


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
                 reuse_fp16_shard: bool = False,
                 *args,
                 **kwargs):
        assert not isinstance(module, ShardedModelV2), 'Nested ShardedModelV2 is not supported.'
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
            self._memstats_collector = MemStatsCollector()
            self._start_collect_memstats = disposable(self._memstats_collector.start_collection)
            self._finish_collect_memstats = disposable(self._memstats_collector.finish_collection)
        else:
            self._memstats_collector = None
        self._tensor_placement_policy: TensorPlacementPolicy = TensorPlacementPolicyFactory.create(
            tensor_placement_policy)(mem_stats_collector=self._memstats_collector)

        if 'warmup_non_model_data_ratio' in kwargs:
            if tensor_placement_policy != 'auto':
                self.logger.warning('setting warmup_non_model_data_ratio is useless if not use auto placement')
            else:
                ratio = kwargs['warmup_non_model_data_ratio']
                self._tensor_placement_policy._warmup_non_model_data_ratio = ratio
                self.logger.info(f'setting warmup_non_model_data_ratio as {ratio} for auto placement')

        self._stateful_tensor_mgr = StatefulTensorMgr(self._tensor_placement_policy)
        param_tensor_list = [p.colo_attr.sharded_data_tensor for p in module.parameters() if hasattr(p, 'colo_attr')]
        self._stateful_tensor_mgr.register_stateful_tensor_list(param_tensor_list)

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
                    f.write('\n')
                    f.write('CUDA non model data (GB)\n')
                    f.write(str(self._memstats_collector._memstats.non_model_data_list('cuda')))
                    f.write('CPU non model data (GB)\n')
                    f.write(str(self._memstats_collector._memstats.non_model_data_list('cpu')))
                    f.write('\n')

    def _pre_forward_operations(self, *args):
        # the operation will affect the memory tracer behavior in ZeroHook
        if self._memstats_collector:
            self._start_collect_memstats()

        for p in self.module.parameters():
            if hasattr(p, 'colo_attr'):
                p.colo_attr.sharded_data_tensor.trans_state(TensorState.HOLD)

        self._stateful_tensor_mgr.start_iter()

    def _post_forward_operations(self):
        for p in self.module.parameters():
            if hasattr(p, 'colo_attr'):
                p.colo_attr.sharded_data_tensor.trans_state(TensorState.HOLD)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._pre_forward_operations(*args)
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
            self._cuda_margin_space = colo_device_memory_capacity(
                get_current_device()) - self._memstats_collector._memstats.max_overall_cuda

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

            param.colo_attr.grad_payload_reset(grad.data)
            # release the memory of param
            # we set a false None for parameter's payload
            # so we can get paramter's device and dtype later in optimizer
            param.colo_attr.data_payload_reset(torch.empty(0, device=grad.device, dtype=grad.dtype))

            if param.colo_attr.is_replicated:
                param.colo_attr.sharded_data_tensor.is_sharded = True
        else:

            fp32_grad = cast_tensor_to_fp32(grad)

            if param.colo_attr.saved_grad.is_null():
                param.colo_attr.grad_payload_reset(fp32_grad)
            else:
                param.colo_attr.grad_payload.add_(fp32_grad.view_as(param.colo_attr.grad_payload))

        # keep saved_grad in HOLD state
        param.colo_attr.saved_grad.trans_state(TensorState.HOLD)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.module.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        return self.module.named_parameters(prefix, recurse)

    def state_dict(self, destination=None, prefix='', keep_vars=False) -> 'OrderedDict[str, torch.Tensor]':
        return self._colo_state_dict(destination,
                                     prefix,
                                     keep_vars,
                                     shard_strategy=self.shard_strategy,
                                     state_dict_func=nn.Module.state_dict,
                                     module_to_load=self.module,
                                     sharded_params=self.sharded_params,
                                     process_group=self.process_group)

    def load_state_dict(self, state_dict: 'OrderedDict[str, torch.Tensor]', strict: bool = True) -> None:
        for name, p in self.named_parameters():
            if name in state_dict:
                p.colo_attr.data_payload_reset(state_dict[name].to(dtype=p.colo_attr.data_payload.dtype,
                                                                   device=p.colo_attr.data_payload.device))
                # Force re-shard
                p.colo_attr.sharded_data_tensor.is_sharded = False
                self.shard_strategy.shard([p.colo_attr.sharded_data_tensor])
            elif strict:
                raise RuntimeError(f'Missing key in state_dict: {name}')

    def _colo_state_dict(self,
                         destination=None,
                         prefix='',
                         keep_vars=False,
                         shard_strategy: Optional[BaseShardStrategy] = None,
                         state_dict_func=None,
                         module_to_load=None,
                         sharded_params=[],
                         process_group=None) -> 'OrderedDict[str, torch.Tensor]':
        if len(sharded_params) == 0:
            for param in self.parameters():
                if param.colo_attr.param_is_sharded:
                    sharded_params.append(param)
        if shard_strategy is not None:
            shard_strategy.gather([p.colo_attr.sharded_data_tensor for p in sharded_params], process_group)
        for p in sharded_params:
            p.data = p.colo_attr.data_payload
        module_to_load = module_to_load or self
        gathered_state_dict = state_dict_func(module_to_load, destination, prefix, keep_vars)
        gathered_state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in gathered_state_dict.items()}
        if shard_strategy is not None:
            shard_strategy.shard([p.colo_attr.sharded_data_tensor for p in sharded_params], process_group)
        for p in sharded_params:
            p.colo_attr.set_data_none()
        return gathered_state_dict

    def _colo_load_from_state_dict(self,
                                   state_dict,
                                   prefix,
                                   local_metadata,
                                   strict,
                                   missing_keys,
                                   unexpected_keys,
                                   error_msgs,
                                   shard_strategy=None):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                if hasattr(param, 'colo_attr'):
                    param.colo_attr.data_payload_reset(
                        input_param.to(dtype=param.colo_attr.data_payload.dtype,
                                       device=param.colo_attr.data_payload.device))
                    if shard_strategy is not None:
                        # Force re-shard
                        param.colo_attr.sharded_data_tensor.is_sharded = False
                        shard_strategy.shard([param.colo_attr.sharded_data_tensor])
                else:
                    # This is used to avoid copying uninitialized parameters into
                    # non-lazy modules, since they dont have the hook to do the checks
                    # in such case, it will error when accessing the .shape attribute.
                    is_param_lazy = torch.nn.parameter.is_lazy(param)
                    # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                    if not is_param_lazy and len(param.shape) == 0 and len(input_param.shape) == 1:
                        input_param = input_param[0]

                    if not is_param_lazy and input_param.shape != param.shape:
                        # local shape should match the one in checkpoint
                        error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                          'the shape in current model is {}.'.format(
                                              key, input_param.shape, param.shape))
                        continue
                    try:
                        with torch.no_grad():
                            param.copy_(input_param)
                    except Exception as ex:
                        error_msgs.append('While copying the parameter named "{}", '
                                          'whose dimensions in the model are {} and '
                                          'whose dimensions in the checkpoint are {}, '
                                          'an exception occurred : {}.'.format(key, param.size(), input_param.size(),
                                                                               ex.args))
            elif strict:
                missing_keys.append(key)

        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if getattr(self.__class__, "set_extra_state", nn.Module.set_extra_state) is not nn.Module.set_extra_state:
            if extra_state_key in state_dict:
                self.set_extra_state(state_dict[extra_state_key])
            elif strict:
                missing_keys.append(extra_state_key)
        elif strict and (extra_state_key in state_dict):
            unexpected_keys.append(extra_state_key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix) and key != extra_state_key:
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.', 1)[0]    # get the name of param/buffer/child
                    if input_name not in self._modules and input_name not in local_state:
                        unexpected_keys.append(key)

    def __getitem__(self, idx: int):
        assert isinstance(self.module, nn.ModuleList)
        return self.module[idx]

    def __len__(self):
        assert isinstance(self.module, nn.ModuleList)
        return len(self.module)

    def __iter__(self):
        assert isinstance(self.module, nn.ModuleList)
        return iter(self.module)
