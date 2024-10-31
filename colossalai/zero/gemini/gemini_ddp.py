import itertools
from collections import OrderedDict
from contextlib import nullcontext
from functools import partial
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group

from colossalai.accelerator import get_accelerator
from colossalai.checkpoint_io.utils import StateDictSharder, gather_distributed_param
from colossalai.interface import ModelWrapper
from colossalai.lazy import LazyTensor
from colossalai.logging import get_dist_logger
from colossalai.quantization.fp8_hook import FP8Hook
from colossalai.tensor.colo_parameter import ColoParameter
from colossalai.tensor.d_tensor import (
    distribute_tensor,
    distribute_tensor_with_customization,
    get_device_mesh,
    get_global_shape,
    get_sharding_spec,
    init_as_dtensor,
    init_tensor_as_customization_distributed,
    is_customized_distributed_tensor,
    is_distributed_tensor,
)
from colossalai.tensor.padded_tensor import (
    init_as_padded_tensor,
    is_padded_tensor,
    to_padded_tensor,
    to_unpadded_tensor,
)
from colossalai.tensor.param_op_hook import ColoParamOpHookManager
from colossalai.utils import _cast_float, free_storage, get_non_persistent_buffers_set, is_ddp_ignored

from .chunk import Chunk, ChunkManager, TensorState, init_chunk_manager
from .gemini_hook import GeminiZeROHook
from .gemini_mgr import GeminiManager
from .memory_tracer import MemStats, OrderedParamGenerator
from .utils import get_temp_total_chunk_on_cuda

try:
    from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX, _IncompatibleKeys
except ImportError:
    _EXTRA_STATE_KEY_SUFFIX = "_extra_state"

__all__ = [
    "GeminiDDP",
]


class GeminiDDP(ModelWrapper):
    """ZeRO DDP.
    Warning: Nested GeminiDDP is not supported now.
    It is designed to be used with ChunkManager and GeminiManager.
    For more details, see the API reference of ``ChunkManager`` and ``GeminiManager``.

    Args:
        module (torch.nn.Module): Module to apply ZeRO-DP.
        gemini_manager (GeminiManager): Manages the chunk manager and heterogeneous memory space.
            For more details, see the API reference of ``GeminiManager``.
        pin_memory (bool): Chunks on CPU Memory use pin-memory.
        force_outputs_fp32 (bool): If set to True, outputs will be fp32. Otherwise, outputs will be fp16.
            Defaults to False.
        strict_ddp_mode (bool): If set to True, there is no tensor sharding, each tensor is replicated.
            Defaults to False. Users can set it to True, when they clearly know that they only need DDP.
        scatter_after_inference (bool): If set to True, the model will be scattered after inference. This will save memory but slow down the consecutive inference.
        mixed_precision (torch.dtype): If set to torch.float16, the model will be trained in fp16. Otherwise, the model will be trained in bf16. Defaults to torch.float16.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        chunk_config_dict: Optional[dict] = None,
        chunk_init_device: torch.device = torch.device("cpu"),
        placement_policy: str = "static",
        enable_gradient_accumulation: bool = False,
        max_prefetch: int = 0,
        shard_param_frac: float = 1.0,  # only for static placement
        offload_optim_frac: float = 0.0,  # only for static placement
        offload_param_frac: float = 0.0,  # only for static placement
        warmup_non_model_data_ratio: float = 0.8,  # only for auto placement
        steady_cuda_cap_ratio: float = 0.9,  # only for auto placement
        search_range_m: int = 32,  # chunk search options
        hidden_dim: Optional[int] = None,  # chunk search options
        min_chunk_size_m: float = 32,  # chunk search options
        pin_memory: bool = False,
        force_outputs_fp32: bool = False,
        strict_ddp_mode: bool = False,
        scatter_after_inference: bool = True,
        mixed_precision: torch.dtype = torch.float16,
        zero_group: Optional[ProcessGroup] = None,
        memstats: Optional[MemStats] = None,  # genimi memory stats
        master_weights: bool = True,
        extra_dp_group: Optional[ProcessGroup] = None,
        verbose: bool = False,
        enable_async_reduce: bool = True,
        fp8_communication: bool = False,
        use_fp8: bool = False,
    ) -> None:
        assert mixed_precision in (torch.float16, torch.bfloat16)
        reuse_fp16_chunk = master_weights if not enable_gradient_accumulation else False
        self.enable_gradient_accumulation = enable_gradient_accumulation
        if chunk_config_dict is not None:
            self.chunk_manager = ChunkManager(
                chunk_config_dict, chunk_init_device, reuse_fp16_chunk=reuse_fp16_chunk, max_prefetch=max_prefetch
            )
        else:
            # some ugly hotfix for the compatibility with Lightning
            if search_range_m is None:
                search_range_m = 32
            self.chunk_manager = init_chunk_manager(
                model=module,
                init_device=chunk_init_device,
                hidden_dim=hidden_dim,
                search_range_m=search_range_m,
                min_chunk_size_m=min_chunk_size_m,
                strict_ddp_flag=strict_ddp_mode,
                process_group=zero_group,
                reuse_fp16_chunk=reuse_fp16_chunk,
                verbose=verbose,
                max_prefetch=max_prefetch,
            )
        if fp8_communication:
            self.chunk_manager.fp8_communication = True
        self.gemini_manager = GeminiManager(
            placement_policy,
            self.chunk_manager,
            memstats,
            shard_param_frac=shard_param_frac,
            offload_optim_frac=offload_optim_frac,
            offload_param_frac=offload_param_frac,
            warmup_non_model_data_ratio=warmup_non_model_data_ratio,
            steady_cuda_cap_ratio=steady_cuda_cap_ratio,
            max_prefetch=max_prefetch,
        )
        self.force_outputs_fp32 = force_outputs_fp32
        self.param_op_hook = GeminiZeROHook(self.gemini_manager)
        self.hooks = [self.param_op_hook]
        if use_fp8:
            self.hooks.append(FP8Hook())
        self.fp32_params: List[torch.Tensor] = list()
        self.fp16_params: List[ColoParameter] = list()
        self.grads_device: Dict[torch.Tensor, torch.device] = dict()
        self.param2name: Dict[nn.Parameter, str] = dict()
        self.name2param: Dict[str, nn.Parameter] = dict()
        self.scatter_after_inference = scatter_after_inference
        self.mixed_precision = mixed_precision
        self.zero_group = zero_group or _get_default_group()
        self.extra_dp_group = extra_dp_group

        self.master_weights = master_weights
        self.enable_async_reduce = enable_async_reduce

        if enable_async_reduce:
            self.async_reduce_stream = torch.cuda.Stream()
        else:
            self.async_reduce_stream = None

        self._logger = get_dist_logger()

        if self.gemini_manager._premade_memstats_:
            # build chunk in param runtime visited order.
            param_order = self.gemini_manager.memstats()._param_runtime_order
        else:
            # build chunk in param initialized order.
            # Note: in this way, it can not get filter unused params during runtime.
            param_order = OrderedParamGenerator()
            for p in module.parameters():
                param_order.append(p)

        for name, param in module.named_parameters():
            self.param2name[param] = name
        for m_name, m_var in module.named_modules():
            for p_name, p_var in m_var.named_parameters(recurse=False):
                param_name = m_name + "." + p_name if m_name else p_name
                self.name2param[param_name] = p_var

        self._init_chunks(
            param_order=param_order,
            strict_ddp_mode=strict_ddp_mode,
            cpu_offload=not (self.gemini_manager.policy_name == "static" and offload_param_frac == 0),
            pin_memory=pin_memory,
        )
        super().__init__(module)
        self._non_persistent_buffers_set = get_non_persistent_buffers_set(module)
        self._cast_buffers()

        # register grad hook
        for p in module.parameters():
            if is_ddp_ignored(p):
                continue
            if p.requires_grad:
                assert not hasattr(p, "_grad_handle")
                p._grad_handle = p.register_hook(
                    partial(
                        GeminiDDP.grad_handle,
                        chunk_manager=self.chunk_manager,
                        param2name=self.param2name,
                        grads_device=self.grads_device,
                        master_weights=self.master_weights,
                        enable_gradient_accumulation=self.enable_gradient_accumulation,
                        p=p,
                        async_reduce_stream=self.async_reduce_stream,
                    )
                )

    def remove_hooks(self):
        for p in self.module.parameters():
            if is_ddp_ignored(p):
                continue
            if p.requires_grad:
                assert hasattr(p, "_grad_handle")
                p._grad_handle.remove()
                delattr(p, "_grad_handle")

    def __del__(self):
        self.remove_hooks()

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self.module.named_parameters(prefix, recurse)

    def named_buffers(self, prefix: str = "", recurse: bool = True):
        return self.module.named_buffers(prefix, recurse)

    def named_children(self):
        return self.module.named_children()

    def named_modules(
        self, memo: Optional[Set[torch.nn.Module]] = None, prefix: str = "", remove_duplicate: bool = True
    ):
        return self.module.named_modules(memo, prefix, remove_duplicate)

    @staticmethod
    def set_params_to_ignore(params_to_ignore: Iterable[torch.Tensor]) -> None:
        """Sets parameters to be ignored by DDP.
        This method must be called before initializing ColoDDP.

        Example:
            >>> params_to_ignore = []
            >>> for p in module.parameters():
            >>>     if should_ignore(p):
            >>>         params_to_ignore.append(p)
            >>> ColoDDP.set_params_to_ignore(params_to_ignore)
            >>> module = ColoDDP(module)

        Args:
            params_to_ignore (Iterable[torch.Tensor]): A list of parameters to be ignored.
        """
        for p in params_to_ignore:
            p._ddp_to_ignore = True

    def _post_forward(self):
        """This function is only triggered for inference."""
        access_list = list(self.chunk_manager.accessed_chunks)
        # we need to scatter all accessed chunks and move them to their original places
        for chunk in access_list:
            if chunk.keep_gathered:
                self.chunk_manager.fake_release_chunk(chunk)
            else:
                assert chunk.can_release
                self.chunk_manager.release_chunk(chunk)
            first_param = next(iter(chunk.tensors_info))
            self.chunk_manager.move_chunk(chunk, self.grads_device[first_param])
        assert self.chunk_manager.accessed_mem == 0

    def forward(self, *args, **kwargs):
        # check whether we are in a inference mode
        grad_flag = torch.is_grad_enabled()
        if not grad_flag:
            assert (
                not self.gemini_manager.need_warmup or not self.gemini_manager.is_warmup()
            ), "You should run a completed iteration as your warmup iter"

        args, kwargs = _cast_float(args, self.mixed_precision), _cast_float(kwargs, self.mixed_precision)
        self.module.zero_grad(set_to_none=True)
        if not grad_flag:
            outputs = self._inference_forward(*args, **kwargs)
        else:
            self.gemini_manager.pre_iter(*args)
            with ColoParamOpHookManager.use_hooks(*self.hooks):
                outputs = self.module(*args, **kwargs)

        if self.force_outputs_fp32:
            return _cast_float(outputs, torch.float)
        return outputs

    def _inference_forward(self, *args, **kwargs):
        """This function is only triggered for inference."""
        fwd_ctx = ColoParamOpHookManager.use_hooks(*self.hooks)
        if not self.scatter_after_inference:
            # gather all chunks
            for chunk in self.chunk_manager.get_chunks(self.fp16_params):
                self.chunk_manager.access_chunk(chunk)
            fwd_ctx = nullcontext()
        with fwd_ctx:
            outputs = self.module(*args, **kwargs)
        if self.scatter_after_inference:
            # scatter chunks
            self._post_forward()
        # reset all recorded attributes
        self.gemini_manager.reset_attributes()
        return outputs

    def _setup_grads_ptr(self):
        for p in self.module.parameters():
            if is_ddp_ignored(p):
                continue
            p.grad = None

    def _pre_backward(self):
        # set a visit label for all parameters
        # the label is used to check whether the parameter is correctly reduced
        for param in self.param2name:
            if not is_ddp_ignored(param):
                setattr(param, "_gemini_reduced", False)

    def _post_backward(self):
        if self.enable_async_reduce:
            self.async_reduce_stream.synchronize()

        if self.chunk_manager.accessed_mem != 0:
            error_params = ["Reduction failed at followed parameters:"]
            for param in self.param2name:
                if not is_ddp_ignored(param) and not getattr(param, "_gemini_reduced"):
                    error_params.append(self.param2name[param])
            error_str = "\n\t".join(error_params)
            raise RuntimeError(
                "ZERO DDP error: the synchronization of gradients doesn't exit properly.",
                "The most possible reason is that the model is not compatible with GeminiDDP.\n",
                f"{error_str}",
            )
        self._setup_grads_ptr()
        if self.enable_gradient_accumulation and not self.chunk_manager.accumulating_grads:
            self.chunk_manager.accumulating_grads = True  # Turn on the state of gradient accumulation.
        self._logger.debug(
            f"comp cuda demand time: {self.gemini_manager._comp_cuda_demand_time}, layout time: {self.gemini_manager._layout_time}, evict time: {self.gemini_manager._evict_time}, CPU->CUDA vol: {self.gemini_manager._h2d_volume}B, CUDA->CPU vol: {self.gemini_manager._d2h_volume}"
        )
        self.gemini_manager.post_iter()

    def backward(self, loss: torch.Tensor):
        self._pre_backward()
        with self.param_op_hook.switch_to_backward(), ColoParamOpHookManager.use_hooks(*self.hooks):
            loss.backward()
        self._post_backward()

    def backward_by_grad(self, tensor, grad):
        raise RuntimeError("Gemini is not compatible with pipeline. backward_by_grad shoudn't be called in Gemini.")

    @staticmethod
    def grad_handle(
        grad,
        chunk_manager: ChunkManager,
        param2name: Dict,
        grads_device: Dict,
        master_weights: bool,
        enable_gradient_accumulation: bool,
        p: nn.Parameter,
        async_reduce_stream: Optional[torch.cuda.Stream] = None,
    ):
        async_reduce_scatter = async_reduce_stream is not None
        setattr(p, "_gemini_reduced", True)
        empty_grad = torch.empty_like(grad)
        free_storage(empty_grad)
        with torch._C.DisableTorchFunction():
            chunk = chunk_manager.get_chunk(p)
            if chunk.tensors_info[p].state != TensorState.HOLD_AFTER_BWD:
                raise RuntimeError(
                    f"Parameter `{param2name[p]}` failed at the gradient reduction. "
                    "Some unsupported torch function is operated upon this parameter."
                )
            grad_chunk = chunk
            if not chunk_manager.reuse_fp16_chunk:
                if not chunk_manager.accumulating_grads:
                    grad_chunk = chunk_manager.init_grad_chunk(chunk)
                else:
                    assert chunk.grad_chunk is not None
                    if chunk.grad_chunk not in chunk_manager.accessed_chunks:
                        grad_chunk = chunk_manager.rearrange_accumulated_grad_chunk(chunk)
                    else:
                        grad_chunk = chunk.grad_chunk
                        chunk.grad_chunk.l2_norm = None

                # hold -> compute -> hold after bwd
                grad_chunk.tensor_trans_state(p, TensorState.COMPUTE)
                grad_chunk.tensor_trans_state(p, TensorState.HOLD_AFTER_BWD)
                # fp16 param chunk: hold after bwd -> ready for reduce -> hold
                chunk.tensor_trans_state(p, TensorState.READY_FOR_REDUCE)
                chunk.tensor_trans_state(p, TensorState.HOLD)

            grad_chunk.tensor_trans_state(p, TensorState.READY_FOR_REDUCE)
            if not chunk_manager.accumulating_grads:
                grad_chunk.copy_tensor_to_chunk_slice(p, grad, update_ptr=chunk_manager.reuse_fp16_chunk)
            else:
                grad_chunk.add_tensor_to_chunk_slice(p, grad)

            if async_reduce_stream is not None:
                async_reduce_stream.wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(async_reduce_stream):
                reduced = chunk_manager.reduce_chunk(grad_chunk, async_op=async_reduce_scatter)
                if reduced:
                    grad_chunk.wait_async_reduce()
                    if not chunk_manager.reuse_fp16_chunk:
                        if chunk.keep_gathered:
                            chunk_manager.fake_release_chunk(chunk)
                        else:
                            chunk_manager.release_chunk(chunk)
                    if grad_chunk.is_gathered:
                        grad_chunk.cuda_global_chunk.div_(chunk.pg_size)
                        if chunk.extra_dp_group is not None:
                            grad_chunk.cuda_global_chunk.div_(chunk.extra_dp_size)
                    else:
                        grad_chunk.cuda_shard.div_(chunk.pg_size)
                        if chunk.extra_dp_group is not None:
                            grad_chunk.cuda_shard.div_(chunk.extra_dp_size)
                            # check overflow elements
                    chunk_manager.overflow_counter += grad_chunk.has_inf_or_nan
                    # record l2 norm for gradient clipping. flag is bound to fp16 chunk
                    if chunk.l2_norm_flag:
                        grad_chunk.set_l2_norm()
                    chunk_manager.move_chunk(
                        grad_chunk, grads_device[p], force_copy=True, async_move=async_reduce_scatter
                    )
                    if not (master_weights) or (enable_gradient_accumulation):
                        chunk_manager.move_chunk(
                            chunk, grads_device[p], force_copy=True, async_move=async_reduce_scatter
                        )
        return empty_grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.module.zero_grad(set_to_none=True)

    def set_chunk_grad_device(self, chunk: Chunk, device: torch.device) -> None:
        for tensor in chunk.get_tensors():
            self.grads_device[tensor] = device

    def state_dict(self, destination=None, prefix="", keep_vars=False, only_rank_0: bool = True):
        """Returns a dictionary containing a whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are included.
        Keys are corresponding parameter and buffer names.
        Parameters and buffers set to ``None`` are not included.

        Warning: The non strict state dict would ignore the parameters if the tensors of the parameters
            are shared with other parameters which have been included in the dictionary.
            When you need to load the state dict, you should set the argument `strict` to False.

        Returns:
            dict:
                a dictionary containing a whole state of the module
        """
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]] = local_metadata = dict(version=self._version)
        self._save_to_state_dict(destination, prefix, keep_vars, only_rank_0)

        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination

    def _get_chunk_to_save_data(self, chunk: Chunk, only_rank_0: bool) -> Dict:
        """
        get gathered chunk content.

        Args:
            chunk (Chunk): a chunk
            only_rank_0 (bool): whether to only save data on rank 0

        Returns:
            Dict: a dict whose key is param name and value is param with correct payload
        """
        # save parameters
        chunk_to_save_data = dict()
        temp_chunk = get_temp_total_chunk_on_cuda(chunk, self.mixed_precision)

        for tensor, tensor_info in chunk.tensors_info.items():
            record_tensor = torch.empty([0])
            record_flag = (not only_rank_0) | (dist.get_rank(chunk.torch_pg) == 0)
            if record_flag:
                record_tensor = temp_chunk[tensor_info.offset : tensor_info.end].view(tensor.shape).to(tensor.device)
                if is_distributed_tensor(tensor):
                    global_shape = get_global_shape(tensor)
                    device_mesh = get_device_mesh(tensor)
                    shard_spec = get_sharding_spec(tensor)
                    record_tensor = init_as_dtensor(
                        record_tensor, device_mesh=device_mesh, sharding_spec=shard_spec, global_shape=global_shape
                    )
                elif is_customized_distributed_tensor(tensor):
                    init_tensor_as_customization_distributed(
                        record_tensor, shard_fn=tensor.shard_fn, gather_fn=tensor.gather_fn
                    )
                record_tensor = gather_distributed_param(record_tensor, keep_vars=False).cpu()
                if is_padded_tensor(tensor):
                    record_tensor = init_as_padded_tensor(
                        record_tensor, tensor._current_length, tensor._origin_length, tensor._padding_dim
                    )
                    record_tensor = to_unpadded_tensor(record_tensor)

            assert tensor not in chunk_to_save_data
            chunk_to_save_data[tensor] = record_tensor

        del temp_chunk
        return chunk_to_save_data

    def _get_param_to_save_data(self, param_list: List[torch.nn.Parameter], only_rank_0: bool) -> Dict:
        """
        get param content from chunks.

        Args:
            param_list (_type_): a list of torch.nn.Parameters
            only_rank_0 (_type_): _description_

        Returns:
            Dict: a dict whose key is param name and value is param with correct payload
        """
        # save parameters
        param_to_save_data = dict()
        chunk_list = self.chunk_manager.get_chunks(param_list)
        for chunk in chunk_list:
            param_to_save_data.update(self._get_chunk_to_save_data(chunk, only_rank_0))
        return param_to_save_data

    def _save_to_state_dict(self, destination, prefix, keep_vars, only_rank_0=True):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Args:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        assert keep_vars is False, "`state_dict` with parameter, `keep_vars=True`, is not supported now."

        # get copies of fp32 parameters in CPU
        # as memory of fp16_params may be reused by grad, it's not reliable, we should use fp32_params and convert to fp16
        params = self.fp32_params if self.chunk_manager.reuse_fp16_chunk else self.fp16_params
        param_to_save_data = self._get_param_to_save_data(params, only_rank_0)
        # get the mapping between copies and fp16 parameters
        p_mapping = dict()
        if self.chunk_manager.reuse_fp16_chunk:
            for p, fp32_p in zip(self.fp16_params, self.fp32_params):
                name = self.param2name[p]
                assert fp32_p in param_to_save_data, "Parameter '{}' is neglected in the chunk list".format(name)
                record_parameter = param_to_save_data[fp32_p]
                p_mapping[p] = record_parameter
        else:
            p_mapping = param_to_save_data
        for name, param in self.name2param.items():
            if param is not None:
                if is_ddp_ignored(param):
                    # deal with ddp ignored parameters
                    destination[prefix + name] = param if keep_vars else param.detach()
                else:
                    if is_padded_tensor(p_mapping[param]):
                        p_mapping[param] = to_unpadded_tensor(p_mapping[param])
                    destination[prefix + name] = p_mapping[param]
        del p_mapping
        del param_to_save_data

        # save all buffers
        for name, buf in self.named_buffers():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf if keep_vars else buf.detach()
        # save extra states
        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if (
            getattr(self.__class__, "get_extra_state", torch.nn.Module.get_extra_state)
            is not torch.nn.Module.get_extra_state
        ):
            destination[extra_state_key] = self.get_extra_state()

    def load_state_dict(self, state_dict: "OrderedDict[str, torch.Tensor]", strict: bool = True):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys

        Note:
            If a parameter or buffer is registered as ``None`` and its corresponding key
            exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
            ``RuntimeError``.
        """
        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata  # type: ignore[attr-defined]

        prefix = ""
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        self._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0,
                    "Unexpected key(s) in state_dict: {}. ".format(
                        ", ".join('"{}"'.format(k) for k in unexpected_keys)
                    ),
                )
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, "Missing key(s) in state_dict: {}. ".format(", ".join('"{}"'.format(k) for k in missing_keys))
                )

        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(self.__class__.__name__, "\n\t".join(error_msgs))
            )
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
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

        persistent_buffers = {k: v for k, v in self.named_buffers() if k not in self._non_persistent_buffers_set}
        local_name_params = itertools.chain(self.named_parameters(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        def load(
            param_name,
            dest_tensor,
            copy_func,
            source_device_mesh=None,
            source_sharding_spec=None,
            shard_fn=None,
            gather_fn=None,
        ):
            state_key = prefix + param_name
            if state_key in state_dict:
                input_param = state_dict[state_key]

                global_shape = dest_tensor.shape
                if source_device_mesh is not None and source_sharding_spec is not None:
                    global_shape = get_global_shape(dest_tensor)

                if is_padded_tensor(dest_tensor):
                    padding_dim = dest_tensor._padding_dim
                    input_param = to_padded_tensor(input_param, global_shape[padding_dim], padding_dim)

                if source_device_mesh is not None and source_sharding_spec is not None:
                    input_param = distribute_tensor(input_param, source_device_mesh, source_sharding_spec)
                elif shard_fn is not None and gather_fn is not None:
                    input_param = distribute_tensor_with_customization(
                        input_param, shard_fn=shard_fn, gather_fn=gather_fn
                    )

                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if len(dest_tensor.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]
                if input_param.shape != dest_tensor.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append(
                        "size mismatch for {}: copying a param with shape {} from checkpoint, "
                        "the shape in current model is {}.".format(state_key, input_param.shape, dest_tensor.shape)
                    )
                    return
                try:
                    with torch.no_grad():
                        copy_func(input_param)
                except Exception as ex:
                    error_msgs.append(
                        'While copying the parameter named "{}", '
                        "whose dimensions in the model are {} and "
                        "whose dimensions in the checkpoint are {}, "
                        "an exception occurred : {}.".format(state_key, dest_tensor.size(), input_param.size(), ex.args)
                    )
            elif strict:
                missing_keys.append(state_key)

        def load_parameter(chunk_slice, data):
            chunk_slice.copy_(data.flatten())

        for name, param in self.named_parameters():
            if is_ddp_ignored(param):
                # deal with ddp ignored parameters
                load(name, param, param.copy_)

        fp32_to_name = dict()
        for p, fp32_p in zip(self.fp16_params, self.fp32_params):
            if p is not None:
                name = self.param2name[p]
                fp32_to_name[fp32_p] = name

        params_to_load = self.fp32_params if self.chunk_manager.reuse_fp16_chunk else self.fp16_params
        chunk_list = self.chunk_manager.get_chunks(params_to_load)
        for chunk in chunk_list:
            temp_chunk = get_temp_total_chunk_on_cuda(chunk, self.mixed_precision)

            for tensor, tensor_info in chunk.tensors_info.items():
                source_device_mesh, source_sharding_spec, shard_fn, gather_fn = None, None, None, None
                if is_distributed_tensor(tensor):
                    # shard the input param
                    source_device_mesh = get_device_mesh(tensor)
                    source_sharding_spec = get_sharding_spec(tensor)
                elif is_customized_distributed_tensor(tensor):
                    shard_fn = tensor.shard_fn
                    gather_fn = tensor.gather_fn

                parameter_name = (
                    fp32_to_name[tensor] if self.chunk_manager.reuse_fp16_chunk else self.param2name[tensor]
                )
                parameter_slice = temp_chunk[tensor_info.offset : tensor_info.end]
                load(
                    parameter_name,
                    tensor,
                    partial(load_parameter, parameter_slice),
                    source_device_mesh,
                    source_sharding_spec,
                    shard_fn,
                    gather_fn,
                )

            if chunk.is_gathered:
                chunk.cuda_global_chunk.copy_(temp_chunk)
            elif chunk.cuda_shard is not None:
                chunk.cuda_shard.copy_(temp_chunk[chunk.shard_begin : chunk.shard_end])
            else:
                chunk.cpu_shard.copy_(temp_chunk[chunk.shard_begin : chunk.shard_end])

            del temp_chunk

        # sync running weights and master weights
        if self.master_weights:
            for loaded_chunk in chunk_list:
                paired_chunk = loaded_chunk.paired_chunk
                assert paired_chunk is not None
                paired_chunk.payload.copy_(loaded_chunk.payload)

        for name, buf in persistent_buffers.items():
            if buf is not None:
                load(name, buf, buf.copy_)

        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if (
            getattr(self.__class__, "set_extra_state", torch.nn.Module.set_extra_state)
            is not torch.nn.Module.set_extra_state
        ):
            if extra_state_key in state_dict:
                self.set_extra_state(state_dict[extra_state_key])
            elif strict:
                missing_keys.append(extra_state_key)
        elif strict and (extra_state_key in state_dict):
            unexpected_keys.append(extra_state_key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix) and key != extra_state_key:
                    input_name = key[len(prefix) :]
                    if input_name not in local_state:
                        unexpected_keys.append(key)

    def _init_chunks(self, param_order, strict_ddp_mode: bool, cpu_offload: bool, pin_memory: bool):
        zero_world_size = dist.get_world_size(self.zero_group)
        for p in param_order.generate():
            self._preprocess_param(p)
            assert type(p) is ColoParameter

            # ignore the parameters with no gradient
            if not p.requires_grad:
                self.set_params_to_ignore([p])

            # move ignored parameters to CUDA
            if is_ddp_ignored(p):
                p.data = p.data.to(device=get_accelerator().get_current_device(), dtype=self.mixed_precision)
                continue

            # create a fp16 parameter
            p.data = p.data.to(self.mixed_precision)
            # register the fp16 parameter
            self.chunk_manager.register_tensor(
                tensor=p,
                group_type="fp16_param",
                config_key=zero_world_size,
                zero_group=self.zero_group,
                extra_dp_group=self.extra_dp_group,
                cpu_offload=cpu_offload,
                pin_memory=pin_memory,
            )
            self.fp16_params.append(p)

            if self.master_weights:
                # create a fp32 parameter
                fp32_p = p.clone()
                fp32_p.data = fp32_p.data.float()
                self.chunk_manager.register_tensor(
                    tensor=fp32_p,
                    group_type="fp32_param",
                    config_key=zero_world_size,
                    zero_group=self.zero_group,
                    extra_dp_group=self.extra_dp_group,
                    cpu_offload=cpu_offload,
                    pin_memory=pin_memory,
                )
                self.fp32_params.append(fp32_p)

        self.chunk_manager.close_all_groups()

        self.gemini_manager.setup_grads_device(self.fp16_params, self.grads_device)

        # move master weights to corresponding device and setup paired chunks
        # if no master weights, fp32_params should be empty and this loop will be skipped
        for p, fp32_p in zip(self.fp16_params, self.fp32_params):
            chunk_16 = self.chunk_manager.get_chunk(p)
            chunk_32 = self.chunk_manager.get_chunk(fp32_p)
            chunk_32.init_pair(chunk_16)
            if chunk_32.device_type != self.grads_device[p].type:
                self.chunk_manager.move_chunk(chunk_32, self.grads_device[p])

    def _cast_buffers(self):
        for buffer in self.module.buffers():
            if isinstance(buffer, LazyTensor):
                buffer.materialize()
        for buffer in self.module.buffers():
            buffer.data = buffer.to(get_accelerator().get_current_device())
            if torch.is_floating_point(buffer):
                buffer.data = buffer.to(self.mixed_precision)

    def _preprocess_param(self, p: Union[nn.Parameter, ColoParameter, "LazyTensor"]) -> None:
        """Convert parameter to ColoParameter in-place.
        Args:
            p (Union[nn.Parameter, ColoParameter, LazyTensor]): parameter to be converted
        """
        if type(p) is ColoParameter:
            # model is initialized with ColoInitContext
            return
        requires_grad = p.requires_grad
        if isinstance(p, LazyTensor):
            # model is initialized with LazyInitContext
            p.materialize()
        p.__class__ = ColoParameter
        p.__init__(p, requires_grad=requires_grad)

    def state_dict_shard(
        self,
        prefix: str = "",
        keep_vars: bool = False,
        max_shard_size: int = 1024,
        only_rank_0: bool = True,
    ) -> Iterator[Tuple[OrderedDict, int]]:
        """Returns dictionaries containing a whole state of the module one by one. The max size of dictionary shard is specified by ``max_shard_size``.

        Both parameters and persistent buffers (e.g. running averages) are included.
        Keys are corresponding parameter and buffer names.
        Parameters and buffers set to ``None`` are not included.

        Args:
            prefix (str, optional): the prefix for parameters and buffers used in this
                module. Defaults to ''.
            keep_vars (bool, optional): whether to keep variables. Defaults to False.
            max_shard_size (int, optional): max size of state dict shard (in MB). Defaults to 1024.
            only_rank_0 (bool, optional): only get data on rank0. Defaults to True.


        Yields:
            Iterator[OrderedDict]: A generator of state dict shard
        """
        sharder = StateDictSharder(max_shard_size)

        # get the mapping between copies and fp16 parameters
        fp16_to_fp32 = dict()
        for p, fp32_p in zip(self.fp16_params, self.fp32_params):
            fp16_to_fp32[p] = fp32_p

        # key is fp32 param, and value is gathered param on CPU
        gathered_param_buffer = dict()
        for name, param in self.name2param.items():
            if param is not None:
                if is_ddp_ignored(param):
                    # deal with ddp ignored parameters
                    gathered_param = param if keep_vars else param.detach()
                else:
                    # as memory of fp16 param may be reused, we should use fp32 param and then convert to fp16
                    param_to_save = fp16_to_fp32[param] if self.chunk_manager.reuse_fp16_chunk else param
                    if param_to_save not in gathered_param_buffer:
                        chunk = self.chunk_manager.get_chunk(param_to_save)
                        gathered_param_buffer.update(self._get_chunk_to_save_data(chunk, only_rank_0))
                    gathered_param = gathered_param_buffer.pop(param_to_save)

                block, block_size = sharder.append_param(prefix + name, gathered_param)
                if block is not None:
                    yield block, block_size

        del fp16_to_fp32
        del gathered_param_buffer

        # save all buffers
        for name, buf in self.named_buffers():
            if buf is not None and name not in self._non_persistent_buffers_set:
                buffer = buf if keep_vars else buf.detach()
                block, block_size = sharder.append_param(prefix + name, buffer)
                if block is not None:
                    yield block, block_size
        # save extra states
        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if (
            getattr(self.__class__, "get_extra_state", torch.nn.Module.get_extra_state)
            is not torch.nn.Module.get_extra_state
        ):
            extra_state = self.get_extra_state()
            block, block_size = sharder.append_param(extra_state_key, extra_state)
            if block is not None:
                yield block, block_size

        yield sharder.current_block, sharder.current_block_size
