import itertools
from collections import OrderedDict
from contextlib import nullcontext
from functools import partial
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from colossalai.checkpoint_io.utils import calculate_tensor_size
from colossalai.lazy import LazyTensor
from colossalai.logging import get_dist_logger
from colossalai.nn.parallel.data_parallel import ColoDDP, _cast_float, free_storage
from colossalai.tensor import ProcessGroup as ColoProcessGroup
from colossalai.tensor import ReplicaSpec
from colossalai.tensor.colo_parameter import ColoParameter, ColoTensor, ColoTensorSpec
from colossalai.tensor.param_op_hook import ColoParamOpHookManager
from colossalai.utils import get_current_device, is_ddp_ignored

from .chunk import Chunk, ChunkManager, TensorState, init_chunk_manager
from .gemini_hook import GeminiZeROHook
from .gemini_mgr import GeminiManager
from .memory_tracer import MemStats, OrderedParamGenerator
from .utils import get_temp_total_chunk_on_cuda

try:
    from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX, _IncompatibleKeys
except ImportError:
    _EXTRA_STATE_KEY_SUFFIX = '_extra_state'

__all__ = [
    'ZeroDDP',
    'GeminiDDP',
]


class ZeroDDP(ColoDDP):
    """ZeRO DDP for ColoTensor.
    Warning: Nested ZeroDDP is not supported now.
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

    def __init__(self,
                 module: torch.nn.Module,
                 gemini_manager: GeminiManager,
                 pin_memory: bool = False,
                 force_outputs_fp32: bool = False,
                 strict_ddp_mode: bool = False,
                 scatter_after_inference: bool = True,
                 mixed_precision: torch.dtype = torch.float16) -> None:
        assert mixed_precision in (torch.float16, torch.bfloat16)
        self.gemini_manager = gemini_manager
        self.chunk_manager: ChunkManager = gemini_manager.chunk_manager
        self.force_outputs_fp32 = force_outputs_fp32
        self.param_op_hook = GeminiZeROHook(gemini_manager)
        self.fp32_params: List[ColoTensor] = list()
        self.fp16_params: List[ColoParameter] = list()
        self.overflow_counter = 0
        self.grads_device: Dict[torch.Tensor, torch.device] = dict()
        self.param2name: Dict[nn.Parameter, str] = dict()
        self.name2param: Dict[str, nn.Parameter] = dict()
        self.scatter_after_inference = scatter_after_inference
        self.mixed_precision = mixed_precision

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

        self._init_chunks(param_order=param_order,
                          strict_ddp_mode=strict_ddp_mode,
                          cpu_offload=self.gemini_manager.policy_name != 'cuda',
                          pin_memory=pin_memory)

        for name, param in module.named_parameters():
            self.param2name[param] = name
        for m_name, m_var in module.named_modules():
            for p_name, p_var in m_var.named_parameters(recurse=False):
                param_name = m_name + '.' + p_name if m_name else p_name
                self.name2param[param_name] = p_var
        super().__init__(module, process_group=ColoProcessGroup())
        self._non_persistent_buffers_set = self._get_non_persistent_buffers_set(module)
        self._cast_buffers()

    def _get_non_persistent_buffers_set(self,
                                        module,
                                        memo: Optional[Set[nn.Module]] = None,
                                        prefix: str = '',
                                        remove_duplicate: bool = True):
        r"""
        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not
        """

        if memo is None:
            memo = set()
        self_non_persistent_set = set()
        if module not in memo:
            if remove_duplicate:
                memo.add(module)
            self_non_persistent_set = set(
                map(lambda key: prefix + ('.' if prefix else '') + key, module._non_persistent_buffers_set))
            for name, sub_module in module._modules.items():
                if sub_module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                child_non_persistent_set = self._get_non_persistent_buffers_set(sub_module, memo, submodule_prefix,
                                                                                remove_duplicate)
                self_non_persistent_set = set.union(self_non_persistent_set, child_non_persistent_set)
        return self_non_persistent_set

    def _post_forward(self):
        """This function is only triggered for inference.
        """
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
            assert not self.gemini_manager.need_warmup or not self.gemini_manager.is_warmup(
            ), "You should run a completed iteration as your warmup iter"

        args, kwargs = _cast_float(args, self.mixed_precision), _cast_float(kwargs, self.mixed_precision)
        self.module.zero_grad(set_to_none=True)
        if not grad_flag:
            outputs = self._inference_forward(*args, **kwargs)
        else:
            self.gemini_manager.pre_iter(*args)
            with ColoParamOpHookManager.use_hooks(self.param_op_hook):
                outputs = self.module(*args, **kwargs)

        if self.force_outputs_fp32:
            return _cast_float(outputs, torch.float)
        return outputs

    def _inference_forward(self, *args, **kwargs):
        """This function is only triggered for inference.
        """
        fwd_ctx = ColoParamOpHookManager.use_hooks(self.param_op_hook)
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
        if self.chunk_manager.accessed_mem != 0:
            error_params = ["Reduction failed at followed parameters:"]
            for param in self.param2name:
                if not is_ddp_ignored(param) and not getattr(param, "_gemini_reduced"):
                    error_params.append(self.param2name[param])
            error_str = "\n\t".join(error_params)
            raise RuntimeError("ZERO DDP error: the synchronization of gradients doesn't exit properly.",
                               "The most possible reason is that the model is not compatible with ZeroDDP.\n",
                               f"{error_str}")
        self._setup_grads_ptr()
        self._logger.debug(
            f'comp cuda demand time: {self.gemini_manager._comp_cuda_demand_time}, layout time: {self.gemini_manager._layout_time}, evict time: {self.gemini_manager._evict_time}, CPU->CUDA vol: {self.gemini_manager._h2d_volume}B, CUDA->CPU vol: {self.gemini_manager._d2h_volume}'
        )
        self.gemini_manager.post_iter()

    def backward(self, loss: torch.Tensor):
        self._pre_backward()
        with self.param_op_hook.switch_to_backward(), ColoParamOpHookManager.use_hooks(self.param_op_hook):
            loss.backward()
        self._post_backward()

    def backward_by_grad(self, tensor, grad):
        with self.param_op_hook.switch_to_backward(), ColoParamOpHookManager.use_hooks(self.param_op_hook):
            torch.autograd.backward(tensor, grad)
        self._post_backward()

    def grad_handle(self, p, grad):
        empty_grad = torch.empty_like(grad)
        free_storage(empty_grad)
        with torch._C.DisableTorchFunction():
            chunk = self.chunk_manager.get_chunk(p)
            if chunk.tensors_info[p].state != TensorState.HOLD_AFTER_BWD:
                raise RuntimeError(f"Parameter `{self.param2name[p]}` failed at the gradient reduction. "
                                   "Some unsupported torch function is operated upon this parameter.")
            self.chunk_manager.trans_tensor_state(p, TensorState.READY_FOR_REDUCE)
            chunk.copy_tensor_to_chunk_slice(p, grad)
            reduced = self.chunk_manager.reduce_chunk(chunk)
            if reduced:
                if chunk.is_gathered:
                    chunk.cuda_global_chunk.div_(chunk.pg_size)
                else:
                    chunk.cuda_shard.div_(chunk.pg_size)
                # check overflow elements
                self.overflow_counter += chunk.has_inf_or_nan
                # record l2 norm for gradient clipping
                if chunk.l2_norm_flag:
                    chunk.set_l2_norm()
                self.chunk_manager.move_chunk(chunk, self.grads_device[p], force_copy=True)
        return empty_grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.module.zero_grad(set_to_none=True)

    def set_chunk_grad_device(self, chunk: Chunk, device: torch.device) -> None:
        for tensor in chunk.get_tensors():
            self.grads_device[tensor] = device

    def state_dict(self,
                   destination=None,
                   prefix='',
                   keep_vars=False,
                   only_rank_0: bool = True,
                   dtype: torch.dtype = torch.float16):
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
        self._save_to_state_dict(destination, prefix, keep_vars, only_rank_0, dtype)

        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination

    def _get_chunk_to_save_data(self, chunk: Chunk, only_rank_0: bool, dtype: torch.dtype = torch.float16) -> Dict:
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
        temp_chunk = get_temp_total_chunk_on_cuda(chunk)
        if torch.is_floating_point(temp_chunk):
            temp_chunk = temp_chunk.to(dtype)
        for tensor, tensor_info in chunk.tensors_info.items():
            record_tensor = torch.empty([0])
            record_flag = (not only_rank_0) | (dist.get_rank(chunk.torch_pg) == 0)
            if record_flag:
                record_tensor = temp_chunk[tensor_info.offset:tensor_info.end].view(tensor.shape).cpu()

            assert tensor not in chunk_to_save_data
            chunk_to_save_data[tensor] = record_tensor

        del temp_chunk
        return chunk_to_save_data

    def _get_param_to_save_data(self, param_list: List[torch.nn.Parameter], only_rank_0: bool,
                                dtype: torch.dtype) -> Dict:
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
            param_to_save_data.update(self._get_chunk_to_save_data(chunk, only_rank_0, dtype))
        return param_to_save_data

    def _save_to_state_dict(self, destination, prefix, keep_vars, only_rank_0=True, dtype=torch.float16):
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
        param_to_save_data = self._get_param_to_save_data(self.fp32_params, only_rank_0, dtype)
        # get the mapping between copies and fp16 parameters
        p_mapping = dict()
        for p, fp32_p in zip(self.fp16_params, self.fp32_params):
            name = self.param2name[p]
            assert fp32_p in param_to_save_data, "Parameter '{}' is neglected in the chunk list".format(name)
            record_parameter = param_to_save_data[fp32_p]
            p_mapping[p] = record_parameter
        for name, param in self.name2param.items():
            if param is not None:
                if is_ddp_ignored(param):
                    # deal with ddp ignored parameters
                    destination[prefix + name] = param if keep_vars else param.detach()
                else:
                    destination[prefix + name] = p_mapping[param]
        del p_mapping
        del param_to_save_data

        # save all buffers
        for name, buf in self.named_buffers():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf if keep_vars else buf.detach()
        # save extra states
        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if getattr(self.__class__, "get_extra_state",
                   torch.nn.Module.get_extra_state) is not torch.nn.Module.get_extra_state:
            destination[extra_state_key] = self.get_extra_state()

    def load_state_dict(self, state_dict: 'OrderedDict[str, torch.Tensor]', strict: bool = True):
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
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata    # type: ignore[attr-defined]

        prefix = ''
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        self._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(', '.join(
                        '"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                self.__class__.__name__, "\n\t".join(error_msgs)))
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
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

        def load(param_name, dest_tensor, copy_func):
            state_key = prefix + param_name
            if state_key in state_dict:
                input_param = state_dict[state_key]
                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if len(dest_tensor.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]
                if input_param.shape != dest_tensor.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                      'the shape in current model is {}.'.format(state_key, input_param.shape,
                                                                                 dest_tensor.shape))
                    return
                try:
                    with torch.no_grad():
                        copy_func(input_param)
                except Exception as ex:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}, '
                                      'an exception occurred : {}.'.format(state_key, dest_tensor.size(),
                                                                           input_param.size(), ex.args))
            elif strict:
                missing_keys.append(state_key)

        def load_fp32_parameter(chunk_slice, data):
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

        chunk_list = self.chunk_manager.get_chunks(self.fp32_params)
        for chunk in chunk_list:
            temp_chunk = get_temp_total_chunk_on_cuda(chunk)

            for tensor, tensor_info in chunk.tensors_info.items():
                parameter_name = fp32_to_name[tensor]
                parameter_slice = temp_chunk[tensor_info.offset:tensor_info.end]
                load(parameter_name, tensor, partial(load_fp32_parameter, parameter_slice))

            if chunk.is_gathered:
                chunk.cuda_global_chunk.copy_(temp_chunk)
            elif chunk.cuda_shard is not None:
                chunk.cuda_shard.copy_(temp_chunk[chunk.shard_begin:chunk.shard_end])
            else:
                chunk.cpu_shard.copy_(temp_chunk[chunk.shard_begin:chunk.shard_end])

            del temp_chunk

        for chunk_32 in chunk_list:
            chunk_16 = chunk_32.paired_chunk
            assert chunk_16 is not None
            chunk_16.optim_update()

        for name, buf in persistent_buffers.items():
            if buf is not None:
                load(name, buf, buf.copy_)

        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if getattr(self.__class__, "set_extra_state",
                   torch.nn.Module.set_extra_state) is not torch.nn.Module.set_extra_state:
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
                    if input_name not in local_state:
                        unexpected_keys.append(key)

    def _init_chunks(self, param_order, strict_ddp_mode: bool, cpu_offload: bool, pin_memory: bool):
        ddp_pg = ColoProcessGroup()
        for p in param_order.generate():
            self._preprocess_param(p)
            assert type(p) is ColoParameter

            # gather sharded parameters in the strict ddp mode
            if strict_ddp_mode:
                if not p.is_replicate():
                    p.set_dist_spec(ReplicaSpec())
                p.set_process_group(pg=ddp_pg)

            # ignore the parameters with no gradient
            if not p.requires_grad:
                self.set_params_to_ignore([p])

            # move ignored parameters to CUDA
            if is_ddp_ignored(p):
                p.data = p.data.to(device=get_current_device(), dtype=self.mixed_precision)
                continue

            # create a fp32 parameter
            fp32_data = p.data.float()
            fp32_p = ColoTensor(fp32_data, spec=ColoTensorSpec(p.process_group))
            # create a fp16 parameter
            p.data = p.data.to(self.mixed_precision)

            # register the fp16 parameter and fp32 parameter in the chunk manager
            dp_world_size = p.process_group.dp_world_size()
            self.chunk_manager.register_tensor(tensor=p,
                                               group_type='fp16_param',
                                               config_key=dp_world_size,
                                               cpu_offload=cpu_offload,
                                               pin_memory=pin_memory)
            self.chunk_manager.register_tensor(tensor=fp32_p,
                                               group_type='fp32_param',
                                               config_key=dp_world_size,
                                               cpu_offload=cpu_offload,
                                               pin_memory=pin_memory)

            self.fp16_params.append(p)
            self.fp32_params.append(fp32_p)
            self.grads_device[p] = self.gemini_manager.default_device

        self.chunk_manager.close_all_groups()

        for p, fp32_p in zip(self.fp16_params, self.fp32_params):
            chunk_16 = self.chunk_manager.get_chunk(p)
            chunk_32 = self.chunk_manager.get_chunk(fp32_p)
            chunk_32.init_pair(chunk_16)

            # keep gathered chunks are in CUDA
            if chunk_16.keep_gathered:
                self.grads_device[p] = get_current_device()

    def _cast_buffers(self):
        for buffer in self.module.buffers():
            if isinstance(buffer, LazyTensor):
                buffer.materialize()
            buffer.data = buffer.cuda()
            if torch.is_floating_point(buffer):
                buffer.data = buffer.to(self.mixed_precision)

    def _preprocess_param(self, p: Union[nn.Parameter, ColoParameter, 'LazyTensor']) -> None:
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

    def state_dict_shard(self,
                         prefix: str = '',
                         keep_vars: bool = False,
                         max_shard_size: int = 1024,
                         only_rank_0: bool = True,
                         dtype: torch.dtype = torch.float16) -> Iterator[Tuple[OrderedDict, int]]:
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
        sharder = _StateDictSharder(max_shard_size)

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
                    fp32_param = fp16_to_fp32[param]
                    if fp32_param not in gathered_param_buffer:
                        chunk = self.chunk_manager.get_chunk(fp32_param)
                        gathered_param_buffer.update(self._get_chunk_to_save_data(chunk, only_rank_0, dtype))
                    gathered_param = gathered_param_buffer.pop(fp32_param)

                block, block_size = sharder.append(prefix + name, gathered_param)
                if block is not None:
                    yield block, block_size

        del fp16_to_fp32
        del gathered_param_buffer

        # save all buffers
        for name, buf in self.named_buffers():
            if buf is not None and name not in self._non_persistent_buffers_set:
                buffer = buf if keep_vars else buf.detach()
                block, block_size = sharder.append(prefix + name, buffer)
                if block is not None:
                    yield block, block_size
        # save extra states
        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if getattr(self.__class__, "get_extra_state",
                   torch.nn.Module.get_extra_state) is not torch.nn.Module.get_extra_state:
            extra_state = self.get_extra_state()
            block, block_size = sharder.append(extra_state_key, extra_state)
            if block is not None:
                yield block, block_size

        yield sharder.current_block, sharder.current_block_size


class _StateDictSharder:

    def __init__(self, max_shard_size: int) -> None:
        self.max_shard_size = max_shard_size
        self.current_block = OrderedDict()
        self.current_block_size = 0

    def append(self, name: str, tensor: torch.Tensor) -> Tuple[Optional[OrderedDict], int]:
        tensor_size = calculate_tensor_size(tensor)
        ret_block = None
        ret_block_size = 0

        # before we return the current block and create a new block,
        # we need to ensure that the current block is not empty
        if self.current_block_size + tensor_size > self.max_shard_size and self.current_block_size > 0:
            ret_block = self.current_block
            ret_block_size = self.current_block_size
            self.current_block = OrderedDict()
            self.current_block_size = 0
        self.current_block[name] = tensor
        self.current_block_size += tensor_size
        return ret_block, ret_block_size


class GeminiDDP(ZeroDDP):

    def __init__(self,
                 module: torch.nn.Module,
                 device: torch.device,
                 placement_policy: str = "cpu",
                 pin_memory: bool = False,
                 force_outputs_fp32: bool = False,
                 strict_ddp_mode: bool = False,
                 scatter_after_inference: bool = True,
                 search_range_m: int = 32,
                 hidden_dim: Optional[int] = None,
                 min_chunk_size_m: float = 32,
                 memstats: Optional[MemStats] = None,
                 mixed_precision: torch.dtype = torch.float16,
                 verbose: bool = False) -> None:
        """
        A torch.Module wrapper using ZeRO-DP and Gemini.
        ZeRO is for parallel. Gemini is for memory management.
        WARNING: The class will modify the module inline!

        Example:
            model is initialized under the context of ColoInitContext
            >>> model = GeminiDDP(model, torch.cuda.current_device(), "cuda")
            >>> logits = model(x)
            >>> loss = criterion(logits, labels)
            >>> model.backward(loss)

        Args:
            module (torch.nn.Module): the model to be wrapped.
            device (torch.device): device to place the model.
            placement_policy (str, optional): "cpu", "cuda", "auto". Defaults to "cpu".
            pin_memory (bool, optional): use pin memory on CPU. Defaults to False.
            force_outputs_fp32 (bool, optional): force outputs are fp32. Defaults to False.
            search_range_m (int, optional): chunk size searching range divided by 2^20. Defaults to 32.
            hidden_dim (int, optional): the hidden dimension of DNN.
                Users can provide this argument to speed up searching.
                If users do not know this argument before training, it is ok. We will use a default value 1024.
            min_chunk_size_m (float, optional): the minimum chunk size divided by 2^20.
                If the aggregate size of parameters is still smaller than the minimum chunk size,
                all parameters will be compacted into one small chunk.
            memstats (MemStats, optional) the memory statistics collector by a runtime memory tracer.
        """
        # some ugly hotfix for the compatibility with Lightning
        if search_range_m is None:
            search_range_m = 32

        chunk_manager = init_chunk_manager(model=module,
                                           init_device=device,
                                           hidden_dim=hidden_dim,
                                           search_range_m=search_range_m,
                                           min_chunk_size_m=min_chunk_size_m,
                                           strict_ddp_flag=strict_ddp_mode,
                                           verbose=verbose)
        gemini_manager = GeminiManager(placement_policy, chunk_manager, memstats)
        super().__init__(module,
                         gemini_manager,
                         pin_memory,
                         force_outputs_fp32,
                         strict_ddp_mode,
                         scatter_after_inference,
                         mixed_precision=mixed_precision)
