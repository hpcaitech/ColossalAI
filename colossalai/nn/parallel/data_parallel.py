import itertools
from collections import OrderedDict
from functools import partial
from typing import Dict, Iterable, List, Optional, Set

import torch
import torch.distributed as dist

from colossalai.gemini.chunk import Chunk, ChunkManager, TensorState
from colossalai.gemini.gemini_mgr import GeminiManager
from colossalai.gemini.memory_tracer import OrderedParamGenerator
from colossalai.logging import get_dist_logger
from colossalai.nn.parallel.utils import get_temp_total_chunk_on_cuda
from colossalai.tensor import ProcessGroup as ColoProcessGroup
from colossalai.tensor import ReplicaSpec
from colossalai.tensor.colo_parameter import ColoParameter, ColoTensor, ColoTensorSpec
from colossalai.tensor.param_op_hook import ColoParamOpHookManager
from colossalai.utils import get_current_device, is_ddp_ignored
from colossalai.zero.utils.gemini_hook import GeminiZeROHook

from .reducer import Reducer
from .utils import get_static_torch_model

try:
    from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX, _IncompatibleKeys
except ImportError:
    _EXTRA_STATE_KEY_SUFFIX = '_extra_state'


def free_storage(data: torch.Tensor) -> None:
    """Free underlying storage of a Tensor."""
    if data.storage().size() > 0:
        # Since we're modifying the Tensor's Storage directly, make sure the Tensor
        # is the sole occupant of the Storage.
        assert data.storage_offset() == 0
        data.storage().resize_(0)


def _cast_float(args, dtype: torch.dtype):
    if isinstance(args, torch.Tensor) and torch.is_floating_point(args):
        args = args.to(dtype)
    elif isinstance(args, (list, tuple)):
        args = type(args)(_cast_float(t, dtype) for t in args)
    elif isinstance(args, dict):
        args = {k: _cast_float(v, dtype) for k, v in args.items()}
    return args


class ColoDDP(torch.nn.Module):
    """Distributed data parallel for ColoTensor. Nested ColoDDP is not supported now.

    Example:
        >>> from colossalai.core import global_context as gpc
        >>> from colossalai.context import ParallelMode
        >>> model = torch.nn.Linear(20, 1)
        >>> pg = ProcessGroup(tp_degree = world_size//2)
        >>> model = ColoDDP(model, pg)
        >>> logits = model(x)
        >>> loss = criterion(logits, labels)
        >>> model.backward(loss)

    Args:
        module (torch.nn.Module): Module to apply DDP.
        process_group (Optional[dist.ProcessGroup], optional): The process group which DDP uses.
            If it's None, the default data parallel group will be used. Defaults to None.
    """

    def __init__(self,
                 module: torch.nn.Module,
                 process_group: ColoProcessGroup,
                 bucket_cap_mb: int = 25,
                 rebuild_bucket: bool = True) -> None:
        assert not isinstance(module, ColoDDP)
        super().__init__()
        self.module = module
        self.comm_stream: torch.cuda.Stream = torch.cuda.Stream()
        assert process_group

        self.process_group = process_group
        self.dp_world_size = self.process_group.dp_world_size()

        self.reducer = Reducer(bucket_cap_mb)
        self.rebuild_bucket = rebuild_bucket
        for p in module.parameters():
            if is_ddp_ignored(p):
                continue
            if p.requires_grad:
                p.register_hook(partial(self.grad_handle, p))

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.module.named_parameters(prefix, recurse)

    def named_buffers(self, prefix: str = '', recurse: bool = True):
        return self.module.named_buffers(prefix, recurse)

    def named_children(self):
        return self.module.named_children()

    def named_modules(self,
                      memo: Optional[Set[torch.nn.Module]] = None,
                      prefix: str = '',
                      remove_duplicate: bool = True):
        return self.module.named_modules(memo, prefix, remove_duplicate)

    def forward(self, *args, **kwargs):
        self.module.zero_grad(set_to_none=True)
        return self.module(*args, **kwargs)

    def backward(self, loss: torch.Tensor):
        loss.backward()
        with torch.cuda.stream(self.comm_stream):
            self.reducer.flush()
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        if self.rebuild_bucket:
            self.reducer.free()
        for p in self.module.parameters():
            if is_ddp_ignored(p):
                continue
            if p.grad.device.type != "cpu":
                p.grad = p._saved_grad

    def grad_handle(self, p, grad):
        if grad.device.type != "cpu":
            empty_grad = torch.empty_like(grad)
            free_storage(empty_grad)
            if self.dp_world_size > 1:
                grad = grad / self.dp_world_size
                self.comm_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.comm_stream):
                    self.reducer.all_reduce_async(grad,
                                                  group=self.process_group.dp_process_group(),
                                                  callback_fn=partial(self._save_grad, p))
                grad.record_stream(self.comm_stream)
            else:
                ColoDDP._save_grad(p, grad)
            return empty_grad

        else:
            # TODO(jiaruifang) fixme
            self.process_group.set_cpu_groups()
            dist.all_reduce(grad, group=self.process_group.cpu_dp_process_group())
            return grad

    @staticmethod
    def _save_grad(p, grad):
        if hasattr(p, '_saved_grad'):
            p._saved_grad.add_(grad)
        else:
            p._saved_grad = grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.module.zero_grad(set_to_none=True)
        for p in self.module.parameters():
            if getattr(p, '_saved_grad', None) is not None:
                if set_to_none:
                    p._saved_grad = None
                else:
                    if p._saved_grad.grad_fn is not None:
                        p._saved_grad.detach_()
                    else:
                        p._saved_grad.requires_grad_(False)
                    p._saved_grad.zero_()

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

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict: 'OrderedDict[str, torch.Tensor]', strict: bool = True):
        return self.module.load_state_dict(state_dict, strict)


class ZeroDDP(ColoDDP):
    """ZeRO DDP for ColoTensor.
    Warning: Nested ZeroDDP is not supported now.
    It is designed to be used with ChunkManager and GeminiManager.
    For more details, see the API reference of ``ChunkManager`` and ``GeminiManager``.

    Args:
        module (torch.nn.Module): Module to apply ZeRO-DP.
        gemini_manager (GeminiManager): Manages the chunk manager and heterogeneous momery space.
            For more details, see the API reference of ``GeminiManager``.
        pin_memory (bool): Chunks on CPU Memory use pin-memory.
        force_outputs_fp32 (bool): If set to True, outputs will be fp32. Otherwise, outputs will be fp16.
            Defaults to False.
        strict_ddp_mode (bool): If set to True, there is no tensor sharding, each tensor is replicated.
            Defaults to False. Users can set it to True, when they clearly know that they only need DDP.
    """

    def __init__(self,
                 module: torch.nn.Module,
                 gemini_manager: GeminiManager,
                 pin_memory: bool = False,
                 force_outputs_fp32: bool = False,
                 strict_ddp_mode: bool = False) -> None:
        super().__init__(module, process_group=ColoProcessGroup())
        self.gemini_manager = gemini_manager
        self.chunk_manager: ChunkManager = gemini_manager.chunk_manager
        self.force_outputs_fp32 = force_outputs_fp32
        self.param_op_hook = GeminiZeROHook(gemini_manager)
        self.fp32_params: List[ColoTensor] = []
        self.overflow_counter = 0
        self.grads_device: Dict[torch.Tensor, torch.device] = {}

        cpu_offload = self.gemini_manager.policy_name != 'cuda'

        if self.gemini_manager._premade_memstats_:
            # build chunk in param runtime visited order.
            param_order = self.gemini_manager.memstats()._param_runtime_order
        else:
            # build chunk in param initialized order.
            # Note: in this way, it can not get filter unused params during runtime.
            param_order = OrderedParamGenerator()
            for p in module.parameters():
                param_order.append(p)

        for p in param_order.generate():
            assert isinstance(p, ColoParameter)

            if strict_ddp_mode and not p.is_replicate():
                p.set_dist_spec(ReplicaSpec())

            if is_ddp_ignored(p):
                p.data = p.data.to(device=get_current_device(), dtype=torch.float16)
                continue

            fp32_data = p.data.float()
            fp32_p = ColoTensor(fp32_data, spec=ColoTensorSpec(p.process_group))
            p.data = p.data.half()
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
            self.fp32_params.append(fp32_p)
            self.grads_device[p] = self.gemini_manager.default_device

        self.chunk_manager.close_all_groups()
        self._cast_buffers()

        params_list = [p for p in param_order.generate() if not is_ddp_ignored(p)]
        for p, fp32_p in zip(params_list, self.fp32_params):
            chunk_16 = self.chunk_manager.get_chunk(p)
            chunk_32 = self.chunk_manager.get_chunk(fp32_p)
            chunk_32.init_pair(chunk_16)

            # keep gathered chunks are in CUDA
            if chunk_16.keep_gathered:
                self.grads_device[p] = get_current_device()

        self._logger = get_dist_logger()

    def _post_forward(self):
        """This function is only triggered for inference.
        """
        access_list = list(self.chunk_manager.accessed_chunks)
        # we need to scatter all accessed chunks and move them to their original places
        for chunk in access_list:
            assert chunk.can_release
            self.chunk_manager.release_chunk(chunk)
            first_param = next(iter(chunk.tensors_info))
            self.chunk_manager.move_chunk(chunk, self.grads_device[first_param])
        assert self.chunk_manager.accessed_mem == 0
        # reset all recorded attributes
        self.gemini_manager.reset_attributes()

    def forward(self, *args, **kwargs):
        # check whether we are in a inference mode
        grad_flag = torch.is_grad_enabled()
        if not grad_flag:
            assert not self.gemini_manager.is_warmup(), "You should run a completed iteration as your warmup iter"

        args, kwargs = _cast_float(args, torch.half), _cast_float(kwargs, torch.half)
        self.module.zero_grad(set_to_none=True)
        self.gemini_manager.pre_iter(*args)
        with ColoParamOpHookManager.use_hooks(self.param_op_hook):
            outputs = self.module(*args, **kwargs)
        # scatter chunks in the inference mode
        if not grad_flag:
            self._post_forward()

        if self.force_outputs_fp32:
            return _cast_float(outputs, torch.float)
        return outputs

    def _setup_grads_ptr(self):
        for p in self.module.parameters():
            if is_ddp_ignored(p):
                continue
            p.grad = None

    def _post_backward(self):
        if self.chunk_manager.accessed_mem != 0:
            raise RuntimeError("ZERO DDP error: the synchronization of gradients doesn't exit properly.",
                               "The most possible reason is that the model is not compatible with ZeroDDP.")
        self._setup_grads_ptr()
        self._logger.debug(
            f'comp cuda demand time: {self.gemini_manager._comp_cuda_demand_time}, layout time: {self.gemini_manager._layout_time}, evict time: {self.gemini_manager._evict_time}, CPU->CUDA vol: {self.gemini_manager._h2d_volume}B, CUDA->CPU vol: {self.gemini_manager._d2h_volume}'
        )
        self.gemini_manager.post_iter()

    def backward(self, loss: torch.Tensor):
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
            assert chunk.tensors_info[p].state == TensorState.HOLD_AFTER_BWD
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

    def state_dict(self, destination=None, prefix='', keep_vars=False, only_rank_0: bool = True, strict: bool = True):
        """
        Args:
            strict (bool): whether to reture the whole model state as the pytorch `Module.state_dict()`

        Returns:
            dict:
                a dictionary containing a whole state of the module

        Example:

            >>> module.state_dict().keys()
            ['bias', 'weight']
        """
        if strict:
            assert keep_vars is False, "`state_dict` with parameter, `keep_vars=True`, is not supported now."
            torch_model = get_static_torch_model(zero_ddp_model=self, only_rank_0=only_rank_0)
            return torch_model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return self._non_strict_state_dict(destination=destination,
                                           prefix=prefix,
                                           keep_vars=keep_vars,
                                           only_rank_0=only_rank_0)

    def _non_strict_state_dict(self, destination=None, prefix='', keep_vars=False, only_rank_0: bool = True):
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
            temp_chunk = get_temp_total_chunk_on_cuda(chunk)

            for tensor, tensor_info in chunk.tensors_info.items():
                record_tensor = torch.empty([0])
                record_flag = (not only_rank_0) | (dist.get_rank(chunk.torch_pg) == 0)
                if record_flag:
                    record_tensor = temp_chunk[tensor_info.offset:tensor_info.end].view(tensor.shape).cpu()

                assert tensor not in param_to_save_data
                param_to_save_data[tensor] = record_tensor

            del temp_chunk
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

        param_to_save_data = self._get_param_to_save_data(self.fp32_params, only_rank_0)
        ddp_param_list = []
        for name, param in self.named_parameters():
            if is_ddp_ignored(param):
                # deal with ddp ignored parameters
                destination[prefix + name] = param if keep_vars else param.detach()
            else:
                ddp_param_list.append((name, param))
        for (name, p), fp32_p in zip(ddp_param_list, self.fp32_params):
            if p is not None:
                assert fp32_p in param_to_save_data, "Parameter '{}' is neglected in the chunk list".format(name)
                record_parameter = param_to_save_data[fp32_p]
                destination[prefix + name] = record_parameter

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

        ddp_param_list = []
        for name, param in self.named_parameters():
            if is_ddp_ignored(param):
                # deal with ddp ignored parameters
                load(name, param, param.copy_)
            else:
                ddp_param_list.append((name, param))

        fp32_to_name = dict()
        for (name, p), fp32_p in zip(ddp_param_list, self.fp32_params):
            if p is not None:
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

    def _cast_buffers(self):
        for buffer in self.module.buffers():
            buffer.data = buffer.cuda()
            if torch.is_floating_point(buffer):
                buffer.data = buffer.half()
