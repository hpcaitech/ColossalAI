import torch
import itertools
import torch.distributed as dist
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
from functools import partial
from colossalai.zero.utils.zero_hook_v2 import ZeROHookV2
from colossalai.tensor.chunk import TensorState, Chunk
from colossalai.tensor.param_op_hook import ParamOpHookManager
from colossalai.gemini.gemini_mgr import GeminiManager
from typing import Dict, Iterable, List
from colossalai.logging import get_dist_logger
from collections import OrderedDict
from colossalai.tensor.colo_parameter import ColoParameter
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

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module
        self.comm_stream: torch.cuda.Stream = torch.cuda.Stream()
        self.dp_world_size = gpc.get_world_size(ParallelMode.DATA)
        for p in module.parameters():
            if getattr(p, '_ddp_to_ignore', False):
                continue
            if p.requires_grad:
                p.register_hook(partial(self.grad_handle, p))

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.module.named_parameters(prefix, recurse)

    def forward(self, *args, **kwargs):
        self.module.zero_grad(set_to_none=True)
        return self.module(*args, **kwargs)

    def backward(self, loss: torch.Tensor):
        loss.backward()
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        for p in self.module.parameters():
            if getattr(p, '_ddp_to_ignore', False):
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
                    group = gpc.get_group(ParallelMode.DATA)
                    dist.all_reduce(grad, group=group)
                    ColoDDP._save_grad(p, grad)
                grad.record_stream(self.comm_stream)
            else:
                ColoDDP._save_grad(p, grad)
            return empty_grad

        else:
            group = gpc.get_cpu_group(ParallelMode.DATA)
            dist.all_reduce(grad, group=group)
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

        Example::
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


class ColoDDPV2(ColoDDP):

    def __init__(self, module: torch.nn.Module, gemini_manager: GeminiManager) -> None:
        super().__init__(module.half())
        self.gemini_manager = gemini_manager
        self.chunk_manager = gemini_manager.chunk_manager
        self.param_op_hook = ZeROHookV2(gemini_manager)
        self.fp32_params: List[ColoParameter] = []
        self.overflow_counter = 0
        self.grads_device: Dict[torch.Tensor, torch.device] = {}
        self.chunk_manager.create_group('fp16_param', force_data_on_cuda=True)
        self.chunk_manager.create_group('fp32_param')
        # TODO: get param order and filter unused params
        for p in module.parameters():
            if getattr(p, '_ddp_to_ignore', False):
                continue
            assert p.dtype == torch.half
            fp32_p = p.float().detach()
            self.chunk_manager.append_tensor(p, 'fp16_param')
            self.chunk_manager.append_tensor(fp32_p, 'fp32_param')
            self.fp32_params.append(fp32_p)
            self.grads_device[p] = self.gemini_manager.default_device
        self._logger = get_dist_logger()

    def forward(self, *args, **kwargs):
        args, kwargs = _cast_float(args, torch.half), _cast_float(kwargs, torch.half)
        self.module.zero_grad(set_to_none=True)
        self.gemini_manager.pre_iter()
        with ParamOpHookManager.use_hooks(self.param_op_hook):
            outputs = self.module(*args, **kwargs)
        self.chunk_manager.exec_lazy_release()
        return _cast_float(outputs, torch.float)

    def _setup_grads_ptr(self):
        for p in self.module.parameters():
            if getattr(p, '_ddp_to_ignore', False):
                continue
            if self.chunk_manager.get_chunk(p).is_empty or not p.requires_grad:
                p.grad = None
            else:
                p.grad = p.data

    def _post_backward(self):
        self.chunk_manager.exec_lazy_release()
        self._setup_grads_ptr()
        self._logger.info(
            f'layout time: {self.gemini_manager._layout_time}, evict time: {self.gemini_manager._evict_time}, PCIE move vol: {self.gemini_manager._cpu_gpu_move_volume}B'
        )
        self.gemini_manager.post_iter()

    def backward(self, loss: torch.Tensor):
        with self.param_op_hook.switch_to_backward(), ParamOpHookManager.use_hooks(self.param_op_hook):
            loss.backward()
        self._post_backward()

    def backward_by_grad(self, tensor, grad):
        with self.param_op_hook.switch_to_backward(), ParamOpHookManager.use_hooks(self.param_op_hook):
            torch.autograd.backward(tensor, grad)
        self._post_backward()

    def grad_handle(self, p, grad):
        empty_grad = torch.empty_like(grad)
        free_storage(empty_grad)
        with torch._C.DisableTorchFunction():
            self.chunk_manager.trans_tensor_state(p, TensorState.READY_FOR_REDUCE)
            if self.dp_world_size > 1:
                grad = grad / self.dp_world_size
            self.chunk_manager.copy_tensor_to_chunk_slice(p, grad)
            chunk = self.chunk_manager.get_chunk(p)
            reduced = self.chunk_manager.reduce_chunk(chunk)
            self.chunk_manager.release_chunk(chunk)
            if reduced and not chunk.is_empty:
                self.overflow_counter += chunk.has_inf_or_nan
                self.chunk_manager.move_chunk(chunk, self.grads_device[p])
        return empty_grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.module.zero_grad(set_to_none=True)

    def _set_chunk_grad_device(self, chunk: Chunk, device: torch.device) -> None:
        for tensor in chunk.get_tensors():
            self.grads_device[tensor] = device

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        r"""Returns a dictionary containing a whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.
        Parameters and buffers set to ``None`` are not included.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        Example::

            >>> module.state_dict().keys()
            ['bias', 'weight']

        """
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]] = local_metadata = dict(version=self._version)
        self._save_to_state_dict(destination, prefix, keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination

    def _save_to_state_dict(self, destination, prefix, keep_vars):
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
        chunks = self.chunk_manager.get_chunks(self.fp32_params)
        for chunk in chunks:
            self.chunk_manager.access_chunk(chunk)
        for (name, p), fp32_p in zip(self.named_parameters(), self.fp32_params):
            if p is not None:
                destination[prefix + name] = fp32_p.clone() if keep_vars else fp32_p.clone().detach()
        for chunk in chunks:
            self.chunk_manager.release_chunk(chunk)
        for name, buf in self.named_buffers():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf if keep_vars else buf.detach()
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

        def load(name, dest_tensor, copy_func):
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if len(dest_tensor.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]
                if input_param.shape != dest_tensor.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                      'the shape in current model is {}.'.format(key, input_param.shape,
                                                                                 dest_tensor.shape))
                    return
                try:
                    with torch.no_grad():
                        # self.chunk_manager.copy_tensor_to_chunk_slice(fp32_p, input_param)
                        copy_func(input_param)
                except Exception as ex:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}, '
                                      'an exception occurred : {}.'.format(key, dest_tensor.size(), input_param.size(),
                                                                           ex.args))
            elif strict:
                missing_keys.append(key)

        def load_fp32_p(fp32_p, data):
            if fp32_p.storage().size() > 0:
                self.chunk_manager.copy_tensor_to_chunk_slice(fp32_p, data)

        for (name, p), fp32_p in zip(self.named_parameters(), self.fp32_params):
            if p is not None:
                load(name, fp32_p, partial(load_fp32_p, fp32_p))
        self.chunk_manager.copy_chunk_group('fp16_param', 'fp32_param')

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
