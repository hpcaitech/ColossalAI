#!/usr/bin/env python
# coding: utf-8

import torch
from colossalai.tensor import ColoParameter, ColoTensor
import types
import inspect
import typing
from typing import List, Callable
from colossalai.utils.model.utils import substitute_init_recursively
import copy


class LazyInitContext():
    """
    A context to allow for lazy weight initialization of PyTorch modules. It intercepts the tensor 
    initialization functions for lazy initialization
    
    Note:
        This API is only experimental and subject to future changes. 
        It should be integrated with meta tensor initialization in the future.
        
    Usage:
        with LazyInitContext() as ctx:
            model = nn.Linear(10, 10)
            model.weight.zero_()
        
        # make sure the weight is a meta tensor
        assert model.weight.is_meta
        
        # initialize weights
        ctx.lazy_init_parameters(model)
        
        # make sure the weight is not a meta tensor 
        # and initialized correctly
        assert not model.weight.is_meta and torch.all(model.weight == 0)
        
    Args:
        extra_torch_tensor_func (List[str]): extra torch tensor functions related 
            to value setting, such as `zero_` and `triu_`. `zero_` is pre-added by default.
    """

    tensor_set_value_func = ['zero_', 'fill_']
    tensor_constructor = ['zeros', 'ones', 'rand']

    def __init__(self, extra_torch_tensor_func: List[str] = None):
        self._intercepted_nn_init_func_cache = {}
        self._nn_init_methods = self._get_nn_init_methods()
        self._torch_mod_cls = torch.nn.modules.module.Module

        if extra_torch_tensor_func:
            # use tuple to remove duplicates
            self._torch_tensor_funcs = tuple(self.tensor_set_value_func + extra_torch_tensor_func)
        else:
            self._torch_tensor_funcs = self.tensor_set_value_func

    def _cache_init_func(self, func):
        """
        This method wraps the ``torch.nn.init`` method and torch tensor value-setting functions
        so that the function call is cached instead of being executed.
        """

        def wrapped_init_func(tensor, *args, **kwargs):
            if tensor not in self._intercepted_nn_init_func_cache:
                self._intercepted_nn_init_func_cache[tensor] = []
            self._intercepted_nn_init_func_cache[tensor].append((func, args, kwargs))

        return wrapped_init_func

    def _get_nn_init_methods(self):
        """
        This method looks for all available functions in the ``torch.nn.init``
        module.
        """
        nn_init_method_names = dir(torch.nn.init)
        nn_init_methods = []

        # look for all methods in ``torch.nn.init`` module
        for name in nn_init_method_names:
            nn_init_methods.append((name, getattr(torch.nn.init, name)))

        def _has_tensor_in_arg(func):
            hints = typing.get_type_hints(func)
            for k, v in hints.items():
                if v is torch.Tensor:
                    return True
            return False

        def _is_init_method(item):
            name, func = item

            if (not isinstance(func, types.FunctionType) or name.startswith('_') or not name.endswith('_')):
                return False
            else:
                return True

        # remove methods which are not init functions
        nn_init_methods = list(filter(_is_init_method, nn_init_methods))
        return nn_init_methods

    def _wrap_module_init(self, func):
        """
        This method wraps the calls to the `__init__` of ``torch.nn.Module`` and replaces
        the argument device with value 'meta' so that all modules are created as meta tensors.
        """
        has_device = 'device' in inspect.signature(func).parameters

        def layer_lazy_init(module, *args, **kwargs):
            # if this module contains device argument
            # we set it to meta to initialize as meta backend
            if has_device:
                kwargs['device'] = 'meta'
            func(module, *args, **kwargs)

            # if device is not found, we intialize it and convert to meta
            if not has_device:
                module.to('meta')

        return layer_lazy_init

    def _get_tmp_origin_func_ref(self, name):
        """
        Generate a function name for consistency during caching and retrieving.
        """
        return f'_orig_{name}'

    def _patch_nn_init_funcs(self):
        # patch nn.init functions
        for name, func in self._nn_init_methods:
            setattr(torch.nn.init, name, self._cache_init_func(func))

    def _unpatch_nn_init_funcs(self):
        # unpatch nn.init functions
        for name, func in self._nn_init_methods:
            setattr(torch.nn.init, name, func)

    def _patch_submodule_init(self):
        # patch classes __init__ methods
        def _activate_wrap_init(cls):
            cls.__orig_init__ = cls.__init__
            cls.__init__ = self._wrap_module_init(cls.__init__)

        substitute_init_recursively(self._torch_mod_cls, _activate_wrap_init)

    def _unpatch_submodule_init(self):

        def _recover_orig_init(cls):
            cls.__init__ = cls.__orig_init__

        substitute_init_recursively(self._torch_mod_cls, _recover_orig_init)

    def _patch_torch_tensor_funcs(self):
        # patch tensor value-setting functions
        for func_name in self._torch_tensor_funcs:
            origin_func_name = self._get_tmp_origin_func_ref(func_name)
            origin_func = getattr(torch.Tensor, func_name)
            setattr(torch.Tensor, origin_func_name, origin_func)
            setattr(torch.Tensor, func_name, self._cache_init_func(origin_func))

    def __enter__(self):
        self._patch_torch_tensor_funcs()
        self._patch_nn_init_funcs()
        self._patch_submodule_init()
        return self

    def __exit__(self, *args, **kwargs):
        self._unpatch_submodule_init()
        self._unpatch_nn_init_funcs()
        self._unpatch_torch_tensor_funcs()

    def lazy_init_parameters(self, model: torch.nn.Module, device='cpu', call_back: Callable = None):
        """
        Initialize the weights of the meta-tensor model.
        
        Args:
            model (`torch.nn.Module`): the model instantiated under the context.
            device (str): the device on which weights are initialized
        """
        # build param mapping
        param_id_to_name = dict()
        for name, param in model.named_parameters():
            param_id_to_name[id(param)] = name
        for name, buffer in model.named_buffers():
            param_id_to_name[id(buffer)] = name

        def _init_recurively(module):
            for mod in module.children():
                _init_recurively(mod)

            for param in module.parameters(recurse=False):
                _convert_to_real_tensor(param, module, is_buffer=False)

            for buf in module.buffers(recurse=False):
                _convert_to_real_tensor(buf, module, is_buffer=True)

        def _convert_to_real_tensor(tensor, module, is_buffer):
            if not tensor.is_meta:
                return tensor
            tensor_id = id(tensor)
            param_full_name = param_id_to_name[tensor_id]
            real_tensor = torch.empty_like(tensor, dtype=tensor.dtype, device=device)

            # look for initialization function
            if tensor in self._intercepted_nn_init_func_cache:
                init_func, args, kwargs = self._intercepted_nn_init_func_cache[tensor][-1]
            else:
                raise RuntimeError(f"{param_full_name} is not associated with a initialization function")

            # initialize
            init_func(real_tensor, *args, **kwargs)

            # convert to colotensor
            # if this is a parameter, we should convert it to coloparameter
            # if this is a buffer, we should convert it to colotensor
            if is_buffer:
                real_tensor = ColoTensor.from_torch_tensor(real_tensor)
            else:
                real_tensor = ColoParameter.from_torch_tensor(real_tensor, requires_grad=tensor.requires_grad)

            # convert to distribted mode
            if hasattr(tensor, 'dist_spec'):
                with torch.no_grad():
                    pg = getattr(tensor, 'pg', None)
                    real_tensor = real_tensor.redistribute(tensor.dist_spec, pg)

            # override the original tensor attribute
            if '.' in param_full_name:
                param_name = param_full_name.rsplit('.')[-1]
            else:
                param_name = param_full_name
            setattr(module, param_name, real_tensor)

            # execute call_back function on the materailized tensor
            # this can be used for possible extension
            if call_back:
                call_back(real_tensor)
            return real_tensor

        # build user specified model
        with torch.no_grad():
            _init_recurively(model)

        return
