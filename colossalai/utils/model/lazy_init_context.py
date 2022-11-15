#!/usr/bin/env python
# coding: utf-8

import inspect
import types
from typing import Callable, List

import torch
import torch.nn as nn

from colossalai.tensor import ColoParameter, ColoTensor
from colossalai.utils.model.utils import substitute_init_recursively


class LazyInitContext():
    """
    A context to allow for lazy weight initialization of PyTorch modules. It intercepts the tensor
    initialization functions for lazy initialization

    Note:
        This API is only experimental and subject to future changes.

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
        to_meta (bool): optional, whether to initialize the model with meta tensors, default is True. This
            argument exists for now because some corner cases such as self.weight = torch.zeros(...) cannot be captured yet.
        extra_torch_tensor_func (List[str]): extra torch tensor functions related
            to value setting, such as `zero_` and `triu_`. `zero_` is pre-added by default.
    """

    tensor_set_value_func = ['zero_', 'fill_']

    def __init__(self, to_meta: bool = True, extra_torch_tensor_func: List[str] = None):
        # TODO: hijack the torch constructor functions as well
        self._to_meta = to_meta
        self._intercepted_nn_init_func_cache = {}
        self._nn_init_methods = self._get_nn_init_methods()
        self._torch_mod_cls = torch.nn.modules.module.Module

        if extra_torch_tensor_func:
            # use tuple to remove duplicates
            self._torch_tensor_funcs = tuple(self.tensor_set_value_func + extra_torch_tensor_func)
        else:
            self._torch_tensor_funcs = self.tensor_set_value_func

    @property
    def to_meta(self):
        return self._to_meta

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

        substitute_init_recursively(self._torch_mod_cls, _activate_wrap_init, set())

    def _unpatch_submodule_init(self):

        def _recover_orig_init(cls):
            cls.__init__ = cls.__orig_init__

        substitute_init_recursively(self._torch_mod_cls, _recover_orig_init, set())

    def _patch_torch_tensor_funcs(self):
        # patch tensor value-setting functions
        for func_name in self._torch_tensor_funcs:
            origin_func_name = self._get_tmp_origin_func_ref(func_name)
            origin_func = getattr(torch.Tensor, func_name)
            setattr(torch.Tensor, origin_func_name, origin_func)
            setattr(torch.Tensor, func_name, self._cache_init_func(origin_func))

    def _unpatch_torch_tensor_funcs(self):
        for func_name in self._torch_tensor_funcs:
            origin_func_name = self._get_tmp_origin_func_ref(func_name)
            origin_func = getattr(torch.Tensor, origin_func_name)
            setattr(torch.Tensor, func_name, origin_func)

    def __enter__(self):
        self._patch_torch_tensor_funcs()
        self._patch_nn_init_funcs()

        if self._to_meta:
            self._patch_submodule_init()
        return self

    def __exit__(self, *args, **kwargs):
        if self._to_meta:
            self._unpatch_submodule_init()
        self._unpatch_nn_init_funcs()
        self._unpatch_torch_tensor_funcs()

    def lazy_init_parameters(self, model: torch.nn.Module, device='cpu'):
        """
        Initialize the weights of the meta-tensor model.

        Args:
            model (`torch.nn.Module`): the model instantiated under the context.
            device (str): the device on which weights are initialized

        """

        def _init_recursively(module: nn.Module):
            # recursively initialize the module
            for mod in module.children():
                _init_recursively(mod)

            # initialize and shard tensors directly attached to the current module
            for name, param in module.named_parameters(recurse=False):
                _init_and_shard(module, name, param)

            for name, buf in module.named_buffers(recurse=False):
                _init_and_shard(module, name, buf)

        @torch.no_grad()
        def _init_and_shard(module, name, tensor):
            # check whether the tensor is a buffer or parameter
            is_param = isinstance(tensor, nn.parameter.Parameter)

            # get sharding spec
            dist_spec = getattr(tensor, 'dist_spec', None)
            pg = getattr(tensor, 'pg', None)
            comp_spec = getattr(tensor, 'comp_spec', None)

            # convert the tensor from meta to materialized one
            if tensor.is_meta:
                materialized_tensor = torch.empty_like(tensor, device=device)
                # if this tensor is a meta tensor, it must have an init function
                assert tensor in self._intercepted_nn_init_func_cache
            else:
                materialized_tensor = tensor

            # apply init function
            if tensor in self._intercepted_nn_init_func_cache:
                init_func, args, kwargs = self._intercepted_nn_init_func_cache[tensor][-1]
                init_func(materialized_tensor, *args, **kwargs)

            # convert it to ColoTensor or ColoParameter
            if is_param:
                tensor = ColoParameter.from_torch_tensor(materialized_tensor, requires_grad=tensor.requires_grad)
            else:
                tensor = ColoTensor.from_torch_tensor(materialized_tensor)

            # override the original tensor
            with torch.no_grad():
                setattr(module, name, tensor)

            # apply sharding
            if dist_spec:
                tensor.process_group = pg
                tensor.set_tensor_spec(dist_spec, comp_spec)

        _init_recursively(model)

        return model
