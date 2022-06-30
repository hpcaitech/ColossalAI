#!/usr/bin/env python
# coding: utf-8

import torch
from colossalai.tensor import ColoParameter
import types
import inspect
import typing
from typing import List, Callable
from colossalai.utils.model.utils import substitute_init_recursively


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

    tensor_set_value_func = ['zero_']

    def __init__(self, extra_torch_tensor_func: List[str] = None):
        self._intercepted_init_func_cache = []
        self._nn_init_methods = self._get_nn_init_methods()
        self._torch_mod_cls = torch.nn.modules.module.Module

        if extra_torch_tensor_func:
            # use tuple to remove duplicates
            self._torch_tensor_funcs = tuple(self.tensor_set_value_func + extra_torch_tensor_func)
        else:
            self._torch_tensor_funcs = self.tensor_set_value_func

    def _cache_func(self, func):
        """
        This method wraps the ``torch.nn.init`` method so that the function call
        is cached instead of being executed.
        """

        def wrapped_init_func(*args, **kwargs):
            self._intercepted_init_func_cache.append(dict(func=func, args=args, kwargs=kwargs))

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
            if (not isinstance(func, types.FunctionType) or name.startswith('_') or not name.endswith('_')
                    or not _has_tensor_in_arg(func)):
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
            self._intercepted_init_func_cache.append(dict(func=func, module=module, args=args, kwargs=kwargs))
            if has_device:
                kwargs['device'] = 'meta'
            func(module, *args, **kwargs)
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
            setattr(torch.nn.init, name, self._cache_func(func))

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
            setattr(torch.Tensor, func_name, self._cache_func(origin_func))

    def _unpatch_torch_tensor_funcs(self):
        for func_name in self._torch_tensor_funcs:
            origin_func_name = self._get_tmp_origin_func_ref(func_name)
            origin_func = getattr(torch.Tensor, origin_func_name)
            setattr(torch.Tensor, func_name, origin_func)

    def __enter__(self):
        self._patch_submodule_init()
        return self

    def __exit__(self, *args, **kwargs):
        self._unpatch_submodule_init()

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

        def _replace_meta_param_with_real_param(meta_param):
            tensor_id = id(meta_param)
            param_full_name = param_id_to_name[tensor_id]
            real_param = torch.empty_like(meta_param, dtype=meta_param.dtype, device=device)
            real_param = ColoParameter(real_param, requires_grad=meta_param.requires_grad)

            if '.' in param_full_name:
                submodule_name, param_name = param_full_name.rsplit('.', 1)
                submodule = model.get_submodule(submodule_name)
            else:
                submodule = model
                param_name = param_full_name
            setattr(submodule, param_name, real_param)

            # execute call_back function on the materailized tensor
            # this can where sharding comes in
            if call_back:
                call_back(real_param)
            return real_param

        # build modules
        # visit the cache list in reverse order
        for index in range(len(self._intercepted_init_func_cache)):
            cache = self._intercepted_init_func_cache[len(self._intercepted_init_func_cache) - index - 1]
            func = cache['func']
            module = cache['module']
            args = list(cache['args'])
            kwargs = cache['kwargs']

            # check args for parameter replacement
            for idx, arg in enumerate(args):
                if torch.is_tensor(arg):
                    tensor_id = id(arg)

                    if tensor_id not in param_id_to_name:
                        continue
                    else:
                        arg = _replace_meta_param_with_real_param(arg)
                        args[idx] = arg

            # check kwargs for parameter replacement
            for arg_name, arg in enumerate(kwargs):
                if torch.is_tensor(arg):
                    tensor_id = id(arg)

                    if tensor_id not in param_id_to_name:
                        continue
                    else:
                        arg = _replace_meta_param_with_real_param(arg)
                        kwargs[arg_name] = arg

            with torch.no_grad():
                func(module, *args, **kwargs)
