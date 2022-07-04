#!/usr/bin/env python
# coding: utf-8

import torch
import types
import inspect
import typing
import copy
from typing import List, Callable
from colossalai.utils.model.utils import substitute_init_recursively, get_nn_init_methods
from colossalai.tensor import ColoTensor
from colossalai.tensor import ColoParameter


class MaterializationContext():
    """
    A context to materialize the meta tensor. It intercepts the nn.init 
    method and only execute the last init method which is recorded during
    lazy init. It will also add necessary communication op to get correct
    weight distribution.
    Note:
        This API is only experimental and subject to future changes. 
        It may inherit ColoInitContext to build module with ColoTensor.

    Usage:
    ctx = MaterializationContext(lazy_init_dict)
    with ctx:
        model = nn.Linear(10, 10)
    """

    def __init__(self, lazy_init_dict):
        self._nn_init_dict = {}
        self._last_init_dict = lazy_init_dict
        self._nn_init_methods = get_nn_init_methods()

    def _exec_func(self, func, tensor, args, kwargs):
        if not isinstance(tensor, ColoTensor):
            func(tensor, *args, **kwargs)
        # TODO(lyl): add communication here if tensor is ColoTensor

    def _process_nn_init_method(self, func):
        """
        This method wraps the ``torch.nn.init`` method so that only the last nn init call
        is executed, other calls are cached. Necessary communication op will be added during
        executing to get correct weight distribution.
        """

        def wrapped_nn_init_func(tensor, *args, **kwargs):
            assert tensor in self._last_init_dict, f'We only support reinitializing tensors which intercepted during initializing by us.'
            init_times = self._last_init_dict[tensor][-1]
            if tensor not in self._intercepted_nn_init_dict:
                self._intercepted_nn_init_dict[tensor] = 1
            else:
                self._intercepted_nn_init_dict[tensor] += 1
            # exec func if it is the last init call
            if self._intercepted_nn_init_dict[tensor] == init_times:
                _exec_func(func, tensor, args, kwargs)

        return wrapped_nn_init_func

    def _patch_nn_init_funcs(self):
        # patch nn.init functions
        for name, func in self._nn_init_methods:
            setattr(torch.nn.init, name, self._process_nn_init_method(func))

    def _unpatch_nn_init_funcs(self):
        # unpatch nn.init functions
        for name, func in self._nn_init_methods:
            setattr(torch.nn.init, name, func)

    def __enter__(self):
        self._patch_nn_init_funcs()
        return self

    def __exit__(self, *args, **kwargs):
        self._unpatch_nn_init_funcs()


class LazyInitContext():
    """
    A context to allow for lazy weight initialization of PyTorch modules. It intercepts the tensor 
    initialization functions for lazy initialization
      
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
        self._intercepted_nn_init_dict = {}
        self._nn_init_methods = get_nn_init_methods()
        self._torch_mod_cls = torch.nn.modules.module.Module

        if extra_torch_tensor_func:
            # use tuple to remove duplicates
            self._torch_tensor_funcs = tuple(self.tensor_set_value_func + extra_torch_tensor_func)
        else:
            self._torch_tensor_funcs = self.tensor_set_value_func

    def _cache_nn_init_method(self, func):
        """
        This method wraps the ``torch.nn.init`` method so that the function call
        is cached instead of being executed.
        """

        def wrapped_nn_init_func(tensor, *args, **kwargs):
            if tensor not in self._intercepted_nn_init_dict:
                self._intercepted_nn_init_dict[tensor] = [func, args, kwargs, 1]
            else:
                init_times = self._intercepted_nn_init_dict[tensor][-1]
                init_times += 1
                self._intercepted_nn_init_dict[tensor] = [func, args, kwargs, init_times]

        return wrapped_nn_init_func

    def _wrap_module_init(self, func):
        """
        This method wraps the calls to the `__init__` of ``torch.nn.Module`` and replaces
        the argument device with value 'meta' so that all modules are created as meta tensors.
        """
        has_device = 'device' in inspect.signature(func).parameters

        def layer_lazy_init(module, *args, **kwargs):
            self._intercepted_init_func_cache.append(
                dict(func=func, module=module, args=args, kwargs=copy.deepcopy(kwargs)))
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
            setattr(torch.nn.init, name, self._cache_nn_init_method(func))

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
        self._patch_nn_init_funcs()
        self._patch_submodule_init()
        return self

    def __exit__(self, *args, **kwargs):
        self._unpatch_submodule_init()
        self._unpatch_nn_init_funcs()
        # build model_rebuild_dict in reverse order to make sure get correct init func for inherited class.
        self.module_rebuild_dict = {}
        self._intercepted_init_func_cache.reverse()
        for cache in self._intercepted_init_func_cache:
            self.module_rebuild_dict[cache['module']] = (cache['func'], cache['args'], cache['kwargs'])
        self._intercepted_init_func_cache.reverse()

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

        assert model in self.module_rebuild_dict, 'We only support rebuild modules which intercepted during initializing by us.'

        def _process_arg(arg):
            """
            Process args recursively. If arg is a torch.nn.Module instance in module_rebuild_dict, 
            we need to rebuild it with real parameters. If arg is a tuple or list, we will process
            the element of arg with this function again.
            """
            if torch.is_tensor(arg):
                tensor_id = id(arg)
                if tensor_id in param_id_to_name:
                    arg = _replace_meta_param_with_real_param(arg)

            elif isinstance(arg, torch.nn.Module):
                if arg in self.module_rebuild_dict:
                    arg = self.lazy_init_parameters(model=arg, device=device, call_back=call_back)

            elif isinstance(arg, (tuple, list)):
                rst_list = []
                for element in arg:
                    processed_element = _process_arg(element)
                    rst_list.append(processed_element)
                arg = rst_list
            return arg

        def _replace_meta_param_with_real_param(meta_param):
            if meta_param.device != 'meta':
                return meta_param
            tensor_id = id(meta_param)
            param_full_name = param_id_to_name[tensor_id]
            real_param = torch.empty_like(meta_param, dtype=meta_param.dtype, device=device)
            # real_param = ColoParameter(real_param, requires_grad=meta_param.requires_grad)

            if '.' in param_full_name:
                submodule_name, param_name = param_full_name.rsplit('.', 1)
                submodule = model.get_submodule(submodule_name)
            else:
                submodule = model
                param_name = param_full_name
            setattr(submodule, param_name, real_param)

            # execute call_back function on the materialized tensor
            # this can where sharding comes in
            if call_back:
                call_back(real_param)
            return real_param

        func, args, kwargs = self.module_rebuild_dict[model]
        args = list(args)

        # check args for parameter replacement
        for idx, arg in enumerate(args):
            arg = _process_arg(arg)
            args[idx] = arg

        # check kwargs for parameter replacement
        for arg_name, arg in kwargs.items():
            if arg_name == 'device':
                arg = device
            else:
                arg = _process_arg(arg)
            kwargs[arg_name] = arg

        # build user specified model
        with torch.no_grad():
            func(model, *args, **kwargs)

        return model
