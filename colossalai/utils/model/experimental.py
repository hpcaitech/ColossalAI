import contextlib
import copy
import pprint
from typing import Callable, List, Union

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from colossalai.fx.profiler import MetaTensor
from colossalai.tensor import ColoParameter, ColoTensor, ColoTensorSpec

# reference: https://pytorch.org/cppdocs/notes/tensor_creation.html
_TorchFactoryFunc = [
    "arange",
    "empty",
    "eye",
    "full",
    "linspace",
    "logspace",
    "ones",
    "rand",
    "randn",
    "randint",
    "randperm",
    "zeros",
    "tensor",
]


class UninitializedTensor(torch.Tensor):

    _repr = True

    @staticmethod
    def __new__(cls, func, *args, dtype=None, device=None, **kwargs):
        elem = func(*args, dtype=dtype, device='meta', **kwargs)
        r = torch.Tensor._make_wrapper_subclass(cls,
                                                elem.size(),
                                                strides=elem.stride(),
                                                storage_offset=elem.storage_offset(),
                                                dtype=elem.dtype,
                                                layout=elem.layout,
                                                device=device if device is not None else torch.device('cpu'),
                                                requires_grad=elem.requires_grad)
        r._meta_data = MetaTensor(elem, fake_device=device)
        return r

    def __init__(self, func, *args, dtype=None, device=None, **kwargs):
        self._factory_fn = (func, args, {'dtype': dtype, 'device': device, **kwargs})    # (func, args, kwargs)
        self._cached_fn = list()    # (func, args, kwargs)
        self._spec = ColoTensorSpec(pg=None, dist_attr=None, compute_attr=None)    # Default Spec

    def __repr__(self):
        if self._repr:
            self.__class__._repr = False
            s = f'UninitializedTensor: {self._factory_fn}\n'\
                f'meta_data: {self._meta_data}\n'\
                f'cached_fn: {pprint.pformat(self._cached_fn)}\n'\
                f'spec: {self._spec}'
            self.__class__._repr = True
            return s
        else:
            return 'UninitializedTensor(...)'

    def materialize(self) -> Union[ColoParameter, ColoTensor]:
        func, args, kwargs = self._factory_fn
        t = func(*args, **kwargs)

        # apply cached_fn
        t = self._apply_cache(t)

        # apply spec
        if isinstance(self, nn.Parameter):
            return ColoParameter.from_torch_tensor(t, t.requires_grad, self._spec)
        else:
            return ColoTensor.from_torch_tensor(t, self._spec)

    def traceable(self) -> MetaTensor:
        func, args, kwargs = self._factory_fn
        t = MetaTensor(func(*args, **{**kwargs, 'device': 'meta'}), fake_device=kwargs['device'])
        if isinstance(self, nn.Parameter):
            return nn.Parameter(self._apply_cache(t), requires_grad=self.requires_grad)
        else:
            return self._apply_cache(t)

    def _apply_cache(self, t) -> torch.Tensor:
        # apply cached methods
        # super-dainiu: support methods for single Tensor only
        replace = lambda x: t if isinstance(x, UninitializedTensor) else x
        packed = None

        for (func, args, kwargs) in self._cached_fn:
            if func == torch.Tensor.requires_grad_:
                packed = func, args, kwargs    # requires grad should be set at last
            else:
                o = func(*tree_map(replace, args), **tree_map(replace, kwargs))
                t = o if isinstance(o, torch.Tensor) else t    # if func returns non-Tensor, discard the value

        # super-dainiu: set requires_grad after all inplace-ops are done
        if packed is not None:
            func, args, kwargs = packed
            func(*tree_map(replace, args), **tree_map(replace, kwargs))

        return t

    # cache everything
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        t = None

        unwrap = lambda t: t._meta_data if isinstance(t, UninitializedTensor) else t

        if isinstance(func, torch._C.ScriptMethod):
            t: UninitializedTensor = args[0].clone()
            t._cached_fn.append((func, (t,) + args[1:], kwargs))
            t._meta_data = getattr(t._meta_data, func.name)(*tree_map(unwrap, args[1:]), **tree_map(unwrap, kwargs))

        else:

            def unwrap(t_):
                nonlocal t
                if isinstance(t_, UninitializedTensor):
                    t = t_ if (func.__name__.endswith('_')
                               or func.__name__ == "__set__") and not (func.__name__.endswith('__')) else t_.clone()
                    t._cached_fn.append((func, args, kwargs))
                    t_ = t_._meta_data
                return t_

            args = tree_map(unwrap, args)
            kwargs = tree_map(unwrap, kwargs)
            t._meta_data = func(*args, **kwargs)

        if isinstance(t._meta_data, MetaTensor):
            return t
        else:
            return t._meta_data

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        pass

    def to(self, *args, **kwargs) -> "UninitializedTensor":
        t = self.clone()
        t._cached_fn.append((torch.Tensor.to, (t,) + args, kwargs))
        t._meta_data = t._meta_data.to(*args, **kwargs)
        return t

    def clone(self) -> "UninitializedTensor":
        func, args, kwargs = self._factory_fn
        t = UninitializedTensor(func, *args, **kwargs)
        t._cached_fn = [x for x in self._cached_fn]
        t._spec = copy.deepcopy(self._spec)
        return t

    @property
    def spec(self) -> ColoTensorSpec:
        return self._spec

    @spec.setter
    def spec(self, other: ColoTensorSpec):
        self._spec = other

    def detach(self):
        return self.clone()


class LazyInitContext():

    def __init__(self):
        self.overrides = {}
        self._orig_nn_param = torch.nn.Parameter

    def __enter__(self):

        def wrap_factory_method(target):
            # factory functions (eg. torch.empty())
            def wrapper(*args, **kwargs):
                return UninitializedTensor(target, *args, **kwargs)

            return wrapper, target

        def wrap_factory_like_method(orig_target, target):
            # factory_like functions (eg. torch.empty_like())
            def wrapper(*args, **kwargs):
                orig_t = args[0]
                return UninitializedTensor(orig_target, *args[1:], device=orig_t.device, dtype=orig_t.dtype, **kwargs)

            return wrapper, target

        self.overrides = {
            target: wrap_factory_method(getattr(torch, target))
            for target in _TorchFactoryFunc
            if callable(getattr(torch, target, None))
        }

        self.overrides.update({
            target + '_like': wrap_factory_like_method(getattr(torch, target), getattr(torch, target + '_like'))
            for target in _TorchFactoryFunc
            if callable(getattr(torch, target + '_like', None))
        })

        for name, (wrapper, orig) in self.overrides.items():
            setattr(torch, name, wrapper)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, (wrapper, orig) in self.overrides.items():
            setattr(torch, name, orig)

    @staticmethod
    def materialize(module: torch.nn.Module):
        """Materialize and shard ``nn.Module`` -- Initialize all ``nn.Parameter`` as ``ColoParameter``

        Args:
            module (torch.nn.Module): LazyInit Module
        """

        @torch.no_grad()
        def init_recursively(module: nn.Module):
            # recursively initialize the module
            for mod in module.children():
                init_recursively(mod)

            # initialize tensors directly attached to the current module
            for name, param in module.named_parameters(recurse=False):
                setattr(module, name, param.materialize())

            for name, buf in module.named_buffers(recurse=False):
                setattr(module, name, buf.materialize())

        init_recursively(module)
        return module

    @staticmethod
    @contextlib.contextmanager
    def traceable(module: torch.nn.Module):
        """Enable ``ColoTracer`` -- Initialize all ``nn.Parameters`` as ``MetaTensor``

        Args:
            module (torch.nn.Module): LazyInit Module
        """
        orig_val = dict()

        def init_recursively(module: nn.Module):
            # recursively initialize the module
            for mod in module.children():
                init_recursively(mod)

            # initialize tensors directly attached to the current module
            for name, param in module.named_parameters(recurse=False):
                setattr(module, name, param.traceable())
                orig_val[(module, name)] = param

            for name, buf in module.named_buffers(recurse=False):
                setattr(module, name, buf.traceable())
                orig_val[(module, name)] = buf

        init_recursively(module)

        yield

        # restore original values
        for (module, name), val in orig_val.items():
            setattr(module, name, val)
