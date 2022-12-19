import contextlib
import copy
import pprint
from typing import Callable, List

import torch
import torch.nn as nn
from torch.types import _bool, _device, _dtype
from torch.utils._pytree import tree_map

from colossalai.fx.profiler import MetaTensor
from colossalai.tensor import ColoParameter, ColoTensor, ColoTensorSpec
from colossalai.utils.model.utils import substitute_init_recursively

init = torch.nn.init

# reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/parameter.py#L73
_TorchTensorMethods = [
    torch.Tensor.half,
    torch.Tensor.float,
    torch.Tensor.double,
    torch.Tensor.char,
    torch.Tensor.short,
    torch.Tensor.int,
    torch.Tensor.long,
    torch.Tensor.cuda,
    torch.Tensor.cpu,
]

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
]


def init_from_spec(cls, t: torch.Tensor, spec: ColoTensorSpec):
    if cls == ColoTensor:
        return cls.from_torch_tensor(t, spec)
    else:
        return cls.from_torch_tensor(t, t.requires_grad, spec)


class UninitializedTensor(torch.Tensor):

    _cls_to_become = ColoTensor
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
        r._data = MetaTensor(elem, fake_device=device)
        return r

    def __init__(self, func, *args, dtype=None, device=None, **kwargs):
        self._factory_fn = (func, args, {'dtype': dtype, 'device': device, **kwargs})    # (func, args, kwargs)
        self._cached_fn = list()    # (func, args, kwargs)
        self._spec = ColoTensorSpec(pg=None, dist_attr=None, compute_attr=None)    # Default Spec

    def __repr__(self):
        if self._repr:
            self.__class__._repr = False
            s = f'UninitializedTensor: {self._factory_fn}\n'\
                f'_data: {self._data}\n'\
                f'cached_fn: {pprint.pformat(self._cached_fn)}\n'\
                f'spec: {self._spec}'
            self.__class__._repr = True
            return s
        else:
            return 'UninitializedTensor(...)'

    def materialize(self):
        func, args, kwargs = self._factory_fn
        t = func(*args, **kwargs)

        # apply cached_fn
        t = self._apply_cache(t)

        # apply spec
        return init_from_spec(self._cls_to_become, t, self._spec)

    # TODO(super-dainiu): device seems incorrect
    def traceable(self):
        func, args, kwargs = self._factory_fn
        t = MetaTensor(func(*args, **{**kwargs, 'device': 'meta'}), fake_device=kwargs['device'])
        return self._apply_cache(t)

    def _apply_cache(self, t):
        # apply cached methods
        # super-dainiu: support methods for single Tensor only
        replace = lambda x: t if isinstance(x, UninitializedTensor) else x

        for (func, args, kwargs) in self._cached_fn:
            o = func(*tree_map(replace, args), **tree_map(replace, kwargs))
            t = o if isinstance(o, torch.Tensor) else t    # if func returns non-Tensor, discard the value
        return t

    # cache everything
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        t = None

        if func in _TorchTensorMethods:
            t: UninitializedTensor = args[0].clone()
            t._cached_fn.append((func, (t,) + args[1:], kwargs))
            t._data = func(t._data, *args[1:], **kwargs)
            if isinstance(t._data, MetaTensor):
                return t
            else:
                return t._data

        def unwrap(t_):
            nonlocal t
            if isinstance(t_, UninitializedTensor):
                t = t_.clone()
                t._cached_fn.append((func, args, kwargs))
                t_ = t_._data
            return t_

        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs)
        t._data = func(*args, **kwargs)

        if isinstance(t._data, MetaTensor):
            return t
        else:
            return t._data

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        pass

    def to(self, *args, **kwargs) -> "UninitializedTensor":
        t = self.clone()
        t._cached_fn.append((torch.Tensor.to, (t,) + args, kwargs))
        t._data = t._data.to(*args, **kwargs)
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


# TODO: Not correct
class UninitializedParameter(UninitializedTensor, nn.Parameter):

    _cls_to_become = ColoParameter

    @staticmethod
    def __new__(cls, elem=None, requires_grad=True):
        if elem is None:
            elem = UninitializedTensor(torch.empty, 0)
        if type(elem) is UninitializedTensor or type(elem) is UninitializedParameter:
            # For ease of BC maintenance, keep this path for standard Tensor.
            # Eventually (tm), we should change the behavior for standard Tensor to match.
            r = torch.Tensor._make_wrapper_subclass(cls,
                                                    elem.size(),
                                                    strides=elem.stride(),
                                                    storage_offset=elem.storage_offset(),
                                                    dtype=elem.dtype,
                                                    layout=elem.layout,
                                                    device=elem.device,
                                                    requires_grad=requires_grad)
            r._data = elem._data
            return r
        raise RuntimeError(f"Creating an `UninitializedParameter` with `Tensor.subclasses` of "
                           f"`{type(elem).__name__}` is unexpected. Should be one of `UninitializedParameter`"
                           f", `UnintializedTensor`, or None.")

    def __init__(self, elem=None, requires_grad=True):
        self._factory_fn = elem._factory_fn
        self._cached_fn = elem._cached_fn
        self._spec = elem._spec

    def clone(self):
        return UninitializedParameter(super().clone(), requires_grad=self.requires_grad)

    __torch_function__ = torch._C._disabled_torch_function_impl


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

        # cannot monkey patch nn.Parameter because it is a class (????)
        setattr(torch.nn.parameter, 'Parameter', UninitializedParameter)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, (wrapper, orig) in self.overrides.items():
            setattr(torch, name, orig)
        setattr(torch.nn.parameter, 'Parameter', self._orig_nn_param)

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

    # Things to hack:
    # 1. torch.Tensor factory function (DONE)
    # 2. nn.Parameter
    # 3. init (DONE)
