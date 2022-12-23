import contextlib
import copy
import gc
import pprint
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx.profiler import MetaTensor
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.tensor.sharding_spec import ShardingSpec

# reference: https://pytorch.org/cppdocs/notes/tensor_creation.html
_TorchFactoryMethod = [
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

orig_empty = torch.empty    # avoid override

scm = ShapeConsistencyManager()


class LazyTensor(torch.Tensor):
    """A naive implementation of LazyTensor (https://arxiv.org/pdf/2102.13267.pdf).

    Usage:
        1. Use ``LazyTensor`` instead of ``torch.Tensor``.
        >>> x = LazyTensor(torch.zeros, 2, 3)
        >>> x += 1
        >>> y = x * x
        >>> y = y.cuda().half()
        >>> y[0, 0] = 0
        >>> y = y.materialize()     # materialize the tensor
        >>> print(y)
        tensor([[0., 1., 1.],
                [1., 1., 1.]], device='cuda:0', dtype=torch.float16)

        2. Generate ``MetaTensor`` from ``LazyTensor``
        >>> x = LazyTensor(torch.zeros, 2, 3)
        >>> x.reshape(3, 2)
        >>> x = x.traceable()    # generate ``MetaTensor``
        >>> print(x)
        MetaTensor(..., size=(3, 2), device=cpu, dtype=torch.float32)

        3. Use ``LazyTensor`` to generate sharded ``nn.Parameter``.
        >>> x = LazyTensor(torch.zeros, 2, 3)
        >>> x.spec = ...    # some ``ShardingSpec``
        >>> x.distribute()    # distribute the tensor according to the ``ShardingSpec``

    Warnings:
        1. Cases that ``LazyTensor`` can't deal with.
        >>> x = LazyTensor(torch.ones, 2, 3)
        >>> x[0, 0] = -x[0, 0]    # this will cause infinite recursion

        2. ``LazyTensor.materialize()`` can't be called multiple times.
        >>> x = LazyTensor(torch.ones, 2, 3)
        >>> x.materialize()
        >>> x.materialize()    # this is disallowed
    """

    _repr = True
    _meta_data: Optional[MetaTensor] = None    # shape, dtype, device
    _cached_data: Optional[torch.Tensor] = None    # materialized data

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
        self._factory_method = (func, args, {'dtype': dtype, 'device': device, **kwargs})    # (func, args, kwargs)
        self._cached_buffer = list()    # (func, args, kwargs)
        self._spec = None
        self._data = self

    def __repr__(self):
        if self._repr:
            # avoid recursive representation
            self.__class__._repr = False
            s = f'LazyTensor(..., size={tuple(self._meta_data.shape)}, device={self._meta_data.device}, dtype={self._meta_data.dtype})\n'\
                f'factory method: {self._factory_method}\n'\
                f'cached: {pprint.pformat(self._cached_buffer) if self._cached_data is None else self._cached_data}\n'\
                f'spec: {self._spec}'
            self.__class__._repr = True
            return s
        else:
            return 'LazyTensor(...)'

    def materialize(self) -> torch.Tensor:
        """Materialize the ``LazyTensor`` to ``torch.Tensor``.

        Warnings:
            Calling ``self.materialize()`` will clear all cached sequence and factory method,
            because we don't allow materialize the same ``LazyTensor`` twice.
            This is mentioned in the paper: https://arxiv.org/pdf/2102.13267.pdf (Part 4.3).

        Returns:
            torch.Tensor: The materialized tensor.
        """
        target = self._data._realize_cached_data()
        if isinstance(self, nn.Parameter):
            target = nn.Parameter(target, requires_grad=self.requires_grad)
        self._clear_all()
        return target

    def traceable(self) -> MetaTensor:
        """Generate ``MetaTensor`` from ``LazyTensor``. (Mostly for tracing)

        Returns:
            MetaTensor: The generated ``MetaTensor``.
        """
        if isinstance(self, nn.Parameter):
            return nn.Parameter(self._meta_data, requires_grad=self.requires_grad)
        else:
            return self._meta_data

    def distribute(self) -> torch.Tensor:
        """Distribute the ``LazyTensor`` according to the ``ShardingSpec``.

        Returns:
            torch.Tensor: The sharded tensor.
        """
        if self._spec is None:
            raise RuntimeError('ShardingSpec is not set for\n{self}')
        spec, device_mesh = self._spec, self._spec.device_mesh
        target = self.materialize()

        # TODO(some man): better not be coupled with auto-parallel
        target.data = scm.apply_for_autoparallel_runtime(target.data, ShardingSpec(device_mesh, target.shape, {}),
                                                         spec).detach().clone()
        return target

    def _realize_cached_data(self) -> torch.Tensor:
        # self._cached_data should be generated after the first call of this function
        if self._cached_data is None:
            if self._factory_method is not None:
                # apply factory method
                func, args, kwargs = self._factory_method

                # apply cached sequence
                self._cached_data = self._apply_cache_buffer(func(*args, **kwargs))
            else:
                # apply cached sequence only
                self._cached_data = self._apply_cache_buffer()
        return self._cached_data

    def _apply_cache_buffer(self, target=None) -> torch.Tensor:
        # dump all cached sequence
        # super-dainiu: support methods for single Tensor only
        def replace(x):
            if x is self:
                return target
            elif isinstance(x, LazyTensor):
                return x._realize_cached_data()
            return x

        packed = None

        for (func, args, kwargs) in self._cached_buffer:
            if func == torch.Tensor.requires_grad_:
                packed = func, args, kwargs    # requires grad should be set at last
            else:
                o = func(*tree_map(replace, args), **tree_map(replace, kwargs))
                target = o if isinstance(o, torch.Tensor) else target    # if func returns non-Tensor, discard the value

        # super-dainiu: set requires_grad after all inplace-ops are done
        if packed is not None:
            func, args, kwargs = packed
            func(*tree_map(replace, args), **tree_map(replace, kwargs))

        return target

    # clear all means:
    #   1. clear factory method
    #   2. clear cached sequence
    #   3. clear cached data
    def _clear_all(self):
        self._cached_data = None
        self._cached_buffer = None
        self._data = None
        gc.collect()    # avoid memory leak

    # cache everything with __torch_function__
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        target = None

        if isinstance(func, torch._C.ScriptMethod):

            def unwrap(x):
                if isinstance(x, LazyTensor):
                    return x._meta_data
                return x

            target: LazyTensor = args[0].clone()
            target._cached_buffer.append((func, args, kwargs))
            target._meta_data = getattr(target._meta_data, func.name)(*tree_map(unwrap, args[1:]),
                                                                      **tree_map(unwrap, kwargs))

        else:

            def unwrap(x):
                nonlocal target
                if isinstance(x, LazyTensor):
                    target = x if (func.__name__.endswith('_') and not (func.__name__.endswith('__'))
                                   or func.__name__ == "__setitem__") else x.clone()
                    target._cached_buffer.append((func, args, kwargs))
                    return x._meta_data
                return x

            args = tree_map(unwrap, args)
            kwargs = tree_map(unwrap, kwargs)
            o = func(*args, **kwargs)

        if isinstance(o, MetaTensor):
            target._meta_data = o
            return target
        else:
            return o

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        pass    # skip

    def clone(self) -> "LazyTensor":
        """Create a new ``LazyTensor`` with same cached sequence and factory method.

        Returns:
            LazyTensor: the new ``LazyTensor``
        """
        target = LazyTensor(orig_empty, 0, dtype=self._meta_data.dtype, device=self._meta_data.device)
        target._factory_method = None
        target._cached_buffer = list()
        target._meta_data = self._meta_data.clone()
        target._cached_data = self._cached_data.clone() if self._cached_data is not None else None
        target._spec = copy.deepcopy(self._spec)
        return target

    def detach(self) -> "LazyTensor":
        target = self.clone()
        target._cached_buffer.append((torch.Tensor.detach_, (self,), {}))
        return target

    @property
    def spec(self) -> ShardingSpec:
        return self._spec

    @spec.setter
    def spec(self, other: ShardingSpec):
        self._spec = other

    @property
    def data(self) -> "LazyTensor":
        return self._data.detach()

    @data.setter
    def data(self, other: "LazyTensor") -> "LazyTensor":
        """This avoid the following infinite recursion, which is very common in ``nn.Module`` initialization.

        Usage:
            >>> a = LazyTensor(torch.empty, 0, dtype=torch.float32, device='cpu')
            >>> b = a.cuda()
            >>> a.data = b
        """
        self._data = other


class LazyInitContext():
    """Context manager for lazy initialization. Enables initializing the model without allocating real memory.

    Usage:
        1. The model is initialized, but no real memory is allocated.
        >>> ctx = LazyInitContext()
        >>> with ctx:
        >>>     model = MyModel().cuda()

        2. The model is initialized with ``MetaTensor`` as weights, but still no real memory is allocated.
        >>> with ctx.traceable(model):
        >>>     gm = symbolic_trace(model, meta_args=meta_args)
        >>> # Solve the execution strategy and apply the strategy to the model
        >>> strategy = StrategyAndSpec()

        3. The model is initialized with ``torch.Tensor`` as weights, and real memory is allocated. (single device)
        >>> model = ctx.materialize(model)

        3. The model is initialized with sharded ``torch.Tensor`` as weights, and real memory is allocated. (distributed scenario)
        >>> model = apply_strategy_to_all_params(model, strategy)
        >>> model = ctx.distribute(model)

    Warnings:
        This API is still experimental and further modifications can be made to it.
        For example:
            1. Quantization strategies can be applied before allocating real memory.
            2. Lazy initialization seems slower than normal initialization.
    """

    def __init__(self):
        self.overrides = {}

    def __enter__(self):

        def wrap_factory_method(target):
            # factory functions (eg. torch.empty())
            def wrapper(*args, **kwargs):
                return LazyTensor(target, *args, **kwargs)

            return wrapper, target

        def wrap_factory_like_method(orig_target, target):
            # factory_like functions (eg. torch.empty_like())
            def wrapper(*args, **kwargs):
                orig_t = args[0]
                return LazyTensor(orig_target, *args[1:], device=orig_t.device, dtype=orig_t.dtype, **kwargs)

            return wrapper, target

        self.overrides = {
            target: wrap_factory_method(getattr(torch, target))
            for target in _TorchFactoryMethod
            if callable(getattr(torch, target, None))
        }

        self.overrides.update({
            target + '_like': wrap_factory_like_method(getattr(torch, target), getattr(torch, target + '_like'))
            for target in _TorchFactoryMethod
            if callable(getattr(torch, target + '_like', None))
        })

        for name, (wrapper, orig) in self.overrides.items():
            setattr(torch, name, wrapper)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, (wrapper, orig) in self.overrides.items():
            setattr(torch, name, orig)

    @staticmethod
    def materialize(module: torch.nn.Module):
        """Initialize all ``nn.Parameter`` from ``LazyTensor``.

        Args:
            module (torch.nn.Module): Target ``nn.Module``
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
    def distribute(module: torch.nn.Module):
        """Initialize and shard all ``nn.Parameter`` from ``LazyTensor``.

        Args:
            module (torch.nn.Module): Sharded target ``nn.Module``
        """

        @torch.no_grad()
        def init_recursively(module: nn.Module):
            # recursively initialize the module
            for mod in module.children():
                init_recursively(mod)

            # initialize tensors directly attached to the current module
            for name, param in module.named_parameters(recurse=False):
                setattr(module, name, param.distribute())

            for name, buf in module.named_buffers(recurse=False):
                setattr(module, name, buf.distribute())

        init_recursively(module)
        return module

    @staticmethod
    @contextlib.contextmanager
    def traceable(module: torch.nn.Module):
        """Initialize all ``nn.Parameters`` as ``MetaTensor``. This enables ``ColoTracer`` with control flow.

        Args:
            module (torch.nn.Module): Traceable ``nn.Module`` with ``MetaTensor`` as parameters.
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
