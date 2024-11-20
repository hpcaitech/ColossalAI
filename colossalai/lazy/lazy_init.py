from types import MethodType
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from packaging import version
from torch import Tensor
from torch.nn import Parameter
from torch.utils._pytree import tree_map

from colossalai.logging import get_dist_logger

from .construction import ConstructorManager
from .pretrained import PretrainedManager

import colossalai._analyzer._subclasses._meta_registration  # noqa

# reference: https://pytorch.org/cppdocs/notes/tensor_creation.html
_NORMAL_FACTORY = [
    "arange",
    "full",
    "empty",
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

# factory function that does not support meta tensor backend
_NO_META_FACTORY = [
    "eye",
]

_EARLY_MATERIALIZED_OPS = ["__getitem__", "split"]

# If your intent is to change the metadata of a Tensor (such as sizes / strides / storage / storage_offset)
# without autograd tracking the change, remove the .data / .detach() call and wrap the change in a `with torch.no_grad():` block.
# These ops cannot be unwrapped using .data
_CHANGE_META_OPS = ["_cudnn_rnn_flatten_weight", "requires_grad_", "__get__", "__set__", "numel", "size", "dim"]

# These ops is not related to tensor value and should not be rerun
_NO_RERUN_OPS = ["__get__", "numel", "size", "dim"]

_LEGACY_TENSOR_CONSTRUCTOR = {
    "FloatTensor": torch.float,
    "DoubleTensor": torch.double,
    "HalfTensor": torch.half,
    "BFloat16Tensor": torch.bfloat16,
    "ByteTensor": torch.uint8,
    "CharTensor": torch.int8,
    "ShortTensor": torch.short,
    "IntTensor": torch.int,
    "LongTensor": torch.long,
    "BoolTensor": torch.bool,
}

# These ops have at least one lazy tensor argument and maybe a scalar argument
# scalar value should be converted to meta tensor
# this is a hack for torch 2.0
_EXPAND_SCALAR_OPS = [
    "where",
    "clamp",
    "clamp_min",
    "clamp_max",
    "clamp_",
    "clamp_min_",
    "clamp_max_",
]
_old_tensor_factory = torch.tensor

_EMPTY_DATA = torch.empty(0)


class _MyTensor(Tensor):
    """This class is only for correctness verification."""

    _pre_op_fn: Callable[["LazyTensor"], None] = lambda *args: None

    default_device: Optional[torch.device] = None

    def __new__(cls, func, *args, concrete_data=None, **kwargs) -> "_MyTensor":
        cls._pre_op_fn()
        if concrete_data is not None:
            # uniform api as LazyTensor
            data = concrete_data
        else:
            kwargs["device"] = cls.default_device
            data = func(*args, **kwargs)
        return Tensor._make_subclass(cls, data, require_grad=data.requires_grad)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        cls._pre_op_fn()
        return super().__torch_function__(func, types, args, kwargs)


def _data_tolist(tensor: torch.Tensor) -> list:
    """tolist() method is not allowed for a subclass of tensor. Tensor.data returns a Tensor."""
    return tensor.data.tolist()


def _convert_cls(tensor: "LazyTensor", target: torch.Tensor) -> torch.Tensor:
    """Convert a lazy tensor's class to target's class, with target's data.

    The reason why we change the class of a lazy tensor in-place is that this can easily handle shared modules/parameters, which is common in huggingface models.
    If we create a new tensor and update the module by ``setattr(module, name, param)``, the shared parameters will not be updated. And we have to track all shared parameters and update them manually.

    Args:
        tensor (LazyTensor): the LazyTensor to be converted
        target (torch.Tensor): target tensor

    Returns:
        torch.Tensor: the converted tensor
    """
    cls_to_become = Parameter if isinstance(tensor, Parameter) else torch.Tensor
    tensor.__class__ = cls_to_become
    if cls_to_become is Parameter:
        # to fit UninitializedParameter
        delattr(tensor, "_is_param")
    tensor.data = target
    tensor.requires_grad = target.requires_grad
    # subclass of torch.Tensor does not have tolist() method
    # overwrite this method after materialization or distribution
    tensor.tolist = MethodType(_data_tolist, tensor)
    return tensor


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

    Warnings:
        1. Cases that ``LazyTensor`` can't deal with.
        >>> x = LazyTensor(torch.ones, 2, 3)
        >>> x[0, 0] = -x[0, 0]    # this will cause infinite recursion
        >>> y = x.clone()
        >>> x.add_(1) # modifying origin tensor after cloning leads to wrong materialization
        >>> z = x.tolist()
        >>> x.zeros_() # modifying origin tensor after cloning tolist is not allowed
        >>> nn.utils.weight_norm(self.conv, name="weight", dim=2) # applying weight norm on a lazy tensor is not allowed


        2. Cases that ``LazyTensor`` becomes eager (early materialization).
        >>> b = a[:, 2:]  # get a slice of a lazy tensor triggers early materialization
        >>> chunks = a.split(3)  # this also triggers early materialization
        >>> x.data = torch.rand(2, 3) # directly setting data of a lazy tensor triggers early materialization

    """

    _repr = True
    _meta_data: Optional[torch.Tensor] = None  # shape, dtype, device
    _pre_op_fn: Callable[["LazyTensor"], None] = lambda *args: None

    default_device: Optional[torch.device] = None
    _device: torch.device  # fake device of mate tensor

    @staticmethod
    def __new__(cls, func, *args, meta_data=None, concrete_data=None, **kwargs):
        # tips for torch 2.0:
        # torch 2.0 disables torch dispatch for subclass of tensor
        # MetaTensor is cannot be used
        # Now lazy tensor contains device injection and meta tensor
        if concrete_data is not None:
            # some ops don't support meta backend and should have concrete data
            elem = concrete_data
        else:
            if meta_data is None:
                with ConstructorManager.disable():
                    # to disable create lazy tensor in inner ops, this is a hack for torch 2.0
                    meta_data = func(*args, **{**kwargs, "device": "meta"})
            elem = meta_data
        # As a meta tensor cannot be modified __class__ to torch.Tensor, we should use an empty real tensor here
        r = torch.Tensor._make_subclass(cls, _EMPTY_DATA, require_grad=elem.requires_grad)
        r._meta_data = meta_data

        return r

    def __init__(self, func, *args, meta_data=None, concrete_data=None, **kwargs):
        self._device = torch.device(kwargs.get("device", None) or "cpu")
        if func.__name__ in _NORMAL_FACTORY:
            kwargs = {**kwargs, "device": LazyTensor.default_device}
        self._factory_method = (func, args, kwargs)  # (func, args, kwargs)
        self._op_buffer = []  # (func, args, kwargs, replace)
        self._materialized_data: Optional[torch.Tensor] = concrete_data  # materialized data

    @property
    def device(self) -> torch.device:
        return self._materialized_data.device if self._materialized_data is not None else self._device

    def __repr__(self):
        return f"LazyTensor(..., size={tuple(self.shape)}, device='{self.device}', dtype={self.dtype})"

    def materialize(self) -> torch.Tensor:
        """Materialize the ``LazyTensor`` to ``torch.Tensor`` by modifying __class__ (inplace).

        Returns:
            torch.Tensor: The materialized tensor (self).
        """
        target = self._materialize_data()
        self.clean()
        return _convert_cls(self, target)

    def clean(self) -> None:
        """Clean all stored operations, meta data and materialized data, which prevents memory leaking. This should be called after all tensors are materialized."""
        delattr(self, "_factory_method")
        delattr(self, "_op_buffer")
        delattr(self, "_materialized_data")
        delattr(self, "_meta_data")

    @staticmethod
    def _replace_with_materialized(x):
        if isinstance(x, LazyTensor):
            return x._materialize_data()
        return x

    def _materialize_data(self) -> torch.Tensor:
        # self._materialized_data should be generated after the first call of this function
        if self._materialized_data is None:
            # apply factory method
            func, args, kwargs = self._factory_method
            # apply cached sequence
            self._pre_op_fn()

            init_val = func(
                *tree_map(self._replace_with_materialized, args), **tree_map(self._replace_with_materialized, kwargs)
            )

            self._materialized_data = self._rerun_ops(init_val)
        return self._materialized_data

    def _rerun_ops(self, target=None) -> torch.Tensor:
        """Do lazy execution by rerunning all (stored) related operations.

        Args:
            target (torc.Tensor, optional): Intial value of the target tensor (self). Defaults to None.
        """

        def replace(x):
            if x is self:
                return target
            elif isinstance(x, LazyTensor):
                return x._materialize_data()
            return x

        packed = None

        for func, args, kwargs in self._op_buffer:
            if func == torch.Tensor.requires_grad_:
                packed = func, args, kwargs  # requires grad should be set at last
            else:
                self._pre_op_fn()
                o = func(*tree_map(replace, args), **tree_map(replace, kwargs))
                target = o if isinstance(o, torch.Tensor) else target  # if func returns non-Tensor, discard the value

        # super-dainiu: set requires_grad after all inplace-ops are done
        if packed is not None:
            func, args, kwargs = packed
            func(*tree_map(replace, args), **tree_map(replace, kwargs))

        return target

    # cache everything with __torch_function__

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func.__name__ in _EARLY_MATERIALIZED_OPS:
            # These OPs cannot be lazy and related tensors should be early materialized
            tree_map(cls._replace_with_materialized, args)
            tree_map(cls._replace_with_materialized, kwargs)
        is_inplace: bool = (
            func.__name__.endswith("_")
            and not (func.__name__.endswith("__"))
            or func.__name__ in ("__setitem__", "__set__")
        )

        is_change_meta_op: bool = func.__name__ in _CHANGE_META_OPS

        if isinstance(func, torch._C.ScriptMethod):
            # FIXME(ver217): torch script functions are not verified

            target = None

            def unwrap(x):
                if isinstance(x, LazyTensor):
                    return x._meta_data
                return x

            target: LazyTensor = args[0].clone()
            target._op_buffer.append((func, args, kwargs))
            target._meta_data = getattr(target._meta_data, func.name)(
                *tree_map(unwrap, args[1:]), **tree_map(unwrap, kwargs)
            )
            return target
        else:
            meta_to_lazy = {}

            def unwrap(x):
                if isinstance(x, LazyTensor):
                    if x._materialized_data is not None:
                        # for early materialized tensor, use its materialized data directly
                        return x._materialized_data if is_change_meta_op else x._materialized_data.data
                    t = x if is_inplace else x.clone()
                    if func.__name__ not in _NO_RERUN_OPS:
                        t._op_buffer.append((func, args, kwargs))
                    meta = x._meta_data if is_change_meta_op else x._meta_data.data
                    meta_to_lazy[meta] = t
                    return meta
                elif (
                    version.parse(torch.__version__) >= version.parse("2.0.0")
                    and func.__name__ in _EXPAND_SCALAR_OPS
                    and not isinstance(x, torch.Tensor)
                ):
                    return _old_tensor_factory(x, device="meta")
                return x

            def wrap(y, i=None):
                if isinstance(y, torch.Tensor):
                    if y.is_meta:
                        if y in meta_to_lazy:
                            # inplace op, just return origin lazy tensor
                            return meta_to_lazy[y]
                        else:
                            # out of place op, create new lazy tensor
                            fn = lambda *a, **kw: func(*a, **kw) if i is None else func(*a, **kw)[i]
                            fn.__name__ = func.__name__
                            lazy_y = LazyTensor(fn, *args, meta_data=y, **kwargs)
                            return lazy_y
                    else:
                        # for early materialized tensor
                        return LazyTensor(lambda: None, concrete_data=y)
                return y

            cls._pre_op_fn()
            with ConstructorManager.disable():
                # to disable create lazy tensor in inner ops, this is a hack for torch 2.0
                o = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
            if isinstance(o, (tuple, list)):
                return type(o)(wrap(y, i=i) for i, y in enumerate(o))
            return wrap(o)

    def to(self, *args, **kwargs) -> torch.Tensor:
        if self._materialized_data is not None:
            return LazyTensor(lambda: None, concrete_data=self._materialized_data.to(*args, **kwargs))

        device = None

        def replace(x):
            nonlocal device
            if isinstance(x, (str, int, torch.device)) and not isinstance(x, bool):
                device = x
                return torch.device("meta")
            return x

        meta_data = self._meta_data.to(*tree_map(replace, args), **tree_map(replace, kwargs))

        if meta_data is self._meta_data and device == self.device:
            return self

        def factory_fn(t: torch.Tensor, **kw):
            return t.to(*args, **kwargs)

        return LazyTensor(factory_fn, self, meta_data=meta_data, device=device)

    def cpu(self, memory_format: torch.memory_format = torch.preserve_format):
        return self.to(device=torch.device("cpu"), memory_format=memory_format)

    def cuda(self, device=None, non_blocking=False, memory_format: torch.memory_format = torch.preserve_format):
        device = torch.device(device or "cuda")
        return self.to(device=device, non_blocking=non_blocking, memory_format=memory_format)

    def clone(self) -> "LazyTensor":
        def factory_fn(t: torch.Tensor, **kw):
            # if self is materialized, return self
            return t.clone()

        target = LazyTensor(factory_fn, self, meta_data=self._meta_data)

        return target

    def detach(self) -> Tensor:
        return self

    def __deepcopy__(self, memo):
        if not self.is_leaf:
            raise RuntimeError(
                "Only Tensors created explicitly by the user "
                "(graph leaves) support the deepcopy protocol at the moment"
            )
        if id(self) in memo:
            return memo[id(self)]

        def factory_fn(t: torch.Tensor, **kw):
            # if self is materialized, return self
            return _copy_tensor(t, t.requires_grad)

        if self._materialized_data is not None:
            # self is early materialized
            copied = _copy_tensor(self._materialized_data, self.requires_grad)
            target = LazyTensor(lambda: None, concrete_data=copied)
        else:
            target = LazyTensor(factory_fn, self, meta_data=self._meta_data)

        if isinstance(self, Parameter):
            # hack isinstance check of parameter
            target._is_param = True

        memo[id(self)] = target
        return target

    @property
    def data(self):
        return self

    @data.setter
    def data(self, other: "LazyTensor"):
        """This is sightly different from oringinal `data` setter.

        E.g.:
            >>> a = torch.randn(3, 3) # a is a Tensor
            >>> b = torch.rand(2, 2)
            >>> a.data = b
            >>> b.add_(1)   # this will affect a
            >>> x = torch.randn(3, 3) # x is a LazyTensor
            >>> y = torch.rand(2, 2) # y is a LazyTensor
            >>> x.data = y
            >>> y.add_(1)   # this will not affect x

        """
        if other is self:
            return

        def replace(x):
            if x is other:
                return self
            return x

        for func, args, kwargs in [other._factory_method, *other._op_buffer]:
            self._op_buffer.append((func, tree_map(replace, args), tree_map(replace, kwargs)))

    def tolist(self) -> list:
        # Though self.__class__ is modified to torch.Tensor, in C++ side, it is still a subclass of torch.Tensor
        # And subclass of torch.Tensor does not have tolist() method
        t = self._materialize_data()
        return t.tolist()

    def __hash__(self):
        return id(self)

    def __rpow__(self, other):
        dtype = torch.result_type(self, other)
        return torch.tensor(other, dtype=dtype, device=self.device) ** self


class LazyInitContext:
    """Context manager for lazy initialization. Enables initializing the model without allocating real memory.

    Args:
        tensor_cls (Union[_MyTensor, LazyTensor], optional): This is only for test. Defaults to LazyTensor.
        default_device (Optional[Union[torch.device, str, int]], optional): Defalt device for initialization.
            If it's cuda, initilization will be accelerated, but cuda memory will be allocated. By default, it's cpu.
            Defaults to None.
    """

    _replaced: bool = False

    def __init__(
        self,
        tensor_cls: Union[_MyTensor, LazyTensor] = LazyTensor,
        default_device: Optional[Union[torch.device, str, int]] = None,
    ):
        assert tensor_cls is LazyTensor or tensor_cls is _MyTensor
        self.tensor_cls = tensor_cls
        self.old_default_device = LazyTensor.default_device
        self.default_device = default_device

    def __enter__(self):
        if LazyInitContext._replaced:
            raise RuntimeError(f"LazyInitContext is not reentrant")
        LazyInitContext._replaced = True
        self.old_default_device = self.tensor_cls.default_device
        self.tensor_cls.default_device = self.default_device

        def wrap_factory_method(target):
            # factory functions (eg. torch.empty())
            def wrapper(*args, **kwargs):
                return self.tensor_cls(target, *args, **kwargs)

            return wrapper, target

        def wrap_factory_like_method(orig_target, target):
            # factory_like functions (eg. torch.empty_like())
            def wrapper(*args, **kwargs):
                orig_t = args[0]
                device = kwargs.pop("device", orig_t.device)
                dtype = kwargs.pop("dtype", orig_t.dtype)
                return self.tensor_cls(orig_target, *orig_t.shape, *args[1:], device=device, dtype=dtype, **kwargs)

            return wrapper, target

        def wrap_legacy_constructor(target, dtype):
            # legacy constructor (e.g. torch.LongTensor())
            def wrapper(*args, **kwargs):
                if len(args) == 1 and isinstance(args[0], torch.Tensor):
                    # (Tensor other)
                    return args[0]
                elif len(args) == 1:
                    # (object data, *, torch.device device)
                    kwargs = {**kwargs, "dtype": dtype}
                    replaced, orig = self.overrides["tensor"]
                    return replaced(*args, **kwargs)
                elif _is_int_tuple(args):
                    # (tuple of ints size, *, torch.device device)
                    kwargs = {**kwargs, "dtype": dtype}
                    replaced, orig = self.overrides["empty"]
                    return replaced(*args, **kwargs)
                else:
                    raise TypeError(
                        f"new() received an invalid combination of arguments - got {tuple(type(x) for x in args)}, but expected one of:\n * (Tensor other)\n * (tuple of ints size, *, torch.device device)\n * (object data, *, torch.device device)"
                    )

            return wrapper, target

        def wrap_no_meta_factory(target):
            # factory functions which don't support meta tensor backend
            def wrapper(*args, **kwargs):
                tensor = target(*args, **kwargs)
                return self.tensor_cls(lambda: None, concrete_data=tensor)

            return wrapper, target

        overrides = {
            target: wrap_factory_method(getattr(torch, target))
            for target in _NORMAL_FACTORY
            if callable(getattr(torch, target, None))
        }

        overrides.update(
            {
                target + "_like": wrap_factory_like_method(getattr(torch, target), getattr(torch, target + "_like"))
                for target in _NORMAL_FACTORY
                if callable(getattr(torch, target + "_like", None))
            }
        )

        overrides.update(
            {
                target: wrap_legacy_constructor(getattr(torch, target), dtype)
                for target, dtype in _LEGACY_TENSOR_CONSTRUCTOR.items()
                if callable(getattr(torch, target, None))
            }
        )

        overrides.update(
            {
                target: wrap_no_meta_factory(getattr(torch, target))
                for target in _NO_META_FACTORY
                if callable(getattr(torch, target, None))
            }
        )

        ConstructorManager.apply(overrides)
        PretrainedManager.inject()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tensor_cls.default_device = self.old_default_device
        LazyInitContext._replaced = False
        ConstructorManager.clear()
        PretrainedManager.recover()

    @staticmethod
    def materialize(module: nn.Module, verbose: bool = False) -> nn.Module:
        """Initialize all ``Parameter`` from ``LazyTensor``. This function will modify the module in-place.

        Args:
            module (nn.Module): Target ``nn.Module``
            verbose (bool): Whether to print lazy initialization rate. Defaults to False.
        """

        def apply_fn(name: str, p: LazyTensor):
            p.materialize()

        return _apply_to_lazy_module(module, apply_fn, verbose)


def _apply_to_lazy_module(
    module: nn.Module, apply_fn: Callable[[str, torch.Tensor], None], verbose: bool = False
) -> nn.Module:
    if verbose:
        # verbose info
        param_cnt = 0
        param_lazy_cnt = 0
        buf_cnt = 0
        buf_lazy_cnt = 0
        total_numel = 0
        non_lazy_numel = 0

    for name, p in module.named_parameters():
        if verbose:
            param_cnt += 1
            total_numel += p.numel()
            if getattr(p, "_materialized_data", False) is None:
                # if no _materialized_data attr, the tensor is not lazy
                param_lazy_cnt += 1
            else:
                non_lazy_numel += p.numel()
        if isinstance(p, LazyTensor):
            apply_fn(name, p)

    for name, buf in module.named_buffers():
        if verbose:
            buf_cnt += 1
            total_numel += buf.numel()
            if getattr(buf, "_materialized_data", False) is None:
                # if no _materialized_data attr, the tensor is not lazy
                buf_lazy_cnt += 1
            else:
                non_lazy_numel += buf.numel()
        if isinstance(buf, LazyTensor):
            apply_fn(name, buf)

    if verbose:
        non_lazy_numel_ratio = non_lazy_numel / total_numel * 100 if non_lazy_numel != 0 else 0
        logger = get_dist_logger()
        logger.info(f"Param lazy rate: {param_lazy_cnt}/{param_cnt}", ranks=[0])
        logger.info(f"Buffer lazy rate: {buf_lazy_cnt}/{buf_cnt}", ranks=[0])
        logger.info(
            f"Non lazy numel: {non_lazy_numel} ({non_lazy_numel/1024**2:.3f} M), ratio: {non_lazy_numel_ratio}%",
            ranks=[0],
        )

    return module


def _is_int_tuple(args) -> bool:
    if not isinstance(args, tuple):
        return False
    for x in args:
        if not isinstance(x, int):
            return False
    return True


def _copy_tensor(tensor: Tensor, requires_grad: bool) -> Tensor:
    copied = tensor.data.clone()
    copied.requires_grad = requires_grad
    return copied
