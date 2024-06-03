import uuid
from functools import partial

import torch
import torch.distributed as dist
from torch.types import _device
from torch.utils._pytree import tree_map

from ._monkey_patch import _AliasATen, _DistCommMethod, _InplaceATen, _MaybeInplaceATen, _TorchOverrideableFactoryMethod

__all__ = ["MetaTensor", "MetaTensorMode"]


def register_storage(r, data_ptr_fn=None):
    if isinstance(r, torch.Tensor):
        if data_ptr_fn is not None:
            r.data_ptr = data_ptr_fn
        elif not r.data_ptr():
            data_ptr = uuid.uuid1()
            r.data_ptr = lambda: data_ptr


def _normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


# a hack of inplace execution in PyTorch
def _assert_alias(func):
    return func in (_AliasATen + _InplaceATen + _MaybeInplaceATen)  # TODO: check if should be this aggressive


class MetaTensor(torch.Tensor):
    """
    A wrapping tensor that hacks ``torch.autograd`` without patching more ``torch.ops.aten`` ops.
    `device` is the device that ``MetaTensor`` is supposed to run on. Meta tensors give you the
    ability to run PyTorch code without having to actually do computation through tensors
    allocated on a `meta` device. Because the device is `meta`, meta tensors do not model
    device propagation. ``MetaTensor`` extends its usage by carrying an additional `device`
    which tracks devices that would have been used.

    Reference:
        https://github.com/pytorch/pytorch/blob/master/torch/_subclasses/fake_tensor.py
    """

    _tensor: torch.Tensor

    @staticmethod
    def __new__(cls, elem, device=None, data_ptr_fn=None):
        requires_grad = elem.requires_grad
        # Avoid multiple wrapping
        while isinstance(elem, MetaTensor):
            device = elem.device if device is None else device
            elem = elem._tensor

        # The wrapping tensor (MetaTensor) shouldn't hold any
        # memory for the class in question, but it should still
        # advertise the same device as before
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            dtype=elem.dtype,
            layout=elem.layout,
            device=device or (elem.device if elem.device.type != "meta" else torch.device("cpu")),
            requires_grad=requires_grad,
        )  # deceive the frontend for aten selections
        r._tensor = elem
        # ...the real tensor is held as an element on the tensor.
        if not r._tensor.is_meta:
            val = elem.data_ptr()
            data_ptr_fn = lambda: val
            r._tensor = r._tensor.to(torch.device("meta"))

        # only tensor not on `meta` should be copied to `meta`
        register_storage(r._tensor, data_ptr_fn)
        if isinstance(elem, torch.nn.Parameter):
            r = torch.nn.Parameter(r)
        return r

    def __repr__(self):
        name = "MetaParameter" if getattr(self, "_is_param", False) else "MetaTensor"
        if self.grad_fn:
            return f"{name}(..., size={tuple(self.shape)}, device='{self.device}', dtype={self.dtype}, grad_fn={self.grad_fn})"
        return f"{name}(..., size={tuple(self.shape)}, device='{self.device}', dtype={self.dtype})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        device = None

        def unwrap(x):
            nonlocal device
            if isinstance(x, MetaTensor):
                device = x.device
                x = x._tensor
            elif isinstance(x, torch.Tensor):
                device = x.device
                x = x.to(torch.device("meta"))
            return x

        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs)

        if "device" in kwargs:
            device = kwargs["device"]
            kwargs["device"] = torch.device("meta")

        # run aten for backend=CPU but actually on backend=Meta
        # here we detect whether or not the execution generates a physical copy
        # of the input tensor
        ret = func(*args, **kwargs)

        if _assert_alias(func):
            val = args[0].data_ptr()
            tree_map(partial(register_storage, data_ptr_fn=lambda: val), _normalize_tuple(ret))

        # Now, we want to continue propagating this tensor, so we rewrap Tensors in
        # our custom tensor subclass
        def wrap(x):
            return MetaTensor(x, device=device) if isinstance(x, torch.Tensor) else x

        return tree_map(wrap, ret)

    def to(self, *args, **kwargs) -> torch.Tensor:
        """An extension of `torch.Tensor.to()` to MetaTensor
        Returns:
            result (MetaTensor): MetaTensor
        Usage:
            >>> tensor = MetaTensor(torch.rand(10), device='cuda:100')
            >>> tensor.to(torch.uint8)
            MetaTensor(tensor(..., device='meta', size=(10,), dtype=torch.uint8), device='cuda:100')
            >>> tensor.to(torch.device('cuda:42'))
            MetaTensor(tensor(..., device='meta', size=(10,)), device='cuda:42')
            >>> tensor.to('vulkan')
            MetaTensor(tensor(..., device='meta', size=(10,)), device='vulkan')
        """
        # this imitates c++ function in the way of @overload
        device = None

        def replace(x):
            nonlocal device
            if isinstance(x, str) or isinstance(x, _device):
                device = x
                return torch.device("meta")
            return x

        elem = self._tensor.to(*tree_map(replace, args), **tree_map(replace, kwargs))
        return MetaTensor(elem, device=device)

    def cpu(self, *args, **kwargs):
        if self.device.type == "cpu":
            return self.to(*args, **kwargs)
        return self.to(*args, device="cpu", **kwargs)

    def cuda(self, device=None, non_blocking=False):
        if device is not None:
            return self.to(device=device, non_blocking=non_blocking)
        return self.to(device="cuda:0", non_blocking=non_blocking)

    def data_ptr(self):
        return self._tensor.data_ptr()


class MetaTensorMode(object):
    """
    A context manager that enables MetaTensor mode.

    Usage:
        >>> with MetaTensorMode():
        >>>     # all torch.xxx and torch.distributed.xxx will be replaced by patched functions
        >>>     # and the actual execution will be on torch.device('meta')
        >>>     a = torch.rand(100000, 100000)
        >>>     b = torch.rand(100000, 100000)
        >>>     c = torch.mm(a, b)
    """

    def __init__(self):
        self.torch_overrides = {}  # override torch.xxx
        self.dist_overrides = {}  # override torch.distributed.xxx

    def __enter__(self):
        def _dummy(*args, **kwargs):
            pass

        def _new(*args, orig_new=torch.empty, **kwargs):
            return MetaTensor(
                orig_new(*args, **{**kwargs, "device": "meta"}), device=kwargs.get("device", torch.device("cpu"))
            )

        for func in _TorchOverrideableFactoryMethod:
            self.torch_overrides[func] = getattr(torch, func)
            setattr(torch, func, partial(_new, orig_new=getattr(torch, func)))

        for func in _DistCommMethod:
            self.dist_overrides[func] = getattr(dist, func)
            setattr(dist, func, _dummy)

    def __exit__(self, exc_type, exc_value, traceback):
        for func, func_impl in self.torch_overrides.items():
            setattr(torch, func, func_impl)

        for func, func_impl in self.dist_overrides.items():
            setattr(dist, func, func_impl)
