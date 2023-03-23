import uuid

import torch
from torch.types import _bool, _device, _dtype
from torch.utils._pytree import tree_flatten, tree_map

from .._compatibility import compatibility
from .constants import ALIAS_ATEN

__all__ = ['MetaTensor']


def set_data_ptr(x):
    if isinstance(x, torch.Tensor):
        if not x.data_ptr():
            data_ptr = uuid.uuid4()
            x.data_ptr = lambda: data_ptr


@compatibility(is_backward_compatible=False)
class MetaTensor(torch.Tensor):
    """
    A wrapping tensor that hacks `torch.autograd` without patching more `torch.ops.aten` ops.
    `fake_device` is the device that `MetaTensor` is supposed to run on.
    """

    _tensor: torch.Tensor

    @staticmethod
    def __new__(cls, elem, fake_device=None):
        # Avoid multiple wrapping
        if isinstance(elem, MetaTensor):
            fake_device = elem.device if fake_device is None else fake_device
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
            device=fake_device or (elem.device if elem.device.type != 'meta' else torch.device('cpu')),
            requires_grad=elem.requires_grad)    # deceive the frontend for aten selections
        r._tensor = elem
        # ...the real tensor is held as an element on the tensor.
        if not r._tensor.is_meta:
            r._tensor = r._tensor.to(torch.device('meta'))
        # only tensor not on `meta` should be copied to `meta`
        set_data_ptr(r._tensor)
        return r

    def __repr__(self):
        if self.grad_fn:
            return f"MetaTensor(..., size={tuple(self.shape)}, device='{self.device}', dtype={self.dtype}, grad_fn={self.grad_fn})"
        return f"MetaTensor(..., size={tuple(self.shape)}, device='{self.device}', dtype={self.dtype})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        fake_device = None

        def unwrap(x):
            nonlocal fake_device
            if isinstance(x, MetaTensor):
                fake_device = x.device
                x = x._tensor
            elif isinstance(x, torch.Tensor):
                fake_device = x.device
                x = x.to(torch.device('meta'))
            return x

        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs)

        if 'device' in kwargs:
            fake_device = kwargs['device']
            kwargs['device'] = torch.device('meta')

        # run aten for backend=CPU but actually on backend=Meta
        out = func(*args, **kwargs)

        # here we keep the uuid of input because ALIAS_ATEN do not generate a physical copy
        # of the input
        if func in ALIAS_ATEN:
            out.data_ptr = args[0].data_ptr

        # Now, we want to continue propagating this tensor, so we rewrap Tensors in
        # our custom tensor subclass
        def wrap(x):
            if isinstance(x, torch.Tensor):
                nonlocal fake_device
                if not x.is_meta:
                    x = x.to(torch.device('meta'))
            return MetaTensor(x, fake_device=fake_device) if isinstance(x, torch.Tensor) else x

        return tree_map(wrap, out)

    def to(self, *args, **kwargs) -> torch.Tensor:
        """An extension of `torch.Tensor.to()` to MetaTensor

        Returns:
            result (MetaTensor): MetaTensor

        Usage:
            >>> tensor = MetaTensor(torch.rand(10), fake_device='cuda:100')
            >>> tensor.to(torch.uint8)
            MetaTensor(tensor(..., device='meta', size=(10,), dtype=torch.uint8), fake_device='cuda:100')
            >>> tensor.to(torch.device('cuda:42'))
            MetaTensor(tensor(..., device='meta', size=(10,)), fake_device='cuda:42')
            >>> tensor.to('vulkan')
            MetaTensor(tensor(..., device='meta', size=(10,)), fake_device='vulkan')
        """
        # this imitates c++ function in the way of @overload
        fake_device = None

        def replace(x):
            nonlocal fake_device
            if isinstance(x, str) or isinstance(x, _device):
                fake_device = x
                return 'meta'
            return x

        elem = self._tensor.to(*tree_map(replace, args), **tree_map(replace, kwargs))
        return MetaTensor(elem, fake_device=fake_device)

    def cpu(self, *args, **kwargs):
        if self.device.type == 'cpu':
            return self.to(*args, **kwargs)
        return self.to(*args, device='cpu', **kwargs)

    def cuda(self, device=None, non_blocking=False):
        if device is not None:
            return self.to(device=device, non_blocking=non_blocking)
        return self.to(device='cuda:0', non_blocking=non_blocking)
