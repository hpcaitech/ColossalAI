import torch
from torch.utils._pytree import tree_map, tree_flatten

__all__ = ['MetaTensor']


class MetaTensor(torch.Tensor):
    """
    A wrapping tensor that hacks `torch.autograd` without patching more `torch.ops.aten` ops.
    `fake_device` is the device that `MetaTensor` is supposed to run on.
    """

    _tensor: torch.Tensor

    __slots__ = ['_tensor']

    @staticmethod
    def __new__(cls, elem, fake_device=None):
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
            device=fake_device if fake_device is not None else elem.device,
            requires_grad=elem.requires_grad)    # deceive the frontend for aten selections
        r._tensor = elem
        # ...the real tensor is held as an element on the tensor.
        if not r._tensor.is_meta:
            r._tensor = r._tensor.to(torch.device('meta'))
        # only tensor not on `meta` should be copied to `meta`
        return r

    def __repr__(self):
        if self.grad_fn:
            return f"MetaTensor({self._tensor}, fake_device='{self.device}', grad_fn={self.grad_fn})"
        return f"MetaTensor({self._tensor}, fake_device='{self.device}')"

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

        if 'device' in kwargs:
            fake_device = kwargs['device']
            kwargs['device'] = torch.device('meta')

        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs)

        # run aten for backend=CPU but actually on backend=Meta
        out = func(*args, **kwargs)

        # Now, we want to continue propagating this tensor, so we rewrap Tensors in
        # our custom tensor subclass
        def wrap(x):
            if isinstance(x, torch.Tensor):
                nonlocal fake_device
                if not x.is_meta:
                    x = x.to(torch.device('meta'))
            return MetaTensor(x, fake_device=fake_device) if isinstance(x, torch.Tensor) else x

        return tree_map(wrap, out)
