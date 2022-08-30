import torch
from torch.utils._pytree import tree_map, tree_flatten

class MetaTensor(torch.Tensor):
    """
    A wrapping tensor that hacks `torch.autograd` without patching more `torch.ops.aten` ops.
    """

    elem: torch.Tensor
 
    __slots__ = ['elem']
 
    @staticmethod
    def __new__(cls, elem):
        # The wrapping tensor (MetaTensor) shouldn't hold any
        # memory for the class in question, but it should still
        # advertise the same device as before
        r = torch.Tensor._make_wrapper_subclass(
            cls, elem.size(),
            strides=elem.stride(), storage_offset=elem.storage_offset(),
            dtype=elem.dtype, layout=elem.layout,
            device='cpu', requires_grad=elem.requires_grad
        )    # deceive the frontend for aten selections
        r.elem = elem
        # ...the real tensor is held as an element on the tensor.
        return r

    @ classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(x):
            if isinstance(x, torch.Tensor) and not hasattr(x, 'elem'):
                x = MetaTensor(x)
            return x.elem.to('meta') if isinstance(x, MetaTensor) else x
        
        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs)

        # run aten for backend=CPU but actually on backend=Meta
        out = func(*args, **kwargs)
        
        # Now, we want to continue propagating this tensor, so we rewrap Tensors in
        # our custom tensor subclass
        def wrap(x):
            return MetaTensor(x) if isinstance(x, torch.Tensor) else x
           
        return tree_map(wrap, out)
