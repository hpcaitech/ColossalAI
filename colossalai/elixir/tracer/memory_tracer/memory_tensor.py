import contextlib
from typing import Iterator

import torch
from torch.utils._pytree import tree_map

from colossalai.elixir.tracer.utils import get_cuda_max_allocated

from .op_cache import wrapped_mm_ops

aten = torch.ops.aten

mm_ops_list = [aten.mm.default, aten.addmm.default, aten.bmm.default, aten.addbmm.default, aten.baddbmm.default]


@contextlib.contextmanager
def no_dispatch() -> Iterator[None]:
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


class MTensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ['elem']

    peak_memory_allocated: int = 0

    @staticmethod
    def reset_peak_memory():
        torch.cuda.reset_peak_memory_stats()
        MTensor.peak_memory_allocated = 0

    @staticmethod
    def update_peak_memory(new_peak):
        MTensor.peak_memory_allocated = max(MTensor.peak_memory_allocated, new_peak)

    @staticmethod
    def current_peak_memory():
        cur_peak = get_cuda_max_allocated()
        return max(MTensor.peak_memory_allocated, cur_peak)

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
        # TODO: clone strides and storage aliasing
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=elem.requires_grad)
        r.elem = elem
        return r

    def __repr__(self):
        return f'MTensor({self.elem})'

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):

        def print_tensor(x):
            if isinstance(x, torch.Tensor):
                print(x.shape)

        # tree_map(print_tensor, args)
        # tree_map(print_tensor, kwargs)

        def unwrap(x):
            return x.elem if isinstance(x, MTensor) else x

        def wrap(x):
            return MTensor(x) if isinstance(x, torch.Tensor) else x

        if func in mm_ops_list:
            res, pre_max = wrapped_mm_ops(func, *tree_map(unwrap, args), **tree_map(unwrap, kwargs))
            MTensor.update_peak_memory(pre_max)
        else:
            with no_dispatch():
                res = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        outs = normalize_tuple(res)
        res = tree_map(wrap, outs)

        if len(res) == 1:
            return res[0]
        else:
            return res
