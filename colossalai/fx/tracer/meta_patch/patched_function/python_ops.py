import operator

import torch

from colossalai.fx.proxy import ColoProxy

from ...registry import meta_patched_function


@meta_patched_function.register(operator.getitem)
def operator_getitem(a, b):
    # copied from huggingface.utils.fx
    def to_concrete(t):
        if isinstance(t, torch.Tensor):
            concrete = torch.ones_like(t, device="cpu")
            if concrete.dtype in [torch.float16, torch.float32, torch.float64, torch.int32]:
                concrete = concrete.to(torch.int64)
            return concrete
        return t

    def _slice_convert(slice_obj):
        attrs = {'start': slice_obj.start, 'stop': slice_obj.stop, 'step': slice_obj.step}
        new_attrs = _slice_attr_convert(attrs)
        attr_dict_to_tuple = (new_attrs['start'], new_attrs['stop'], new_attrs['step'])
        return slice(*attr_dict_to_tuple)

    def _slice_attr_convert(attrs):
        new_attrs = {}
        for key, value in attrs.items():
            if isinstance(value, ColoProxy):
                new_attrs[key] = value.meta_data
            else:
                new_attrs[key] = value
        return new_attrs

    if isinstance(b, tuple):
        b = list(b)
        for index, element in enumerate(b):
            if isinstance(element, slice):
                b[index] = _slice_convert(element)
        b = tuple(b)
    elif isinstance(b, slice):
        b = _slice_convert(b)

    if isinstance(a, torch.Tensor):
        # TODO: infer shape without performing the computation.
        if isinstance(b, tuple):
            b = tuple(map(to_concrete, b))
        else:
            b = to_concrete(b)
        return operator.getitem(torch.empty_like(a, device="cpu"), b).to("meta")

    if isinstance(a, ColoProxy):
        # TODO: infer shape without performing the computation.
        if isinstance(b, tuple):
            b = tuple(map(to_concrete, b))
        else:
            b = to_concrete(b)
        return operator.getitem(torch.empty_like(a.meta_data, device="cpu"), b).to("meta")
    return operator.getitem(a, b)
