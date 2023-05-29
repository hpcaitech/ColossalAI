import torch
import xformers.ops as xops
from torch.utils._pytree import tree_map

from colossalai.elixir.tracer.memory_tracer.memory_tensor import MTensor


def lower_triangular_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, p: float = 0.0):

    args = (query, key, value)
    meta_flag = False

    for x in args:
        if x.device.type == 'meta':
            meta_flag = True
            break

    if meta_flag:
        atten = query @ key.transpose(-2, -1)
        output = atten @ value
        return output

    profile_flag = False

    def to_torch_tensor(x):
        if isinstance(x, MTensor):
            nonlocal profile_flag
            profile_flag = True
            return x.elem
        return x

    args = tree_map(to_torch_tensor, args)
    query, key, value = args
    output = xops.memory_efficient_attention(query=query,
                                             key=key,
                                             value=value,
                                             p=p,
                                             attn_bias=xops.LowerTriangularMask())

    if profile_flag:
        output = MTensor(output)

    return output
