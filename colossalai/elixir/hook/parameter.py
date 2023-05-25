from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from colossalai.elixir.chunk import ChunkFetcher
from colossalai.elixir.kernels import fused_torch_functions
from colossalai.elixir.tensor import OutplaceTensor, is_no_hook_op, to_outplace_tensor

from .functions import postfwd_prebwd_function, prefwd_postbwd_function
from .storage import BufferStore


class HookParam(OutplaceTensor, nn.Parameter):
    """HookParam is a special type of tensor that is used to triggered hooks on parameters.
    HookParam adds chunk fetching before torch functions.
    """
    pre_fwd_func = None
    post_fwd_func = None
    use_fused_kernel = False

    @staticmethod
    def attach_fetcher(fetcher: ChunkFetcher, store: BufferStore):
        HookParam.pre_fwd_func = prefwd_postbwd_function(fetcher, store)
        HookParam.post_fwd_func = postfwd_prebwd_function(fetcher, store)

    @staticmethod
    def release_fetcher():
        HookParam.pre_fwd_func = None
        HookParam.post_fwd_func = None

    @staticmethod
    def enable_fused_kernel():
        HookParam.use_fused_kernel = True

    @staticmethod
    def disable_fused_kernel():
        HookParam.use_fused_kernel = False

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if is_no_hook_op(func):
            with torch._C.DisableTorchFunction():
                ret = func(*args, **kwargs)
            return ret

        params_to_index = OrderedDict()
        params_index = 0

        def append_param(x):
            nonlocal params_index
            if isinstance(x, HookParam):
                params_to_index[x] = params_index
                params_index += 1

        tree_map(append_param, args)
        tree_map(append_param, kwargs)

        params = tuple(params_to_index.keys())
        new_params = HookParam.pre_fwd_func(params, *params)

        def replace_param(x):
            if isinstance(x, HookParam):
                return new_params[params_to_index[x]]
            return x

        with torch._C.DisableTorchFunction():
            if HookParam.use_fused_kernel and func in fused_torch_functions:
                func = fused_torch_functions.get(func)
            ret = func(*tree_map(replace_param, args), **tree_map(replace_param, kwargs))
        if not isinstance(ret, tuple):
            ret = (ret,)

        ptr_set = set()
        for p in new_params:
            ptr_set.add(p.data_ptr())

        def clone_inplace_tensor(x):
            if isinstance(x, torch.Tensor):
                start_point = x.data_ptr() - x.element_size() * x.storage_offset()
                if start_point in ptr_set:
                    return x.clone()
            return x

        ret = tree_map(clone_inplace_tensor, ret)
        ret = HookParam.post_fwd_func(params, *ret)

        def convert(t):
            if isinstance(t, torch.Tensor):
                t = to_outplace_tensor(t)
            return t

        ret = tree_map(convert, ret)

        if len(ret) == 1:
            return ret[0]
        else:
            return ret
