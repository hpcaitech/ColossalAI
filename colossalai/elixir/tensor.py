import torch
from torch.utils._pytree import tree_map

debug_flag = False

white_list = {torch.Tensor.__getitem__}

fake_allowed = {
    # pre-commit: don't move
    torch.Tensor.numel,
    torch.Tensor.size,
    torch.Tensor.stride,
    torch.Tensor.storage_offset,
    torch.Tensor.is_floating_point
}

inpalce_mapping = {
    torch.Tensor.add_: torch.Tensor.add,
    torch.Tensor.sub_: torch.Tensor.sub,
    torch.Tensor.mul_: torch.Tensor.mul,
    torch.Tensor.div_: torch.Tensor.div
}


def is_no_hook_op(func) -> bool:
    if func.__name__.startswith('__') and func not in white_list:
        return True
    if func in fake_allowed:
        return True
    return False


class FakeTensor(torch.Tensor):

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        r = torch.Tensor._make_wrapper_subclass(cls,
                                                elem.size(),
                                                strides=elem.stride(),
                                                storage_offset=elem.storage_offset(),
                                                dtype=elem.dtype,
                                                layout=elem.layout,
                                                device=elem.device,
                                                requires_grad=elem.requires_grad)
        return r

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise NotImplementedError


def to_outplace_tensor(t):
    if isinstance(t, OutplaceTensor):
        return t
    assert type(t) is torch.Tensor, f'type: {type(t)}'
    t.__class__ = OutplaceTensor
    return t


class OutplaceTensor(torch.Tensor):
    # TODO: rename this class
    def __new__(cls, tensor):
        rt = tensor.as_subclass(cls)
        return rt

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):

        if kwargs is None:
            kwargs = {}
        # in order to trigger pre-op hook in the forward of checkpoint module
        # we have to capture the `backward` function
        # and make sure that it does not in `torch._C.DisableTorchFunction()` context
        if func is torch.Tensor.backward:
            assert len(args) == 1    # only has 1 paramter
            backward_tensor = torch.Tensor(args[0])
            tensor_kwargs = {k: torch.Tensor(v) if torch.is_tensor(v) else v for k, v in kwargs.items()}
            return backward_tensor.backward(**tensor_kwargs)
        # return a tensor if the output needs to be a torch.Tensor (such as Tensor.data.__get__)
        if is_no_hook_op(func):
            with torch._C.DisableTorchFunction():
                ret = func(*args, **kwargs)
            return ret

        # debug inplace operations
        if debug_flag:
            if func.__name__.endswith('_'):
                print(f'found inplace operation {func.__name__}')

        # replace the in-place function
        if func in inpalce_mapping:
            func = inpalce_mapping[func]
        # set the 'inplace' kwargs to False
        if 'inplace' in kwargs:
            kwargs['inplace'] = False

        with torch._C.DisableTorchFunction():
            ret = func(*args, **kwargs)
        if not isinstance(ret, tuple):
            ret = (ret,)

        def convert(t):
            if isinstance(t, torch.Tensor):
                t = to_outplace_tensor(t)
            return t

        ret = tree_map(convert, ret)

        if len(ret) == 1:
            ret = ret[0]

        return ret
