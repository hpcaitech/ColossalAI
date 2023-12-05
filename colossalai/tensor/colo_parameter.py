from typing import Optional

import torch

from colossalai.tensor.colo_tensor import ColoTensor
from colossalai.tensor.param_op_hook import ColoParamOpHookManager

from .colo_tensor import _convert_output

WHITE_LIST_FUNCS = {torch.Tensor.__getitem__, torch.Tensor.is_floating_point}


def is_no_hook_op(func) -> bool:
    return func.__name__.startswith("__") and func not in WHITE_LIST_FUNCS


def filter_colo_parameters(*args, **kwargs):
    param_list = []

    def get_colo_parameters(element) -> None:
        if isinstance(element, list) or isinstance(element, tuple):
            for e in element:
                get_colo_parameters(e)
        elif isinstance(element, dict):
            raise RuntimeError("Found Dict: ColoParameter can't deal with complicated arguments.")
        elif isinstance(element, ColoParameter):
            param_list.append(element)
        return

    for a in args:
        get_colo_parameters(a)
    for v in kwargs.values():
        get_colo_parameters(v)

    return param_list


def replace_args(args, kwargs, new_args):
    args = new_args[: len(args)]
    for k, v in zip(kwargs.keys(), new_args[len(args) :]):
        kwargs[k] = v
    return tuple(args), kwargs


class ColoParameter(ColoTensor, torch.nn.Parameter):
    r"""A kind of ColoTensor to be considered as a module parameter."""

    def __new__(cls, data: Optional[torch.Tensor] = None, requires_grad: bool = True) -> "ColoParameter":
        if data is None:
            data = torch.empty(0)
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    @classmethod
    def __torch_function__(cls, func, types, args=..., kwargs=None):
        if kwargs is None:
            kwargs = {}
        if ColoParamOpHookManager.has_hook() and not is_no_hook_op(func):
            params = filter_colo_parameters(*args, **kwargs)
            if len(params) > 0:
                with torch._C.DisableTorchFunction():
                    new_args = ColoParamOpHookManager.pre_op(params, *args, *kwargs.values())
                args, kwargs = replace_args(args, kwargs, new_args)
                ret = super().__torch_function__(func, types, args, kwargs)
                with torch._C.DisableTorchFunction():
                    ret = ColoParamOpHookManager.post_op(params, ret)
                return _convert_output(ret, func)
        return super().__torch_function__(func, types, args, kwargs)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            with torch._C.DisableTorchFunction():
                data = self.data.clone()
            tensor = ColoParameter(data, self.requires_grad)
            memo[id(self)] = tensor
            return tensor

    def __reduce_ex__(self, proto):
        # Adapted from torch._utils._rebuild_parameter
        # def _rebuild_colo_parameter(data, requires_grad, backward_hooks):
        #     colo_param = ColoParameter(data, requires_grad)
        #     colo_param._backward_hooks = backward_hooks
        #     return colo_param

        # return (
        #     _rebuild_colo_parameter,
        #     (self.data, self.requires_grad, OrderedDict())
        # )

        # TODO(jzy) we don't support object reflection now.
        # distspec cannot be pickled or rebuilt because it's tightly connected to runtime attribute `process_group`.
        raise NotImplementedError
