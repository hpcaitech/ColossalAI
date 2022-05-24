import torch
from contextlib import contextmanager
from abc import ABC, abstractmethod
from typing import List


class ParamOpHook(ABC):

    @abstractmethod
    def pre_forward(self, params: List[torch.Tensor]) -> None:
        pass

    @abstractmethod
    def post_forward(self, params: List[torch.Tensor]) -> None:
        pass

    @abstractmethod
    def pre_backward(self, params: List[torch.Tensor]) -> None:
        pass

    @abstractmethod
    def post_backward(self, params: List[torch.Tensor]) -> None:
        pass


_COLOSSAL_PARAM_OP_HOOKS: List[ParamOpHook] = []


class PreFwdPostBwd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, params, *args):
        ctx.params = params
        for hook in _COLOSSAL_PARAM_OP_HOOKS:
            hook.pre_forward(ctx.params)
        if len(args) == 1:
            return args[0]
        return args

    @staticmethod
    def backward(ctx, *grads):
        for hook in _COLOSSAL_PARAM_OP_HOOKS:
            hook.post_backward(ctx.params)
        return (None,) + grads


class PostFwdPreBwd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, params, args):
        ctx.parmas = params
        for hook in _COLOSSAL_PARAM_OP_HOOKS:
            hook.post_forward(params)
        return args

    @staticmethod
    def backward(ctx, *grads):
        for hook in _COLOSSAL_PARAM_OP_HOOKS:
            hook.pre_backward(ctx.params)
        return (None,) + grads


@contextmanager
def use_param_op_hooks(hooks: List[ParamOpHook]):
    try:
        global _COLOSSAL_PARAM_OP_HOOKS
        old_param_op_hooks = _COLOSSAL_PARAM_OP_HOOKS
        _COLOSSAL_PARAM_OP_HOOKS = hooks
        yield
    finally:
        _COLOSSAL_PARAM_OP_HOOKS = old_param_op_hooks
