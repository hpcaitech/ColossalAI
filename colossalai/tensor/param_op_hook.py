import torch
from contextlib import contextmanager
from abc import ABC, abstractmethod
from typing import List, Tuple


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


class _ParamOpHookWrapper:
    hooks: Tuple[ParamOpHook, ...] = tuple()


class PreFwdPostBwd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, params, *args):
        ctx.params = params
        for hook in _ParamOpHookWrapper.hooks:
            hook.pre_forward(ctx.params)
        if len(args) == 1:
            return args[0]
        return args

    @staticmethod
    def backward(ctx, *grads):
        for hook in _ParamOpHookWrapper.hooks:
            hook.post_backward(ctx.params)
        return (None,) + grads


class PostFwdPreBwd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, params, args):
        ctx.params = params
        for hook in _ParamOpHookWrapper.hooks:
            hook.post_forward(params)
        return args

    @staticmethod
    def backward(ctx, *grads):
        for hook in _ParamOpHookWrapper.hooks:
            hook.pre_backward(ctx.params)
        return (None,) + grads


@contextmanager
def use_param_op_hooks(*hooks: ParamOpHook):
    try:
        old_param_op_hooks = _ParamOpHookWrapper.hooks
        _ParamOpHookWrapper.hooks = hooks
        yield
    finally:
        _ParamOpHookWrapper.hooks = old_param_op_hooks
