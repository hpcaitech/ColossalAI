from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, List, Tuple

import torch
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten


class ColoParamOpHook(ABC):
    """
    Hook which is triggered by each operation when operands contain ColoParameter.
    To customize it, you must inherit this abstract class, and implement ``pre_forward``,
    ``post_forward``, ``pre_backward`` and ``post_backward``.
    These four methods apply a list of ColoParameter as input args.
    """

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

    def rewrite_op(self, func) -> Any:
        return func


class ColoParamOpHookManager:
    """
    Manage your param op hooks. It only has static methods.
    The only static method you should call is ``use_hooks(*hooks)``.
    """

    hooks: Tuple[ColoParamOpHook, ...] = tuple()

    @staticmethod
    @contextmanager
    def use_hooks(*hooks: ColoParamOpHook):
        """Change the param op hooks you use. Nested calling is allowed.

        Example:
            >>> with ColoParamOpHookManager.use_hooks(*hooks):
            >>>     do_something()
            >>>     with ColoParamOpHookManager.use_hooks():
            >>>         // clear hooks
            >>>         do_something()
        """
        try:
            old_param_op_hooks = ColoParamOpHookManager.hooks
            ColoParamOpHookManager.hooks = hooks
            yield
        finally:
            ColoParamOpHookManager.hooks = old_param_op_hooks

    @staticmethod
    def _trigger_pre_forward(params: List[torch.Tensor]) -> None:
        for hook in ColoParamOpHookManager.hooks:
            hook.pre_forward(params)

    @staticmethod
    def _trigger_post_forward(params: List[torch.Tensor]) -> None:
        for hook in ColoParamOpHookManager.hooks:
            hook.post_forward(params)

    @staticmethod
    def _trigger_pre_backward(params: List[torch.Tensor]) -> None:
        for hook in ColoParamOpHookManager.hooks:
            hook.pre_backward(params)

    @staticmethod
    def _trigger_post_backward(params: List[torch.Tensor]) -> None:
        for hook in ColoParamOpHookManager.hooks:
            hook.post_backward(params)

    @staticmethod
    def pre_op(params: List[torch.Tensor], *args: Any) -> list:
        ColoParamOpHookManager._trigger_pre_forward(params)
        # auto grad function can only recognize torch.Tensor, thus we have to flatten the input
        # if one of the input requires grad, all the output will be treated as requires grad
        # and will have grad fn even the corresponding input does not require grad
        # we have to extract tensors requiring grad into flat list and then merge them back
        grad_args, other_args, grad_flags, spec = _flatten_grad_args(args)
        new_grad_args = PreFwdPostBwd.apply(params, *grad_args)
        return _merge_args(new_grad_args, other_args, grad_flags, spec)

    @staticmethod
    def post_op(params: List[torch.Tensor], arg: Any) -> Any:
        ColoParamOpHookManager._trigger_post_forward(params)
        # incase the output is a tuple, we have to flatten it
        grad_args, other_args, grad_flags, spec = _flatten_grad_args(arg)
        new_grad_args = PostFwdPreBwd.apply(params, *grad_args)
        return _merge_args(new_grad_args, other_args, grad_flags, spec)

    @staticmethod
    def has_hook() -> bool:
        return len(ColoParamOpHookManager.hooks) > 0

    @staticmethod
    def rewrite_op(func) -> Any:
        for hook in ColoParamOpHookManager.hooks:
            func = hook.rewrite_op(func)
        return func


class PreFwdPostBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params, *args):
        ctx.params = params
        return args

    @staticmethod
    def backward(ctx, *grads):
        ColoParamOpHookManager._trigger_post_backward(ctx.params)
        return (None,) + grads


class PostFwdPreBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params, *args):
        ctx.params = params
        return args

    @staticmethod
    def backward(ctx, *grads):
        ColoParamOpHookManager._trigger_pre_backward(ctx.params)
        return (None,) + grads


def _is_grad_tensor(obj) -> bool:
    if torch.is_tensor(obj):
        if obj.grad_fn is not None or obj.requires_grad:
            return True
    return False


def _flatten_grad_args(args) -> Tuple[list, list, List[bool], TreeSpec]:
    flat_args, spec = tree_flatten(args)
    grad_args = []
    other_args = []
    grad_flags = []
    for arg in flat_args:
        flag = _is_grad_tensor(arg)
        grad_flags.append(flag)
        if flag:
            grad_args.append(arg)
        else:
            other_args.append(arg)
    return grad_args, other_args, grad_flags, spec


def _merge_args(grad_args, other_args, grad_flags, spec):
    grad_iter = iter(grad_args)
    other_iter = iter(other_args)
    flat_args = [next(grad_iter) if flag else next(other_iter) for flag in grad_flags]
    return tree_unflatten(flat_args, spec)
