from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, List, Tuple

import torch

from colossalai.tensor.colo_tensor import ColoTensor
from colossalai.tensor.tensor_spec import ColoTensorSpec


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
        grad_args, rear_args = _get_grad_args(*args)
        colo_info = _get_colo_tensors_info(*grad_args)
        rets = PreFwdPostBwd.apply(params, *grad_args)
        update_args = _update_colo_tensors(colo_info, *rets)
        if rear_args is None:
            return update_args
        else:
            arg_zero = (tuple(update_args),)
            return arg_zero + rear_args

    @staticmethod
    def post_op(params: List[torch.Tensor], arg: Any) -> Any:
        ColoParamOpHookManager._trigger_post_forward(params)
        colo_info = _get_colo_tensors_info(arg)
        ret = PostFwdPreBwd.apply(params, arg)
        res = _update_colo_tensors(colo_info, ret)
        if len(res) == 1:
            return res[0]
        else:
            return res

    @staticmethod
    def has_hook() -> bool:
        return len(ColoParamOpHookManager.hooks) > 0


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
    def forward(ctx, params, args):
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


def _has_grad_tensor(obj) -> bool:
    if isinstance(obj, tuple) or isinstance(obj, list):
        for x in obj:
            if _has_grad_tensor(x):
                return True
        return False
    elif isinstance(obj, dict):
        for x in obj.values():
            if _has_grad_tensor(x):
                return True
        return False
    else:
        return _is_grad_tensor(obj)


def _get_grad_args(*args):
    # if there is no grad tensors, do nothing
    if not _has_grad_tensor(args):
        return args, None
    # returns the identical args if there is a grad tensor
    for obj in args:
        if _is_grad_tensor(obj):
            return args, None
    # otherwise, the first arguement should be a tuple of grad tensors
    # if there is no grad tensor, the backward of PreFwdPostBwd can't be triggered
    arg_zero = args[0]
    if not isinstance(arg_zero, tuple):
        raise NotImplementedError("Some torch function is incompatible because of its complcated inputs.")
    check_grad_flag = False
    for obj in arg_zero:
        check_grad_flag |= _is_grad_tensor(obj)
    if not check_grad_flag:
        raise NotImplementedError("Some torch function is incompatible because of its complcated inputs.")
    return arg_zero, args[1:]


def _get_colo_tensors_info(*args) -> list:
    info = []
    for arg in args:
        if isinstance(arg, ColoTensor):
            info.append((arg.__class__, ColoTensorSpec(arg.get_process_group(), arg.dist_spec, arg.compute_spec)))
        else:
            info.append(None)
    return info


def _update_colo_tensors(info, *args) -> list:
    ret = []
    for t_info, arg in zip(info, args):
        if t_info is not None:
            t_cls, spec = t_info
            arg = t_cls.from_torch_tensor(arg, spec=spec)
        ret.append(arg)
    return ret
