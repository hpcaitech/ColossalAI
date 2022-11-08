from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, List, Tuple

import torch

from colossalai.tensor.colo_tensor import ColoTensor
from colossalai.tensor.tensor_spec import ColoTensorSpec


class ParamOpHook(ABC):
    """Hook which is triggered by each operation when operands contain ColoParameter.
    To customize it, you must inherit this abstract class, and implement ``pre_forward``,
    ``post_forward``, ``pre_backward`` and ``post_backward``. These four methods take a list
    of ColoParameter.
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


class ParamOpHookManager:
    """Manage your param op hooks. It only has static methods.
    The only static method you should call is ``use_hooks(*hooks)``.
    """
    hooks: Tuple[ParamOpHook, ...] = tuple()

    @staticmethod
    @contextmanager
    def use_hooks(*hooks: ParamOpHook):
        """Change the param op hooks you use. Nested calling is allowed.

        Example:
            >>> with ParamOpHookManager.use_hooks(*hooks):
            >>>     do_something()
            >>>     with ParamOpHookManager.use_hooks():
            >>>         // clear hooks
            >>>         do_something()
        """
        try:
            old_param_op_hooks = ParamOpHookManager.hooks
            ParamOpHookManager.hooks = hooks
            yield
        finally:
            ParamOpHookManager.hooks = old_param_op_hooks

    @staticmethod
    def _trigger_pre_forward(params: List[torch.Tensor]) -> None:
        for hook in ParamOpHookManager.hooks:
            hook.pre_forward(params)

    @staticmethod
    def _trigger_post_forward(params: List[torch.Tensor]) -> None:
        for hook in ParamOpHookManager.hooks:
            hook.post_forward(params)

    @staticmethod
    def _trigger_pre_backward(params: List[torch.Tensor]) -> None:
        for hook in ParamOpHookManager.hooks:
            hook.pre_backward(params)

    @staticmethod
    def _trigger_post_backward(params: List[torch.Tensor]) -> None:
        for hook in ParamOpHookManager.hooks:
            hook.post_backward(params)

    @staticmethod
    def pre_op(params: List[torch.Tensor], *args: Any) -> list:
        ParamOpHookManager._trigger_pre_forward(params)
        args_info = _get_colo_tensors_info(*args)
        rets = PreFwdPostBwd.apply(params, *args)
        return _update_colo_tensors(args_info, *rets)

    @staticmethod
    def post_op(params: List[torch.Tensor], arg: Any) -> Any:
        ParamOpHookManager._trigger_post_forward(params)
        arg_info = _get_colo_tensors_info(arg)
        ret = PostFwdPreBwd.apply(params, arg)
        return _unpack_args(_update_colo_tensors(arg_info, ret))

    @staticmethod
    def has_hook() -> bool:
        return len(ParamOpHookManager.hooks) > 0


class PreFwdPostBwd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, params, *args):
        ctx.params = params
        return _unpack_args(args)

    @staticmethod
    def backward(ctx, *grads):
        ParamOpHookManager._trigger_post_backward(ctx.params)
        return (None,) + grads


class PostFwdPreBwd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, params, args):
        ctx.params = params
        return args

    @staticmethod
    def backward(ctx, *grads):
        ParamOpHookManager._trigger_pre_backward(ctx.params)
        return (None,) + grads


def _unpack_args(args):
    if len(args) == 1:
        return args[0]
    return args


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
