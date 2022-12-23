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
        pack_info, unpacked_args = _unpack_args(*args)
        # unpacked_args = args
        colo_info = _get_colo_tensors_info(*unpacked_args)
        rets = PreFwdPostBwd.apply(params, *unpacked_args)
        update_args = _update_colo_tensors(colo_info, *rets)
        return _pack_args(pack_info, *update_args)
        # return update_args

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


def _unpack_args(*args):
    unpacked_args_list = []

    def get_pack_info(original_args):
        pack_info_list = []
        if isinstance(original_args, tuple) or isinstance(original_args, list):
            for oa in original_args:
                pack_info_list.append(get_pack_info(oa))
        elif isinstance(original_args, dict):
            raise RuntimeError("Found Dict: ColoParameterOp hooks can't support such complicated arguments")
        else:
            unpacked_args_list.append(original_args)    # appends a single arguement
            return type(original_args)    # returns the type of the arguement
        if isinstance(original_args, tuple):
            pack_info_list = tuple(pack_info_list)
        return pack_info_list

    pack_info = get_pack_info(args)
    return pack_info, tuple(unpacked_args_list)


def _pack_args(pack_info, *args):
    cursor = 0

    def dfs_pack(cur_pack_info):
        pack_list = []
        if isinstance(cur_pack_info, tuple) or isinstance(cur_pack_info, list):
            for cpi in cur_pack_info:
                pack_list.append(dfs_pack(cpi))
        else:
            nonlocal cursor
            cursor += 1
            return args[cursor - 1]
        if isinstance(cur_pack_info, tuple):
            pack_list = tuple(pack_list)
        return pack_list

    return dfs_pack(pack_info)


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
