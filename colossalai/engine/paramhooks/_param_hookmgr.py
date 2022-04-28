from typing import Callable, List
import torch
import functools


class BaseParamHookMgr(object):

    def __init__(self, param_list: List[torch.nn.Parameter]) -> None:
        r"""
        register backward hook on every parameters of module
        """
        self._param_list = param_list
        self._hook_list = []

    def register_backward_hooks(self, hook_call: Callable) -> None:
        r"""
        The hook_call will be called every time a gradient with respect to the a param in self.param_list
        is computed.
        The hook should have the following signature:
        ```
        hook(param, grad) -> Tensor or None
        ```
        """
        if not torch.is_grad_enabled():
            return    # don't register grad hooks if grad isn't enabled
        for p in self._param_list:
            if p.requires_grad and not hasattr(p, '_base_param_hook'):
                handle = p.register_hook(functools.partial(hook_call, p))
                p._base_param_hook = handle

    def remove_hooks(self) -> None:
        """
        Remove hooks from model parameters.
        """

        for p in self._param_list:
            if p.requires_grad and hasattr(p, '_base_param_hook'):
                p._base_param_hook.remove()
