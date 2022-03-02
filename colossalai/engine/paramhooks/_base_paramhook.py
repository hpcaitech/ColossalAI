from typing import Callable, List
import torch

class BaseParamHook(object):
    def __init__(self, param_list: List[torch.nn.Parameter], backward_hook_call: Callable) -> None:
        r"""
        register backward hook on every parameters of module
        """
        self.param_list = param_list
        self._register_backward_hooks(backward_hook_call)

    def _register_backward_hooks(self, hook_call : Callable) -> None:
        r"""
        The hook_call will be called every time a gradient with respect to the a param in self.param_list 
        is computed. 
        The hook should have the following signature:
        ```
        hook(grad) -> Tensor or None
        ```
        """
        if not torch.is_grad_enabled():
            return  # don't register grad hooks if grad isn't enabled
        for p in self.param_list:
            if p.requires_grad and not hasattr(p, 'zero_shard_bwd_hook'):
                # For mixed precision with activation checkpoint, hooks on GradAccumulation won't be fired normally
                # Instead we register hook on parameter
                # In this way, we can't modify param.grad and param.data directly, which leads to more memory usage
                # Register a hook on the first call, empirically, autograd
                # fires it at the end for this param, which makes sense.
                # p_tmp = p.expand_as(p)  # Get a grad_fn on p_tmp.
                # assert p_tmp.grad_fn is not None
                # grad_acc = p_tmp.grad_fn.next_functions[0][0]  # Gets its GradAccumulation object.
                # handle = grad_acc.register_hook(functools.partial(self._post_backward_hook, p))
                # p.zero_shard_bwd_hook = (grad_acc, handle)
                # handle = p.register_hook(functools.partial(self._post_backward_hook, p))
                handle = p.register_hook(hook_call)
                p.zero_shard_bwd_hook = handle
