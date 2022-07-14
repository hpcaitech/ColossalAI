import torch
from typing import List, Callable, Optional

from abc import ABC, abstractmethod
import torch


class BaseOpHook(ABC):
    """This class allows users to add customized operations
    before and after the execution of a PyTorch submodule"""

    def __init__(self):
        pass

    @abstractmethod
    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        pass

    @abstractmethod
    def post_fwd_exec(self, module: torch.nn.Module, *args):
        pass

    @abstractmethod
    def pre_bwd_exec(self, module: torch.nn.Module, input, output):
        pass

    @abstractmethod
    def post_bwd_exec(self, module: torch.nn.Module, input):
        pass

    @abstractmethod
    def post_iter(self):
        pass


# apply torch.autograd.Function that calls a backward_function to tensors in output
def _apply_to_tensors_only(module, functional, backward_function, outputs):
    if type(outputs) is tuple:
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_to_tensors_only(module, functional, backward_function, output)
            touched_outputs.append(touched_output)
        return tuple(touched_outputs)
    elif type(outputs) is torch.Tensor:
        return functional.apply(module, backward_function, outputs)
    else:
        return outputs


class PreBackwardFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, module, pre_backward_function, outputs):
        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        module.applied_pre_backward = False
        outputs = outputs.detach()
        return outputs

    @staticmethod
    def backward(ctx, *args):
        ctx.pre_backward_function(ctx.module)
        return (None, None) + args


class PostBackwardFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, module, pre_backward_function, output):
        ctx.module = module
        output = output.detach()
        ctx.pre_backward_function = pre_backward_function
        return output

    @staticmethod
    def backward(ctx, *args):
        """
        Args:
            activation_grad of the next layer.
        Returns:
            grad of the input activation.
        """
        ctx.pre_backward_function(ctx.module)
        return (None, None) + args


def register_ophooks_recursively(module: torch.nn.Module,
                                 ophook_list: List[BaseOpHook],
                                 name: str = "",
                                 filter_fn: Optional[Callable] = None):
    r"""Recursilvely register pre/post hooks for all submodules in the module in FWD and BWD."""
    assert isinstance(module, torch.nn.Module)
    assert isinstance(ophook_list, (list, tuple))
    assert len(ophook_list) > 0, 'expected at least 1 hook in the argument ophook_list but found 0'
    for hook in ophook_list:
        assert (isinstance(hook, BaseOpHook))

    # Add hooks for submodules
    for child_name, child in module.named_children():
        register_ophooks_recursively(child, ophook_list, name + child_name, filter_fn)

    # Early return on modules with no parameters.
    if len(list(module.parameters(recurse=False))) == 0:
        return

    # return from flitered module
    if filter_fn is not None and filter_fn(module):
        return

    def _pre_forward_module_hook(submodule, *args):
        for hook in ophook_list:
            assert isinstance(submodule, torch.nn.Module)
            hook.pre_fwd_exec(submodule, *args)

    def _post_forward_module_hook(submodule, *args):
        for hook in ophook_list:
            assert isinstance(submodule, torch.nn.Module)
            hook.post_fwd_exec(submodule, *args)

    def _pre_backward_module_hook(submodule, inputs, output):

        def _run_before_backward_function(submodule):
            for hook in ophook_list:
                assert isinstance(submodule, torch.nn.Module)
                hook.pre_bwd_exec(submodule, inputs, output)

        return _apply_to_tensors_only(submodule, PreBackwardFunction, _run_before_backward_function, output)

    def _post_backward_module_hook(submodule, inputs):

        def _run_after_backward_function(submodule):
            for hook in ophook_list:
                assert isinstance(submodule, torch.nn.Module)
                hook.post_bwd_exec(submodule, inputs)

        return _apply_to_tensors_only(submodule, PostBackwardFunction, _run_after_backward_function, inputs)

    module.register_forward_pre_hook(_pre_forward_module_hook)
    module.register_forward_hook(_post_forward_module_hook)

    module.register_forward_hook(_pre_backward_module_hook)
    module.register_forward_pre_hook(_post_backward_module_hook)
