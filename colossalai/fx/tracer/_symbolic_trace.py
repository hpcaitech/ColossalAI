from typing import Any, Callable, Dict, Optional, Union

import torch

from colossalai.fx import ColoGraphModule
from colossalai.fx._compatibility import compatibility

from .tracer import ColoTracer


@compatibility(is_backward_compatible=True)
def symbolic_trace(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[Dict[str, Any]] = None,
    meta_args: Optional[Dict[str, Any]] = None,
    trace_act_ckpt=False,
) -> ColoGraphModule:
    """
    Symbolic tracing API

    Given an ``nn.Module`` or function instance ``root``, this function will return a ``ColoGraphModule``
    constructed by recording operations seen while tracing through ``root``.

    With ``meta_args``, we can trace the model that are untraceable subject to control flow. If specified using
    ``meta_args`` only, the tracing can be done ahead of time.

    Note that ``meta_args`` are kwargs, which contains the key of the argument's names and the value of the
    argument's values.

    Uses:
        >>> model = ...

        # if this works
        >>> gm = symbolic_trace(model, concrete_args=concrete_args)

        # else try this
        >>> gm = symbolic_trace(model, concrete_args=concrete_args, meta_args={'x': torch.rand(1, 3, 224, 224, device='meta')})

    Args:
        root (Union[torch.nn.Module, Callable[..., Any]]): Module or function to be traced and converted
            into a Graph representation.
        concrete_args (Optional[Dict[str, Any]], optional): Concrete arguments to be used for tracing.
        meta_args (Optional[Dict[str, Any]], optional): Inputs to be partially specialized, special for ``ColoTracer``.
            Defaults to None.

    Returns:
        ColoGraphModule: A ``ColoGraphModule`` created from the recorded operations from ``root``.

    Warnings:
        This API is still under development and can incur some bugs. Feel free to report any bugs to the Colossal-AI team.

    """
    graph = ColoTracer(trace_act_ckpt=trace_act_ckpt).trace(root, concrete_args=concrete_args, meta_args=meta_args)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    return ColoGraphModule(root, graph, name)
