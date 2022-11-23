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
) -> ColoGraphModule:
    """
    Symbolic tracing API

    Given an ``nn.Module`` or function instance ``root``, this function will return a ``ColoGraphModule``
    constructed by recording operations seen while tracing through ``root``.

    With ``meta_args`` and ``concrete_args``, we can trace the model that are untraceable subject to control flow.
    If specified using ``meta_args`` only, the tracing can be done ahead of time.

    Note that both ``meta_args`` and ``concrete_args`` are kwargs, which contains the key of the argument's names
    and the value of the argument's values.

    Uses:
        >>> model = ...

        # if this works
        >>> gm = symbolic_trace(model)

        # else try this
        >>> gm = symbolic_trace(model, meta_args={'x': torch.rand(1, 3, 224, 224, device='meta')})

        # else try this
        >>> gm = symbolic_trace(model, concrete_args={'x': torch.rand(1, 3, 224, 224)})

    Args:
        root (Union[torch.nn.Module, Callable[..., Any]]): Module or function to be traced and converted
            into a Graph representation.
        concrete_args (Optional[Dict[str, Any]], optional): Inputs to be partially specialized. Defaults to None.
        meta_args (Optional[Dict[str, Any]], optional): Inputs to be partially specialized, special for ``ColoTracer``.
            Defaults to None.

    Returns:
        ColoGraphModule: A ``ColoGraphModule`` created from the recorded operations from ``root``.

    Warnings:
        This API is still under development and can incur some bugs. Feel free to report any bugs to the Colossal-AI team.

    """
    tracer = ColoTracer()
    graph = tracer.trace(root, concrete_args, meta_args)
    name = (root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__)
    return ColoGraphModule(tracer.root, graph, name)
