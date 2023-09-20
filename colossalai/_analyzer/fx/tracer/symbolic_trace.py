from typing import Any, Callable, Dict, Optional, Union

import torch
from torch.fx import Tracer
from torch.utils._pytree import tree_map

from colossalai._analyzer._subclasses import MetaTensor

try:
    from ..codegen import ActivationCheckpointCodeGen

    SUPPORT_ACTIVATION = True
except:
    SUPPORT_ACTIVATION = False
from ..graph_module import ColoGraphModule
from .tracer import ColoTracer


def _default_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _current_device(module: torch.nn.Module):
    try:
        return next(module.parameters()).device
    except:
        return _default_device()


def symbolic_trace(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[Dict[str, Any]] = None,
    meta_args: Optional[Dict[str, Any]] = None,
    trace_act_ckpt: bool = False,
    bias_addition_split: bool = False,
) -> ColoGraphModule:
    """
    Traces a ``torch.nn.Module`` or a function and returns a ``GraphModule`` with ``Node``s and ``MetaInfo``
    attached to the ``Node``s.

    Can be used to trace the usage of ``torch.utils.checkpoint`` and the path of module
    (https://github.com/pytorch/examples/blob/main/fx/module_tracer.py).

    This tracer is able to trace basic control flow and for loops.

    It will split the bias addition into two parts if ``bias_addition_split`` is set to be ``True``.
    (See ./bias_addition.py for more details).

    Examples:
    1. Tracing a ``torch.nn.Module`` with control flow.

    .. code-block:: python

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                if x.size(0) > 1:
                    x = x.sum(dim=0)
                return self.linear(x)

        traced = symbolic_trace(MyModule(), meta_args={'x': torch.randn(1, 2, 2)})

        # traced code like:
        # def forward(self, x):
        #     linear_1 = self.linear(x)
        #     return linear_1

        traced = symbolic_trace(MyModule(), meta_args={'x': torch.randn(2, 2, 2)})

        # traced code like:
        # def forward(self, x):
        #     sum = x.sum(dim=0); x = None
        #     linear = self.linear(sum); sum = None
        #     return linear

    2. Tracing a ``torch.nn.Module`` with ``torch.utils.checkpoint``.

    .. code-block:: python

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                def custom_forward(x):
                    return self.linear(x)
                return torch.utils.checkpoint.checkpoint(custom_forward, x)

        traced = symbolic_trace(MyModule(), meta_args={'x': torch.randn(1, 2, 2)}, trace_act_ckpt=True)

        # traced code like:
        # def checkpoint_0(self, x):
        #     linear = self.linear(x); x = None
        #     return linear
        #
        # def forward(self, x):
        #     linear = torch.utils.checkpoint.checkpoint(checkpoint_0, x); x = None
        #     return linear

    3. Tracing a ``torch.nn.Module`` with ``bias_addition_split``.

    .. code-block:: python

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2, bias=True)

            def forward(self, x):
                return self.linear(x)

        traced = symbolic_trace(MyModule(), meta_args={'x': torch.randn(1, 2, 2)}, bias_addition_split=True)

        # traced code like:
        # def forward(self, x):
        #     linear_bias = self.linear.bias
        #     linear_weight = self.linear.weight
        #     linear = torch._C._nn.linear(x, linear_weight);  x = linear_weight = None
        #     add = linear + linear_bias;  linear = linear_bias = None
        #     return add

    Args:
        root (Union[torch.nn.Module, Callable[..., Any]]): The ``torch.nn.Module`` or function to be traced.
        concrete_args (Optional[Dict[str, Any]], optional): Concrete arguments to be passed to the ``root``.
            Defaults to {}.
        meta_args (Optional[Dict[str, Any]], optional): Meta arguments to be passed to the ``root``. Mostly used
            for tracing control flow. Defaults to {}.
        trace_act_ckpt (bool, optional): Whether to trace the usage of ``torch.utils.checkpoint``.
            Defaults to False.
        bias_addition_split (bool, optional): Whether to split the bias addition into two parts. Defaults to False.

    Returns:
        ColoGraphModule: A traced ``GraphModule`` that is ready for activation checkpoint ``CodeGen``.

    Remarks:
        This part of ``symbolic_trace()`` is maintained by Colossal-AI team. If you encountered
        any unexpected error during tracing, feel free to raise an issue on Colossal-AI GitHub
        repo. We welcome any feedback and contributions to enhance the extensibility of
        Colossal-AI.
    """
    if meta_args:
        device, orig_device = _default_device(), _current_device(root)
        wrap_fn = lambda elem: MetaTensor(elem, device=device) if isinstance(elem, torch.Tensor) else elem
        graph = ColoTracer(trace_act_ckpt=trace_act_ckpt, bias_addition_split=bias_addition_split).trace(
            root.to(device), concrete_args=concrete_args, meta_args=tree_map(wrap_fn, meta_args)
        )
        if trace_act_ckpt and SUPPORT_ACTIVATION:
            graph.set_codegen(ActivationCheckpointCodeGen())
        root.to(orig_device)
    else:
        graph = Tracer().trace(root, concrete_args=concrete_args)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    return ColoGraphModule(root, graph, name)
