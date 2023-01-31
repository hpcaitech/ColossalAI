"""``torch.fx.ShapeProp``, but with ``MetaTensor``"""

from typing import Any, Callable, Dict, Tuple, Union

import torch
import torch.fx
from siu._subclasses import MetaTensor, MetaTensorMode
from siu.fx.node_util import MetaInfo
from torch.utils._pytree import tree_map

from colossalai.fx._compatibility import compatibility

Target = Union[Callable[..., Any], str]


def _normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


def _current_device(module):
    return next(module.parameters()).device


@compatibility(is_backward_compatible=False)
class ShapeProp(torch.fx.Interpreter):
    """
    Execute an FX graph Node-by-Node and record the meta data of the result
    into the corresponding node.

    Usage:
        >>> model = MyModule()
        >>> x = torch.rand(10, 10)
        >>> gm = colossalai.fx.symbolic_trace(model, meta_args = {'x': x})
        >>> interp = ShapeProp(gm)
        >>> interp.propagate(x)

    Args:
        module (GraphModule): The module to be executed

    Hints:
        If you want to add a new shape propagation rule, you can do so by
        adding a new method to this class with the ``@register_shape_impl``
        decorator. The method should take (*args, **kwargs) instance as its
        input and generate output.

        For example, if you want to add a shape propagation rule for
        ``torch.nn.functional.linear``, you can do so by adding a new method
        to this class with the ``@register_shape_impl`` decorator (Since the
        ``MetaTensorMode`` is compatible with ``torch.nn.functional.linear``,
        in practice you don't have to do as follows):

        >>> @register_shape_impl(torch.nn.functional.linear)
        >>> def linear_shape_impl(*args, **kwargs):
        >>>     # do something here
        >>>     return torch.empty(output_shape, device=output_device)
    """
    _custom_dispatch_func = {}
    _mode = MetaTensorMode()

    def run_node(self, n: torch.fx.Node) -> Any:
        """
        Run a specific node ``n`` and return the result. Attach
        (
            ``inputs``, ``outputs``, ``parameters``, ``buffers``
        ) to ``n``.

        Args:
            n (Node): The ``Node`` to execute

        Returns:
            Any: The result of executing ``n``
        """
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        r = getattr(self, n.op)(n.target, args, kwargs)

        unwrap_fn = lambda elem: elem._tensor if isinstance(elem, MetaTensor) else elem
        is_pure_tensor = lambda elem: isinstance(elem, MetaTensor) and not isinstance(elem, torch.nn.Parameter)
        n_info = MetaInfo(n)
        n_info.outputs = _normalize_tuple(r)

        if n.op == 'call_module':
            submod = self.fetch_attr(n.target)
            n_info.parameters.update({k: v.to(torch.device('meta')) for k, v in submod.named_parameters()})
            n_info.buffers.update({k: v.to(torch.device('meta')) for k, v in submod.named_buffers()})

        else:
            # fix-me: ``nn.Parameter`` cannot be ``kwargs``?
            n_info.parameters.update(
                {k.name: v.to(torch.device('meta')) \
                    for k, v in zip(n.args, args) \
                        if isinstance(k, torch.fx.Node) and isinstance(v, torch.nn.Parameter)
                }
            )

        n_info.inputs = tuple(v for v in args if is_pure_tensor(v)) + \
                        tuple(v for v in kwargs.values() if is_pure_tensor(v))

        n._meta_data = tree_map(unwrap_fn, _normalize_tuple(r))
        return r

    def call_function(self, target: 'Target', args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a ``call_function`` node and return the result.
        If the target of ``Node`` is registered with ``@register_shape_impl``,
        the registered function will be used to execute the node. This is common
        if we insert some customized kernels.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the function invocation
        """
        if target in self._custom_dispatch_func:
            return self._custom_dispatch_func[target](*args, **kwargs)
        else:
            return super().call_function(target, args, kwargs)

    def propagate(self, *args, device=None):
        """
        Run `module` via interpretation and return the result and record the
        shape of each node.
        Args:
            *args (Tensor): The sample input.
        Returns:
            Any: The value returned from executing the Module
        """
        wrap_fn = lambda elem: MetaTensor(elem, device=device)
        with self._mode:
            return super().run(*tree_map(wrap_fn, args))


def shape_prop_pass(module: torch.fx.GraphModule, *args) -> torch.fx.GraphModule:
    """
    Run ``module`` via interpretation and return the result and record the
    shape of each ``Node``.

    Args:
        module (GraphModule): The GraphModule to profile
        *args (Any): The sample input

    Returns:
        GraphModule: The same GraphModule with shape information
    """

    ShapeProp(module).propagate(*args, device=_current_device(module))
    return module
