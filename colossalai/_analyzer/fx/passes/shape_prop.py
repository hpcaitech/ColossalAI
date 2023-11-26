"""``torch.fx.ShapeProp``, but with ``MetaTensor``"""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.fx
from torch.autograd.graph import saved_tensors_hooks
from torch.utils._pytree import tree_map

from colossalai._analyzer._subclasses import MetaTensor, MetaTensorMode
from colossalai._analyzer.fx.node_util import MetaInfo
from colossalai.fx._compatibility import compatibility

Target = Union[Callable[..., Any], str]


class sim_env(saved_tensors_hooks):
    """
    A simulation of memory allocation and deallocation in the forward pass
    using ``saved_tensor_hooks``.

    Attributes:
        ctx (Dict[int, torch.Tensor]): A dictionary that maps the
            data pointer of a tensor to the tensor itself. This is used
            to track the memory allocation and deallocation.

        param_ctx (Dict[int, torch.Tensor]): A dictionary that maps the
            data pointer of all model parameters to the parameter itself.
            This avoids overestimating the memory usage of the intermediate activations.
    """

    def __init__(self, module: Optional[torch.nn.Module] = None):
        super().__init__(self.pack_hook, self.unpack_hook)
        self.ctx = {}
        self.param_ctx = {param.data_ptr(): param for param in module.parameters()}
        self.buffer_ctx = {buffer.data_ptr(): buffer for buffer in module.buffers()} if module else {}

    def pack_hook(self, tensor: torch.Tensor):
        if tensor.data_ptr() not in self.param_ctx and tensor.data_ptr() not in self.buffer_ctx:
            self.ctx[tensor.data_ptr()] = tensor
        return tensor

    def unpack_hook(self, tensor):
        return tensor


def _normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


def _current_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


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

    def __init__(self, module: torch.fx.GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)
        self.global_hook = sim_env(module=self.module)

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
        with self.global_hook:
            r = getattr(self, n.op)(n.target, args, kwargs)

        def unwrap_fn(elem):
            def _convert_meta(t: torch.Tensor):
                if t.device == "meta":
                    return t
                else:
                    return t.to("meta")

            if isinstance(elem, MetaTensor):
                if getattr(self, "_is_param", False):
                    return torch.nn.Parameter(_convert_meta(elem._tensor))
                return _convert_meta(elem._tensor)

            elif isinstance(elem, torch.Tensor):
                if isinstance(elem, torch.nn.Parameter):
                    return torch.nn.Parameter(_convert_meta(elem))
                return _convert_meta(elem)

            else:
                return elem

        is_pure_tensor = lambda elem: isinstance(elem, MetaTensor) and not isinstance(elem, torch.nn.Parameter)
        n_info = MetaInfo(n)
        n_info.outputs = _normalize_tuple(r)

        if n.op == "call_module":
            submod = self.fetch_attr(n.target)
            n_info.parameters.update({k: MetaTensor(v) for k, v in submod.named_parameters()})
            n_info.buffers.update({k: MetaTensor(v) for k, v in submod.named_buffers()})

        else:
            n_info.parameters.update(
                {
                    k.name: MetaTensor(v)
                    for k, v in zip(n.args, args)
                    if isinstance(k, torch.fx.Node) and isinstance(v, torch.nn.Parameter)
                }
            )
            n_info.parameters.update({k: MetaTensor(v) for k, v in kwargs.items() if isinstance(v, torch.nn.Parameter)})

        n_info.inputs = tuple(v for v in args if is_pure_tensor(v)) + tuple(
            v for v in kwargs.values() if is_pure_tensor(v)
        )

        # align with SPMD
        if isinstance(r, (tuple, list)):
            n._meta_data = tree_map(unwrap_fn, _normalize_tuple(r))
        else:
            n._meta_data = unwrap_fn(r)

        n_info.global_ctx = self.global_hook.ctx
        n_info.curr_ctx = self.global_hook.ctx.copy()

        crit = lambda x: x.data_ptr() in self.global_hook.ctx if isinstance(x, torch.Tensor) else False
        n_info.is_alias = _normalize_tuple(tree_map(crit, n_info.outputs))
        return r

    def call_function(self, target: "Target", args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
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
        convert_to_param = False
        if target in (torch.transpose, torch.reshape) and isinstance(args[0], torch.nn.parameter.Parameter):
            convert_to_param = True
        if target in self._custom_dispatch_func:
            res = self._custom_dispatch_func[target](*args, **kwargs)
        else:
            res = super().call_function(target, args, kwargs)
        if convert_to_param:
            return torch.nn.Parameter(res)
        else:
            return res

    def call_method(self, target: "Target", args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a ``call_method`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the method invocation
        """
        # args[0] is the `self` object for this method call
        self_obj, *args_tail = args

        target_method = getattr(self_obj.__class__, target)

        convert_to_parameter = False
        if target_method in (torch.Tensor.view, torch.Tensor.transpose) and isinstance(
            args[0], torch.nn.parameter.Parameter
        ):
            convert_to_parameter = True
        # Execute the method and return the result
        assert isinstance(target, str)
        res = getattr(self_obj, target)(*args_tail, **kwargs)
        if convert_to_parameter:
            return torch.nn.Parameter(res)
        else:
            return res

    def propagate(self, *args, device=None):
        """
        Run `module` via interpretation and return the result and record the
        shape of each node.
        Args:
            *args (Tensor): The sample input.
        Returns:
            Any: The value returned from executing the Module
        """

        # wrap_fn = lambda elem: MetaTensor(elem, device=device)
        def wrap_fn(elem, device=device):
            if isinstance(elem, torch.Tensor):
                return MetaTensor(elem, device=device)
            else:
                return elem

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
