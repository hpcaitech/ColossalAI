from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.fx
from torch.autograd.graph import saved_tensors_hooks
from torch.autograd.profiler_util import _format_memory, _format_time
from torch.fx import GraphModule
from torch.fx.node import Argument, Node, Target
from torch.utils._pytree import tree_map

from siu._subclasses import MetaTensor, flop_count
from siu.fx.node_util import MetaInfo


def _format_flops(flops: float) -> str:
    if flops > 1e12:
        return f'{flops / 1e12:.2f} TFLOPs'
    elif flops > 1e9:
        return f'{flops / 1e9:.2f} GFLOPs'
    elif flops > 1e6:
        return f'{flops / 1e6:.2f} MFLOPs'
    elif flops > 1e3:
        return f'{flops / 1e3:.2f} kFLOPs'
    return f'{flops} FLOPs'


class sim_env(saved_tensors_hooks):

    def __init__(self):
        super().__init__(self.pack_hook, self.unpack_hook)
        self.ctx = {}

    def pack_hook(self, tensor: torch.Tensor):
        self.ctx[tensor.data_ptr()] = tensor._tensor if hasattr(tensor, '_tensor') else tensor
        return tensor

    def unpack_hook(self, tensor):
        return tensor


def _denormalize_tuple(t: Tuple[int, ...]) -> Tuple[int, ...]:
    return t[0] if len(t) == 1 else t


def _normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


def _current_device(module):
    return next(module.parameters()).device


class GraphProfile(torch.fx.Interpreter):
    """
    Execute an FX graph Node-by-Node and record the meta data of the result
    into the corresponding node.

    Usage:
        >>> model = MyModule()
        >>> x = torch.rand(10, 10)
        >>> gm = colossalai.fx.symbolic_trace(model, meta_args = {'x': x}})
        >>> shape_interp = ShapeProp(gm)    # must do this first
        >>> shape_interp.propagate(x)
        >>> profile_interp = GraphProfile(gm)
        >>> profile_interp.propagate(x)

    Args:
        module (GraphModule): The module to be executed

    Hints:
        If you want to add a new graph profile rule, you can do so by
        adding a new method to this class with the ``@register_profile_impl``
        decorator. The method should take (*args, **kwargs) instance as its
        input and generate output.

        For example, if you want to add a shape propagation rule for
        ``all_reduce``, you can do so by adding a new method
        to this class with the ``@register_profile_impl`` decorator:

        >>> @register_profile_impl(all_reduce)
        >>> def all_reduce_profile_impl(*args, **kwargs):
        >>>     return 0, 0, all_reduce_cost(*args, **kwargs), 0
    """
    _profileable = [
        'call_function',
        'call_module',
        'call_method',    # FIXME: call_method encountered error
    ]
    _custom_profile_impl = {}

    def __init__(self, module: GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)
        self.ctx = {}

    def run(self, *args, initial_env: Optional[Dict[Node, Any]] = None, enable_io_processing: bool = True) -> Any:
        """
        Run `module` via interpretation and return the result.

        Args:
            *args: The arguments to the Module to run, in positional order
            initial_env (Optional[Dict[Node, Any]]): An optional starting environment for execution.
                This is a dict mapping `Node` to any value. This can be used, for example, to
                pre-populate results for certain `Nodes` so as to do only partial evaluation within
                the interpreter.
            enable_io_processing (bool): If true, we process the inputs and outputs with graph's process_inputs and
                process_outputs function first before using them.

        Returns:
            Any: The value returned from executing the Module
        """
        self.env = initial_env if initial_env else {}

        # Positional function args are consumed left-to-right by
        # `placeholder` nodes. Use an iterator to keep track of
        # position and extract those values.
        if enable_io_processing:
            args = self.module.graph.process_inputs(*args)
        self.args_iter: Iterator[Any] = iter(args)

        for node in self.module.graph.nodes:

            self.env[node] = self.run_node(node)

            if self.garbage_collect_values:
                for to_delete in self.user_to_last_uses.get(node, []):
                    del self.env[to_delete]

            if node.op == 'output':
                output_val = self.env[node]
                return self.module.graph.process_outputs(output_val) if enable_io_processing else output_val

    def run_node(self, n: torch.fx.Node) -> Any:
        """
        Run a specific node ``n`` and profile its execution time and memory usage.
        Calls into call_function, call_method, and call_module only.

        Args:
            n (Node): The Node to profile

        Returns:
            Any: The output of the node

        Raises:
            RuntimeError: If the node is not profileable.
        """
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        n_info = MetaInfo(n)

        if n.op in self._profileable:
            try:
                inner_hook = sim_env()
                with inner_hook:
                    (
                        n_info.fwd_flop,
                        n_info.bwd_flop,
                        n_info.fwd_comm,
                        n_info.bwd_comm,
                    ) = getattr(self, n.op)(n.target, args, kwargs)
                n_info.local_ctx = inner_hook.ctx
                self.ctx.update(inner_hook.ctx)
                # from siu.fx.node_util import intersect, compute_size_in_bytes
                # input_ctx = {i.data_ptr(): i for i in n_info.inputs if isinstance(i, torch.Tensor)}
                # print(_format_memory(compute_size_in_bytes(input_ctx)))
                # print(n, _format_memory(compute_size_in_bytes(inner_hook.ctx)))
                # print(_format_memory(compute_size_in_bytes(intersect(input_ctx, self.ctx))))
                # print(_format_memory(compute_size_in_bytes(intersect(input_ctx, inner_hook.ctx))))
            except Exception as e:
                raise RuntimeError(
                    f'Error {str(e)} occurred when profiling node {n}, node.target = {n.target}. '\
                    f'Please refer to function\'s docstring to register the relevant profile_impl for this node!'
                ) from e
        n_info.global_ctx = self.ctx

        # retain the autograd graph
        for param in self.module.parameters():
            param.grad = None

        crit = lambda x: x.data_ptr() in self.ctx if isinstance(x, torch.Tensor) else False
        n_info.is_alias = _normalize_tuple(tree_map(crit, n_info.outputs))
        return _denormalize_tuple(n_info.outputs)

    def call_function(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a ``call_function`` node and return the profiling result.
        Dispatch to _custom_profile_impl`` if ``call_function`` should be
        profiled in a user-defined behavior.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Tuple(int): (fwd_flop, bwd_flop, fwd_comm, bwd_comm)
        """
        assert not isinstance(target, str)

        # Dispatch the impl for profiling, default will be ``flop_count``
        if target in self._custom_profile_impl:
            return self._custom_profile_impl[target](*args, **kwargs)
        else:
            return *flop_count(target, *args, **kwargs), 0, 0

    def call_method(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a ``call_method`` node and return the profiling result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Tuple(int): (fwd_flop, bwd_flop, fwd_comm, bwd_comm)
        """
        # Execute the method and return the result
        assert isinstance(target, str)
        return *flop_count(getattr(torch.Tensor, target), *args, **kwargs), 0, 0

    def call_module(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a ``call_module`` node and return the profiling result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Tuple(int): (fwd_flop, bwd_flop, fwd_comm, bwd_comm)
        """
        # Retrieve executed args and kwargs values from the environment

        # Execute the method and return the result
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        return *flop_count(submod, *args, **kwargs), 0, 0

    def fetch_initial_env(self, device=None) -> Dict[Node, Any]:
        """
        Fetch ``initial_env`` for execution. This is because ``ShapeProp``
        has already attached outputs of each ``Node`` to its ``MetaInfo``.

        Args:
            device (torch.device): The device to place the execution, default to ``None``

        Returns:
            Dict[Node, Any]: The initial environment for execution
        """
        initial_env = {}
        for n in self.module.graph.nodes:
            initial_env[n] = _denormalize_tuple(MetaInfo(n).outputs)
        return initial_env

    def propagate(self, *args, device=None):
        """
        Run `module` via interpretation and profile the execution
        of each ``Node``.

        Args:
            *args (Tensor): The sample input, not used
            device (torch.device): The device to place the execution, default to ``None``

        Returns:
            Any: The value returned from executing the Module
        """
        initial_env = self.fetch_initial_env(device)

        return self.run(initial_env=initial_env)

    def summary(self) -> str:
        """
        Summarizes the profiled statistics of the `GraphModule` in
        tabular format. Note that this API requires the ``tabulate`` module
        to be installed.

        Returns:
            str: The summary of the profiled statistics
        """
        # https://github.com/pytorch/pytorch/blob/master/torch/fx/graph.py
        try:
            from tabulate import tabulate
        except ImportError:
            print("`summary` relies on the library `tabulate`, "
                  "which could not be found on this machine. Run `pip "
                  "install tabulate` to install the library.")

        # Build up a list of summary information for each node
        node_summaries: List[List[Any]] = []

        for node in self.module.graph.nodes:
            node: Node
            n_info = MetaInfo(node)
            node_summaries.append([
                node.op,
                str(node),
                _format_memory(n_info.total_size),
                _format_memory(n_info.output_size),
                _format_memory(n_info.temp_size),
                _format_memory(n_info.param_size),
                _format_memory(n_info.backward_size),
                _format_flops(n_info.fwd_flop),
                _format_flops(n_info.bwd_flop),
            ])

        # Use the ``tabulate`` library to create a well-formatted table
        # presenting our summary information
        headers: List[str] = [
            'Op type',
            'Op',
            'Total size',
            'Output size',
            'Temp size',
            'Param size',
            'Backward size',
            'Fwd FLOPs',
            'Bwd FLOPs',
        ]

        return tabulate(node_summaries, headers=headers, stralign='right')


def graph_profile_pass(module: GraphModule, *args, verbose=False) -> GraphModule:
    """
    Run ``module`` via interpretation and profile the execution
    of each ``Node``.

    Args:
        module (GraphModule): The GraphModule to profile
        *args (Any): The sample input, not used
        verbose (bool): Whether to print the profiling summary

    Returns:
        GraphModule: The same GraphModule with profiling information
    """
    interp = GraphProfile(module)
    interp.propagate(*args, device=_current_device(module))
    if verbose:
        print(interp.summary())
    return module
