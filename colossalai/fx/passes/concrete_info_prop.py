from dataclasses import asdict
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.fx
from torch.fx.node import Argument, Node, Target
from torch.utils._pytree import tree_flatten

from colossalai.fx._compatibility import compatibility
from colossalai.fx.profiler import GraphInfo, profile_function, profile_method, profile_module


@compatibility(is_backward_compatible=True)
class ConcreteInfoProp(torch.fx.Interpreter):
    """
    Execute an FX graph Node-by-Node with concrete tensor and record the memory
    usage, execution time of forward and backward, and type of the result into
    the corresponding node.

    Usage:
        BATCH_SIZE = 2
        DIM_IN = 4
        DIM_HIDDEN = 16
        DIM_OUT = 16
        model = torch.nn.Sequential(
            torch.nn.Linear(DIM_IN, DIM_HIDDEN),
            torch.nn.Linear(DIM_HIDDEN, DIM_OUT),
            ).cuda()
        input_sample = torch.rand(BATCH_SIZE, DIM_IN, device="cuda")
        gm = symbolic_trace(model)
        interp = ConcreteInfoProp(gm)
        interp.run(input_sample)
        print(interp.summary(unit='kb'))


        output of above code is
        Op type       Op             Forward time             Backward time    SAVE_FWD_IN    FWD_OUT    FWD_TMP    BWD_OUT    BWD_TMP
        -----------  -------  -----------------------  ------------------------  -------------  ---------  ---------  ---------  ---------
        placeholder  input_1                    0.0 s                     0.0 s          False    0.00 KB    0.00 KB    0.00 KB    0.00 KB
        call_module       _0  0.0003993511199951172 s     0.00706791877746582 s          False    0.50 KB    0.00 KB    0.03 KB    0.66 KB
        call_module       _1   6.29425048828125e-05 s  0.00018286705017089844 s          False    0.50 KB    0.00 KB    0.12 KB    0.81 KB
             output   output                    0.0 s                     0.0 s           True    0.00 KB    0.00 KB    0.00 KB    0.00 KB
    Args:
         module (GraphModule): The module to be executed

    """

    _is_proped: bool = False

    def run(self, *args, initial_env: Optional[Dict[Node, Any]] = None, enable_io_processing: bool = True) -> Any:
        """Customized run for ConcreteInfoProp
        We need to store the device in self.device

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

        flatten_args, _ = tree_flatten(args)
        self.device = next(item for item in flatten_args if hasattr(item, "device")).device
        return super().run(*args, initial_env, enable_io_processing)

    @compatibility(is_backward_compatible=True)
    def run_node(self, n: Node) -> Any:
        """
        Run a specific node ``n`` and return the result.
        Calls into placeholder, get_attr, call_function,
        call_method, call_module, or output depending
        on ``node.op``

        Args:
            n (Node): The Node to execute

        Returns:
            Any: The result of executing ``n``
        """
        self._is_proped = True
        result, meta_info = super().run_node(n)

        n.meta = {**n.meta, **asdict(meta_info)}    # extend MetaInfo to `n.meta`
        # TODO: the attribute node_size should be removed in the future
        setattr(n, 'node_size', n.meta.get('fwd_mem_tmp', 0) + n.meta.get('fwd_mem_out', 0))
        n.meta['type'] = type(result)

        # retain the autograd graph
        for param in self.module.parameters():
            param.grad = None

        return result

    # Main Node running APIs
    @compatibility(is_backward_compatible=True)
    def placeholder(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a ``placeholder`` node. Note that this is stateful:
        ``Interpreter`` maintains an internal iterator over
        arguments passed to ``run`` and this method returns
        next() on that iterator.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            result (Any): The argument value that was retrieved
            meta_info (MetaInfo): The memory cost and forward & backward time.
        """
        return super().placeholder(target, args, kwargs), GraphInfo()

    @compatibility(is_backward_compatible=True)
    def get_attr(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a ``get_attr`` node. Will retrieve an attribute
        value from the ``Module`` hierarchy of ``self.module``.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return:
            result (Any): The argument value that was retrieved
            meta_info (MetaInfo): The memory cost and FLOPs estimated with `MetaTensor`.
        """
        return super().get_attr(target, args, kwargs), GraphInfo()

    @compatibility(is_backward_compatible=True)
    def call_function(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a ``call_function`` node with meta tensor and return the result and its meta profile.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            result (Any): The argument value that was retrieved
            meta_info (MetaInfo): The memory cost and forward & backward time.
        """
        assert not isinstance(target, str)
        return profile_function(target, self.device)(*args, **kwargs)

    @compatibility(is_backward_compatible=True)
    def call_method(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a ``call_method`` node with meta tensor and return the result and its meta profile.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            result (Any): The argument value that was retrieved
            meta_info (MetaInfo): The memory cost and forward & backward time.
        """
        return profile_method(target, self.device)(*args, **kwargs)

    @compatibility(is_backward_compatible=True)
    def call_module(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute a ``call_module`` node with meta tensor and return the result and its meta profile.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            result (Any): The argument value that was retrieved
            meta_info (MetaInfo): The memory cost and forward & backward time.
        """
        # Retrieve executed args and kwargs values from the environment
        # Execute the method and return the result
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        return profile_module(submod, self.device)(*args, **kwargs)

    @compatibility(is_backward_compatible=True)
    def output(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute an ``output`` node. This really just retrieves
        the value referenced by the ``output`` node and returns it.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return:
            result (Any): The argument value that was retrieved
            meta_info (MetaInfo): The memory cost and forward & backward time.
        """
        return args[0], GraphInfo(save_fwd_in=True)

    def propagate(self, *args):
        """
        Run `module` via interpretation and return the result and
        record the shape and type of each node.

        Args:
            *args (Tensor): the sample input.

        Returns:
            Any: The value returned from executing the Module
        """
        return super().run(*args)

    def summary(self, unit: str = 'MB') -> str:
        """
        Summarizes the memory and FLOPs statistics of the `GraphModule` in
        tabular format. Note that this API requires the ``tabulate`` module
        to be installed.
        """
        # https://github.com/pytorch/pytorch/blob/master/torch/fx/graph.py
        try:
            from tabulate import tabulate
        except ImportError:
            print("`summary` relies on the library `tabulate`, "
                  "which could not be found on this machine. Run `pip "
                  "install tabulate` to install the library.")

        assert self._is_proped, "Please call `interp.run(input)` before calling `interp.summary()`."

        # Build up a list of summary information for each node
        node_summaries: List[List[Any]] = []

        def mem_repr(mem: int) -> str:
            unit_divisor_map = {
                'kb': 1024,
                'mb': 1024**2,
                'gb': 1024**3,
                'tb': 1024**4,
            }
            return f"{mem / unit_divisor_map[unit.lower()]:.2f} {unit.upper()}"

        def time_repr(time: float):
            return f"{time:,} s"

        for node in self.module.graph.nodes:
            node: Node
            node_summaries.append([
                node.op,
                str(node),
                time_repr(node.meta['fwd_time']),
                time_repr(node.meta['bwd_time']),
                node.meta['save_fwd_in'],
                mem_repr(node.meta['fwd_mem_out']),
                mem_repr(node.meta['fwd_mem_tmp']),
                mem_repr(node.meta['bwd_mem_out']),
                mem_repr(node.meta['bwd_mem_tmp']),
            ])

        # Use the ``tabulate`` library to create a well-formatted table
        # presenting our summary information
        headers: List[str] = [
            'Op type',
            'Op',
            'Forward time',
            'Backward time',
            'SAVE_FWD_IN',
            'FWD_OUT',
            'FWD_TMP',
            'BWD_OUT',
            'BWD_TMP',
        ]

        return tabulate(node_summaries, headers=headers, stralign='right')
