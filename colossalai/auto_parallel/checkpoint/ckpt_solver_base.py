from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, List

import torch
from torch.fx import Graph, Node

from colossalai.auto_parallel.passes.runtime_apply_pass import (
    runtime_apply,
    runtime_apply_for_iterable_object,
    runtime_comm_spec_apply,
)
from colossalai.fx.codegen.activation_checkpoint_codegen import ActivationCheckpointCodeGen

__all___ = ['CheckpointSolverBase']


def _copy_output(src: Graph, dst: Graph):
    """Copy the output node from src to dst"""
    for n_src, n_dst in zip(src.nodes, dst.nodes):
        if n_src.op == 'output':
            n_dst.meta = n_src.meta


def _get_param_size(module: torch.nn.Module):
    """Get the size of the parameters in the module"""
    return sum([p.numel() * torch.tensor([], dtype=p.dtype).element_size() for p in module.parameters()])


class CheckpointSolverBase(ABC):

    def __init__(
        self,
        graph: Graph,
        free_memory: float = -1.0,
        requires_linearize: bool = False,
        cnode: List[str] = None,
        optim_multiplier: float = 1.0,
    ):
        """``CheckpointSolverBase`` class will integrate information provided by the components
        and use an existing solver to find a possible optimal strategies combination for target
        computing graph.

        Existing Solvers:
            Chen's Greedy solver: https://arxiv.org/abs/1604.06174  (CheckpointSolverChen)
            Rotor solver: https://hal.inria.fr/hal-02352969  (CheckpointSolverRotor)

        Args:
            graph (Graph): The computing graph to be optimized.
            free_memory (float): Memory constraint for the solution.
            requires_linearize (bool): Whether the graph needs to be linearized.
            cnode (List[str], optional): Common node List, should be the subset of input. Default to None.
            optim_multiplier (float, optional): The multiplier of extra weight storage for the
            ``torch.optim.Optimizer``. Default to 1.0.

        Warnings:
            Meta information of the graph is required for any ``CheckpointSolver``.
        """
        # super-dainiu: this graph is a temporary graph which can refer to
        # the owning module, but we will return another deepcopy of it after
        # the solver is executed.
        self.graph = deepcopy(graph)
        self.graph.owning_module = graph.owning_module
        _copy_output(graph, self.graph)
        self.graph.set_codegen(ActivationCheckpointCodeGen())

        # check if has meta information
        if any(len(node.meta) == 0 for node in self.graph.nodes):
            raise RuntimeError(
                "Nodes meta information hasn't been prepared! Please extract from graph before constructing the solver!"
            )

        # parameter memory = parameter size + optimizer extra weight storage
        self.free_memory = free_memory - _get_param_size(self.graph.owning_module) * (optim_multiplier + 1)
        self.cnode = cnode
        self.requires_linearize = requires_linearize
        if self.requires_linearize:
            self.node_list = self._linearize_graph()
        else:
            self.node_list = self.get_node_list()

    @abstractmethod
    def solve(self):
        """Solve the checkpointing problem and return the solution.
        """
        pass

    def get_node_list(self):
        """Get the node list.
        """
        return [[node] for node in self.graph.nodes]

    def _linearize_graph(self) -> List[List[Node]]:
        """Linearizing the graph

        Args:
            graph (Graph): The computing graph to be optimized.

        Returns:
            List[List[Node]]: List of list, each inside list of Node presents
            the actual 'node' in linearized manner.

        Remarks:
            Do merge the inplace ops and shape-consistency ops into the previous node.
        """

        # Common nodes are type of nodes that could be seen as attributes and remain
        # unchanged throughout the whole model, it will be used several times by
        # different blocks of model, so that it is hard for us to linearize the graph
        # when we encounter those kinds of nodes. We let users to annotate some of the
        # input as common node, such as attention mask, and the followings are some of
        # the ops that could actually be seen as common nodes. With our common node prop,
        # we could find some of the "real" common nodes (e.g. the real attention mask
        # used in BERT and GPT), the rule is simple, for node who's parents are all common
        # nodes or it's op belongs to the following operations, we view this node as a
        # newly born common node.
        # List of target name that could be seen as common node
        common_ops = ["getattr", "getitem", "size"]

        def _is_cop(target: Any) -> bool:
            """Check if an op could be seen as common node

            Args:
                target (Any): node target

            Returns:
                bool
            """

            if isinstance(target, str):
                return target in common_ops
            else:
                return target.__name__ in common_ops

        def _is_sink() -> bool:
            """Check if we can free all dependencies

            Returns:
                bool
            """

            def _is_inplace(n: Node):
                """Get the inplace argument from ``torch.fx.Node``
                """
                inplace = False
                if n.op == "call_function":
                    inplace = n.kwargs.get("inplace", False)
                elif n.op == "call_module":
                    inplace = getattr(n.graph.owning_module.get_submodule(n.target), "inplace", False)
                return inplace

            def _is_shape_consistency(n: Node):
                """Check if this node is shape-consistency node (i.e. ``runtime_apply`` or ``runtime_apply_for_iterable_object``)
                """
                return n.target in [runtime_apply, runtime_apply_for_iterable_object, runtime_comm_spec_apply]

            return not sum([v for _, v in deps.items()]) and not any(map(_is_inplace, n.users)) and not any(
                map(_is_shape_consistency, n.users))

        # make sure that item in cnode is valid
        if self.cnode:
            for name in self.cnode:
                try:
                    assert next(node for node in self.graph.nodes if node.name == name).op == "placeholder", \
                    f"Common node {name} is not an input of the model."
                except StopIteration:
                    raise ValueError(f"Common node name {name} not in graph.")

        else:
            self.cnode = []

        deps = {}
        node_list = []
        region = []

        for n in self.graph.nodes:
            if n.op != "placeholder" and n.op != "output":
                for n_par in n.all_input_nodes:
                    if n_par.op != "placeholder" and n_par.name not in self.cnode:
                        deps[n_par] -= 1
                region.append(n)

                # if the node could free all dependencies in graph
                # we could begin a new node
                if _is_sink():
                    node_list.append(region)
                    region = []

                # propagate common node attr if possible
                if len(n.all_input_nodes) == len([node for node in n.all_input_nodes if node.name in self.cnode
                                                 ]) or _is_cop(n.target):
                    self.cnode.append(n.name)
                else:
                    deps[n] = len([user for user in n.users if user.op != "output"])
        return node_list
