from typing import Dict

import torch
from torch.fx.graph import Graph
from torch.fx.node import Node, map_arg


def get_comm_size(prev_partition, next_partition):
    """
    Given two partitions (parent and child),
    calculate the communication size between the two.
    """
    # Keep tracking the communication size between parent and child
    comm_size = 0
    # Keep tracking all the counted node
    visited_nodes = set()
    # Go through all nodes in the child partition
    # If a node has input nodes from the parent partition,
    # the output size of those input nodes will be counted
    # and added to comm_size
    parent_node_names = [n.name for n in prev_partition.graph.nodes]
    for node in next_partition.graph.nodes:
        input_nodes: Dict[Node, None] = {}
        map_arg(node.args, lambda n: input_nodes.setdefault(n))
        map_arg(node.kwargs, lambda n: input_nodes.setdefault(n))
        for n in input_nodes:
            if n.name in parent_node_names and n not in visited_nodes:
                comm_size += n.meta["tensor_meta"].numel
                visited_nodes.add(n)
    return comm_size


def get_leaf(graph: Graph):
    """
    Given a graph, return leaf nodes of this graph.
    Note: If we remove ``root`` nodes, ``placeholder`` nodes, and ``output`` nodes from fx graph,
    we will get a normal DAG. Leaf nodes in this context means leaf nodes in that DAG.
    """
    input_nodes: Dict[Node, None] = {}
    for node in graph.nodes:
        if node.op == "output":
            map_arg(node.args, lambda n: input_nodes.setdefault(n))
            map_arg(node.kwargs, lambda n: input_nodes.setdefault(n))
    placeholder_nodes = []
    for node in input_nodes.keys():
        if node.op == "placeholder":
            placeholder_nodes.append(node)
    for node in placeholder_nodes:
        input_nodes.pop(node)
    return list(input_nodes.keys())


def is_leaf(graph: Graph, node: Node):
    return node in get_leaf(graph)


def get_top(graph: Graph):
    """
    Given a graph, return top nodes of this graph.
    Note: If we remove ``root`` nodes, ``placeholder`` nodes, and ``output`` nodes from fx graph,
    we will get a normal DAG. Top nodes in this context means nodes with BFS level 0 in that DAG.
    """
    top_node_list = set()
    for node in graph.nodes:
        if node.op == "output":
            continue
        is_top = False

        def _get_top(node):
            nonlocal is_top
            if node.op == "placeholder":
                is_top = True

        map_arg(node.args, lambda n: _get_top(n))
        map_arg(node.kwargs, lambda n: _get_top(n))
        if is_top:
            top_node_list.add(node)
    return list(top_node_list)


def is_top(graph: Graph, node: Node):
    return node in get_top(graph)


def get_all_consumers(graph: Graph, node: Node):
    """
    Given a graph and a node of this graph, return all consumers of the node.

    Returns:
        List of ``Nodes`` that node appear in these nodes ``args`` and ``kwargs``.
    """
    consumer_list = []
    for n in graph.nodes:
        if node in n.all_input_nodes:
            consumer_list.append(n)
    return consumer_list


def assign_bfs_level_to_nodes(graph: Graph):
    """
    Give a graph, assign bfs level to each node of this graph excluding ``placeholder`` and ``output`` nodes.
    Example:
        class MLP(torch.nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.linear1 = torch.nn.Linear(dim, dim)
                self.linear2 = torch.nn.Linear(dim, dim)
                self.linear3 = torch.nn.Linear(dim, dim)
                self.linear4 = torch.nn.Linear(dim, dim)
                self.linear5 = torch.nn.Linear(dim, dim)
            def forward(self, x):
                l1 = self.linear1(x)
                l2 = self.linear2(x)
                l3 = self.linear3(l1)
                l4 = self.linear4(l2)
                l5 = self.linear5(l3)
                return l4, l5
        model = MLP(4)
        gm = symbolic_trace(model)
        print(gm.graph)
        assign_bfs_level_to_nodes(gm.graph)
        for node in gm.graph.nodes:
            if hasattr(node, 'bfs_level'):
                print(node.name, node.bfs_level)

    Output:
        graph():
            %x : [#users=2] = placeholder[target=x]
            %linear1 : [#users=1] = call_module[target=linear1](args = (%x,), kwargs = {})
            %linear2 : [#users=1] = call_module[target=linear2](args = (%x,), kwargs = {})
            %linear3 : [#users=1] = call_module[target=linear3](args = (%linear1,), kwargs = {})
            %linear4 : [#users=1] = call_module[target=linear4](args = (%linear2,), kwargs = {})
            %linear5 : [#users=1] = call_module[target=linear5](args = (%linear3,), kwargs = {})
            return (linear4, linear5)
        linear1 0
        linear2 0
        linear3 1
        linear4 1
        linear5 2
    """
    current_level = 0
    nodes_to_process = []

    top_nodes = get_top(graph)
    for node in top_nodes:
        node.bfs_level = current_level
        nodes_to_process.extend(get_all_consumers(graph, node))

    current_level += 1
    while nodes_to_process:
        new_process_list = []
        for node in nodes_to_process:
            if node.op == "output":
                continue
            node.bfs_level = current_level
            new_process_list.extend(get_all_consumers(graph, node))
        nodes_to_process = new_process_list
        current_level += 1


def get_node_module(node) -> torch.nn.Module:
    """
    Find the module associated with the given node.
    Args:
        node (torch.fx.Node): a torch.fx.Node object in the fx computation graph
    Returns:
        torch.nn.Module: the module associated with the given node
    """

    assert (
        node.graph.owning_module is not None
    ), "Cannot find the owning_module for node.graph, please make sure the graph is associated with a GraphModule object"
    assert node.op == "call_module", f"Expected node.op to be call_module, but found {node.op}"
    module = node.graph.owning_module.get_submodule(node.target)
    return module
