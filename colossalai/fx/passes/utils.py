import torch
from typing import Dict, Set
from torch.fx.node import Node, map_arg
from torch.fx.graph import Graph


def get_comm_size(prev_partition, next_partition):
    """Given two partitions (parent and child),
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
    parent_node_names = [n.name for n in parent_partition.graph.nodes]
    for node in child_partition.graph.nodes:
        input_nodes: Dict[Node, None] = {}
        map_arg(node.args, lambda n: input_nodes.setdefault(n))
        map_arg(node.kwargs, lambda n: input_nodes.setdefault(n))
        for n in input_nodes:
            if n.name in parent_node_names and n not in visited_nodes:
                comm_size += n.meta['tensor_meta'].numel
                visited_nodes.add(n)
    return comm_size


def get_leaf(graph: Graph):
    """Given a graph, get leaf node of this graph.

    Note: This method will get the leaf nodes of given graph excluding `output` node.
    """
    input_nodes: Dict[Node, None] = {}
    for node in graph.nodes:
        if node.op == 'output':
            map_arg(node.args, lambda n: input_nodes.setdefault(n))
            map_arg(node.kwargs, lambda n: input_nodes.setdefault(n))
    return list(input_nodes.keys())


def is_leaf(graph: Graph, node: Node):
    return node in get_leaf(graph)


def get_top(graph: Graph):
    """Given a graph, get top node of this graph.

    Note: This method will get the top nodes of given graph excluding `output` node.
    """
    top_node_list = set()
    for node in graph.nodes:
        is_top = False

        def _get_top(node):
            nonlocal is_top
            if node.op == 'placeholder':
                is_top = True

        map_arg(node.args, lambda n: _get_top(n))
        map_arg(node.kwargs, lambda n: _get_top(n))
        if is_top:
            top_node_list.add(node)
    return list(top_node_list)


def is_top(graph: Graph, node: Node):
    return node in get_top(graph)
