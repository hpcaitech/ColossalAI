import torch
from typing import Dict, Set
from torch.fx.node import Node, map_arg


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
    parent_node_names = [n.name for n in prev_partition.graph.nodes]
    for node in next_partition.graph.nodes:
        input_nodes: Dict[Node, None] = {}
        map_arg(node.args, lambda n: input_nodes.setdefault(n))
        map_arg(node.kwargs, lambda n: input_nodes.setdefault(n))
        for n in input_nodes:
            if n.name in parent_node_names and n not in visited_nodes:
                comm_size += n.meta['tensor_meta'].numel
                visited_nodes.add(n)
    return comm_size
