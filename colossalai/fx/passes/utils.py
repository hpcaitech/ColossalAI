import torch
from typing import Dict
from torch.fx.node import Node, map_arg
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from colossalai.pipeline.middleware import Partition, PartitionInputVal, PartitionOutputVal, Topo

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
                comm_size += n.meta['tensor_meta'].numel
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
        if node.op == 'output':
            map_arg(node.args, lambda n: input_nodes.setdefault(n))
            map_arg(node.kwargs, lambda n: input_nodes.setdefault(n))
    placeholder_nodes = []
    for node in input_nodes.keys():
        if node.op == 'placeholder':
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
        if node.op == 'output':
            continue
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
            if node.op == 'output':
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

    assert node.graph.owning_module is not None, 'Cannot find the owning_module for node.graph, please make sure the graph is associated with a GraphModule object'
    assert node.op == 'call_module', f'Expected node.op to be call_module, but found {node.op}'
    module = node.graph.owning_module.get_submodule(node.target)
    return module

def partition_name_to_id(partition_name, is_input=False, is_output=False):
    if is_input:
        partition_id = 0
    elif is_output:
        partition_id = 1
    else:
        prefix = 'submod_'
        partition_id = int(partition_name.split(prefix)[-1]) + 2
    return partition_id

def find_input_in_partition(node, partitions, input_partitions=None):
    p_input_val = None
    direct_def = not node.name.startswith('getitem')
    # search in input
    if direct_def and input_partitions is not None:
        partition_id = partition_name_to_id('', is_input=True)
        for i, input_node in enumerate(input_partitions):
            if input_node == node:
                p_input_val = PartitionInputVal(partition_id=partition_id, offset=i)
                return p_input_val
    # search submod in mid part
    if direct_def:
        for partition in partitions:
            if partition == node:
                partition_id = partition_name_to_id(partition.name)
                p_input_val = PartitionInputVal(partition_id=partition_id, offset=0)
                return p_input_val
    # search temporary value in graph
    else:
        for partition in partitions:
            for offset, mid_val in enumerate(partition.users):
                if mid_val == node:
                    partition_id = partition_name_to_id(partition.name)
                    p_input_val = PartitionInputVal(partition_id=partition_id, offset=offset)
                    return p_input_val
        
    return p_input_val
        
def find_output_in_partition(node, partitions, output_partitions=None):
    p_output_val = PartitionOutputVal()
    for user in node.users:
        direct_use = not user.name.startswith('getitem')
        # user is mid partition
        for partition in partitions:
            # direct call
            if direct_use:
                if user == partition:
                    partition_id = partition_name_to_id(partition.name)
                    for i, arg in enumerate(partition.args):
                        if arg == node:
                            p_output_val.add(partition_id=partition_id, offset=i)
                            break
            # getitem call
            else:
                if user in partition.args:
                    partition_id = partition_name_to_id(partition.name)
                    for i, arg in enumerate(partition.args):
                        if arg == user:
                            p_output_val.add(partition_id=partition_id, offset=i)
                            break
        
        # user is output
        if output_partitions is not None:
            output_node = output_partitions[0]
            if user.op == output_node.op:
                output_keys = {}
                partition_id = partition_name_to_id('', is_output=True)
                torch.fx.graph.map_arg(output_node.args[0], lambda n: output_keys.setdefault(n))
                for i, arg in enumerate(output_keys):
                    if arg == node:
                        p_output_val.add(partition_id=partition_id, offset=i)
                        break
    return p_output_val

def get_topology(gm: GraphModule):
    topo = Topo()
    
    topo_partitions = []
    topo_output_partition = Partition()
    
    input_partitions = []
    partitions = []
    output_partitions = []
    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            input_partitions.append(node)
        elif node.name.startswith('submod_'):
            partitions.append(node)
        elif node.op == 'output':
            output_partitions.append(node)

    # set output for input_partition
    topo_input_partition = Partition()
    for partition in input_partitions:
        cur_node = partition
        p_output_val = find_output_in_partition(cur_node, partitions, output_partitions)
        topo_input_partition.add_output_val(p_output_val)
    topo.set_partitions(partition_id=0, partition=topo_input_partition)
    topo.set_input_partition(partition_id=0)
    
    for i, partition in enumerate(partitions):
        topo_mid_partition = Partition()
        # set input for submodule
        for arg in partition.args:
            cur_node = arg
            p_input_val = find_input_in_partition(cur_node, partitions, input_partitions)
            topo_mid_partition.add_input_val(p_input_val)
        # set output for submodule
        direct_use = True
        for user in partition.users:
            if user.name.startswith('getitem'):
                direct_use = False
                break
        if direct_use:
            cur_node = partition
            p_output_val = find_output_in_partition(cur_node, partitions, output_partitions)
            topo_mid_partition.add_output_val(p_output_val)
        else:
            for user in partition.users:
                cur_node = user
                p_output_val = find_output_in_partition(cur_node, partitions, output_partitions)
                topo_mid_partition.add_output_val(p_output_val)  
        topo.set_partitions(partition_id=i+2, partition=topo_mid_partition)
        
    # set input for output_partition
    for partition in output_partitions:
        topo_output_partition = Partition()
        torch.fx.graph.map_arg(partition.args[0], lambda n: topo_output_partition.add_input_val(
            find_input_in_partition(n, partitions, input_partitions)))
    topo.set_partitions(partition_id=1, partition=topo_output_partition)
    topo.set_output_partition(partition_id=1)

    return topo