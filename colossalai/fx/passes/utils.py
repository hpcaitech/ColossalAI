import torch
from typing import Dict, Set
from torch.fx.node import Node, map_arg
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule

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

def find_def_in_partition(node, partitions, input_partitions=None, direct=False):
    # find def in input
    if input_partitions is not None:
        for placeholder in input_partitions:
            if placeholder.name == node.name:
                return 'MODEL_INPUT'
            
    # find direct def
    if direct:
        for partition in partitions:
            if node == partition:
                return partition.name
    # find def with getitem call
    else:
        for partition in partitions:
            if node in partition.users.keys():
                return partition.name

    print(f'Not found def in partition {node.name}')
    return None
        
def find_user_in_partition(node, partitions, output_partitions=None, direct=False):
    user_partition_names = []
    # find direct user
    if direct:
        for partition in partitions:
            if node == partition:
                user_partition_names.append(partition.name)
    # find user with getitem call
    else:
        for partition in partitions:
            if node in partition.args:
                user_partition_names.append(partition.name)
    
    is_output = False
    def find_output(def_node, output_node):
        nonlocal is_output
        if def_node == output_node:
            is_output = True
        
    if output_partitions is not None:
        output_node = output_partitions[0]
        torch.fx.graph.map_arg(output_node.args[0], lambda n: find_output(node, n))
    
    if is_output:
        user_partition_names.append('MODEL_OUTPUT')
    
    if len(user_partition_names) > 0:
        return user_partition_names
    
    print(f'Not found user in partition {node.name}')
    return None
    
def get_partition_depends(partition, partitions, input_partitions=None, output_partitions=None):
    # e.g. Partition2: {input: {Partition0: [sub1_1], Partition1: [sub2_0]}, output:{Output: [sub3_0]}},
    input = {}
    output = {}
    
    for offset, arg in enumerate(partition.args):
        def_partition_name = None
        if not arg.name.startswith('getitem'):
            def_partition_name = find_def_in_partition(arg, partitions, input_partitions, direct=True)
        else:
            def_partition_name = find_def_in_partition(arg, partitions, input_partitions, direct=False)
        if def_partition_name is None:
            continue
        if def_partition_name not in input:
            input[def_partition_name] = []
        input[def_partition_name].append(offset)

    offset = -1
    for user in partition.users.keys():
        user_partition_names = None
        if input_partitions is None or not user.name.startswith('getitem'):
            user_partition_names = find_user_in_partition(user, partitions, output_partitions, direct=True)
            offset = 0
        else:
            user_partition_names = find_user_in_partition(user, partitions, output_partitions, direct=False)
            offset += 1
        if user_partition_names is None:
            continue
        for user_partition_name in user_partition_names:
            if user_partition_name not in output:
                output[user_partition_name] = []
            output[user_partition_name].append(offset)
    
    return input, output, offset+1

# DAG just looks like following case.
# the int in every list represents the offset of the partition's input arg or output arg.
# {
# 'input_partition': {
#     'input_ids': {
#         'input': {}, 
#         'output': {'submod_0': [0], 'submod_1': [1]}, 
#         'output_len': 0}, 
#     'attention_mask': {
#         'input': {}, 
#         'output': {'submod_2': [0]}, 
#         'output_len': 0}}, 
# 'submod_0': {
#     'input': {'MODEL_INPUT': [0]}, 
#     'output': {'submod_1': [0], 'submod_2': [0, 1]}, 
#     'output_len': 2}, 
# 'submod_1': {
#     'input': {'submod_0': [0], 'MODEL_INPUT': [1]}, 
#     'output': {'submod_2': [0]}, 
#     'output_len': 1}, 
# 'submod_2': {
#     'input': {'MODEL_INPUT': [0], 'submod_0': [1, 2]}, 
#     'output': {'submod_3': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
#                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
#                             22, 23, 24]},
#     'output_len': 25}, 
# 'submod_3': {
#     'input': {'submod_2': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
#                                 12, 13, 14, 15, 16, 17, 18, 19, 20, 
#                                 21, 22, 23, 24]}, 
#     'output': {'MODEL_OUTPUT': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
#                                 11, 12, 13, 14, 15, 16, 17, 18, 19, 
#                                 20, 21, 22, 23, 24]}, 
#     'output_len': 25},
# 'output_partition': {
#     'input': {'logits': 'submod_3', 'past_key_values': (('submod_3', 'submod_3'), ('submod_3', 'submod_3'), 
#                                                         ('submod_3', 'submod_3'), ('submod_3', 'submod_3'), 
#                                                         ('submod_3', 'submod_3'), ('submod_3', 'submod_3'), 
#                                                         ('submod_3', 'submod_3'), ('submod_3', 'submod_3'), 
#                                                         ('submod_3', 'submod_3'), ('submod_3', 'submod_3'), 
#                                                         ('submod_3', 'submod_3'), ('submod_3', 'submod_3'))}, 
#     'output': {}, 'output_len': 0}
# }

# TODO(jiangziyue) Define a Class for DAG.
def get_DAG(gm: GraphModule):
    DAG = {}
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

    for partition in input_partitions:
        DAG_node = {'input': {}, 'output': {}, 'output_len': 1}
        _, output, _ = get_partition_depends(partition, partitions, None, output_partitions)
        DAG_node['output'] = output
        if 'input_partition' not in DAG:
            DAG['input_partition'] = {}
        DAG['input_partition'][partition.name] = DAG_node
    
    for partition in partitions:
        DAG_node = {'input': {}, 'output': {}}
        DAG_node['input'], DAG_node['output'], DAG_node['output_len'] = get_partition_depends(partition, partitions, input_partitions, output_partitions)
        DAG[partition.name] = DAG_node
        
    for partition in output_partitions:
        DAG_node = {'input': {}, 'output': {}, 'output_len': 0}
        DAG_node['input'] = torch.fx.graph.map_arg(partition.args[0], lambda n: find_def_in_partition(n, partitions, input_partitions))
        DAG['output_partition'] = DAG_node

    return DAG