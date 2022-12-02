from torch.fx.graph_module import GraphModule
from colossalai.pipeline.middleware.topo import Partition, PartitionInputVal, PartitionOutputVal, Topo
import torch

def partition_name_to_id(partition_name, is_input=False, is_output=False):
    if is_input:
        partition_id = 0
    elif is_output:
        partition_id = 1
    else:
        prefix = 'submod_'
        partition_id = int(partition_name.split(prefix)[-1]) + 2
    return partition_id

# There are two kinds of def in fx.graph
# 1. non direct_use & non direct_def, which means the output is used by next partition with a temporary mid value.
#    e.g. submod1 = call_module(...)
#         temporary_val = submod1[0]
#         submod2 = call_module(temporary_val, ...)
# 2. direct_use & direct_def, which means the output is used by next partition directly.
#    e.g. submod1 = call_module(...)
#         submod2 = call_module(submod1, ...)
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
        else:
            continue

    # set output for input_partition
    topo_input_partition = Partition()
    for partition in input_partitions:
        cur_node = partition
        p_output_val = find_output_in_partition(cur_node, partitions, output_partitions)
        topo_input_partition.add_output_val(p_output_val)
    topo.set_partitions(partition_id=0, partition=topo_input_partition)
    topo.set_input_partition_id(partition_id=0)
    
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
    topo.set_output_partition_id(partition_id=1)

    return topo