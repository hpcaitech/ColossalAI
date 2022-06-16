import torch
from torch.fx.node import map_arg
from torch.fx.node import Node
from torch.fx.passes.split_module import split_module

import colossalai
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc


def all_gather_function(input_):
    world_size = gpc.get_world_size(ParallelMode.PARALLEL_1D)
    rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    group = gpc.get_group(ParallelMode.PARALLEL_1D)
    torch.distributed.all_gather(tensor_list, input_, group=group)
    output = torch.cat(tensor_list, dim=-1).contiguous()
    return output


def all_reduce_function(input_):
    if gpc.get_world_size(ParallelMode.PARALLEL_1D) == 1:
        return input_
    torch.distributed.all_reduce(input_, group=gpc.get_group(ParallelMode.PARALLEL_1D))
    return input_


def weight_split(weight, dim):
    #TODO: this function will be refactored by using ColoTensor dist_spec when a stable reshaper feature is ready to use.
    num_partition = gpc.get_world_size(ParallelMode.TENSOR)
    shape = weight.shape
    length = shape[dim] // num_partition
    sharded_weight_list = []
    for i in range(num_partition):
        sharded_weight_list.append(weight.narrow(dim, i * length, length))
    return sharded_weight_list[gpc.get_local_rank(ParallelMode.PARALLEL_1D)]


def replace_all_uses_except_replaced(node, replace_node):
    """
        Replace all uses of ``node`` in the Graph with the Node ``replace_node``,
        except the user of ``node`` is ``replace_node``.

        Args:

            replace_node (Node): The node to replace all uses of ``node`` with.

        Returns:

            The list of Nodes on which this change was made.
    """
    to_process = list(node.users)
    for use_node in to_process:
        if use_node == replace_node:
            continue

        def may_replace_node(n):
            if n == node:
                return replace_node
            else:
                return n

        new_args = map_arg(use_node.args, may_replace_node)
        new_kwargs = map_arg(use_node.kwargs, may_replace_node)
        use_node._args = new_args
        use_node._kwargs = new_kwargs
        for old_use in use_node._input_nodes.keys():
            old_use.users.pop(use_node)
        use_node._input_nodes = {}
        map_arg(use_node._args, lambda n: use_node._input_nodes.setdefault(n))
        map_arg(use_node._kwargs, lambda n: use_node._input_nodes.setdefault(n))
        for new_use in use_node._input_nodes.keys():
            new_use.users.setdefault(use_node)
    return to_process


def column_shard_linear_pass(gm: torch.fx.GraphModule):
    mod_graph = gm.graph
    for node in mod_graph.nodes:
        if node.op == "call_module":
            target_module = node.graph.owning_module.get_submodule(node.target)
            if isinstance(target_module, torch.nn.Linear):
                target_module.weight.data = weight_split(target_module.weight.data, dim=0)
                if target_module.bias is not None:
                    target_module.bias.data = weight_split(target_module.bias.data, dim=0)

                # inserting communication node after the sharded linear node
                with mod_graph.inserting_after(node):
                    new_node = mod_graph.create_node('call_function', all_gather_function, args=(node,))
                    replace_all_uses_except_replaced(node, new_node)
    gm.recompile()
    return gm


def row_shard_linear_pass(gm: torch.fx.GraphModule):
    mod_graph = gm.graph
    for node in mod_graph.nodes:
        if node.op == "call_module":
            target_module = node.graph.owning_module.get_submodule(node.target)
            if isinstance(target_module, torch.nn.Linear):
                target_module.weight.data = weight_split(target_module.weight.data, dim=-1)

                # insert input sharding node before the sharded linear node
                with mod_graph.inserting_before(node):
                    input_node_list = list(node._input_nodes.keys())
                    assert len(input_node_list) == 1, 'linear forward must have and only have one input tensor.'
                    input_node = input_node_list[0]
                    new_input_node = mod_graph.create_node('call_function', weight_split, args=(input_node, -1))
                    replace_all_uses_except_replaced(input_node, new_input_node)

                # inserting communication node after the sharded linear node
                with mod_graph.inserting_after(node):
                    new_node = mod_graph.create_node('call_function', all_reduce_function, args=(node,))
                    replace_all_uses_except_replaced(node, new_node)
    gm.recompile()
    return gm


#TODO: add elementwise op process pass, then we can try to use column and row mixed strategy.
