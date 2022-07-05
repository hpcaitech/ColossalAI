import torch
from torch.fx.node import map_arg
from torch.fx.node import Node
from torch.fx.passes.split_module import split_module

import colossalai

from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.tensor import TensorSpec, distspec


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


def weight_split(weight):
    # split tensor with tensor_spec
    assert isinstance(weight, colossalai.tensor.colo_tensor.ColoTensor), \
        f'The type of the input tensor should be colotensor' \
        f'Your Input tensor is {type(weight)}'
    # assert weight.device.type == "meta"
    tensor_dist_spec = weight.tensor_spec
    num_partition = tensor_dist_spec.get_process_group_size()
    dim = 0 if tensor_dist_spec.is_shard_1dcol() else -1
    dim_size = weight.size(dim)
    assert dim_size % num_partition == 0, \
        f'The dimension to split ({dim_size}) is not a multiple of world size ({num_partition}), ' \
        f'cannot split tensor evenly'
    spec = TensorSpec(distspec.shard(gpc.get_group(ParallelMode.PARALLEL_1D), [dim], [num_partition]))
    weight.set_tensor_spec(spec)
    return weight


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
                target_module.weight.data = weight_split(target_module.weight.data)
                if target_module.bias is not None:
                    target_module.bias.data = weight_split(target_module.bias.data)

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
                target_module.weight.data = weight_split(target_module.weight.data)

                # insert input sharding node before the sharded linear node
                with mod_graph.inserting_before(node):
                    input_node_list = list(node._input_nodes.keys())
                    assert len(input_node_list) == 1, 'linear forward must have and only have one input tensor.'
                    input_node = input_node_list[0]
                    new_input_node = mod_graph.create_node('call_function', weight_split, args=(input_node))
                    replace_all_uses_except_replaced(input_node, new_input_node)

                # inserting communication node after the sharded linear node
                with mod_graph.inserting_after(node):
                    new_node = mod_graph.create_node('call_function', all_reduce_function, args=(node,))
                    replace_all_uses_except_replaced(node, new_node)
    gm.recompile()
    return gm
