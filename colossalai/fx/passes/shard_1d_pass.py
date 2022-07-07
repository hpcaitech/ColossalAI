import torch
from torch.fx.node import map_arg
from colossalai.tensor import ColoTensorSpec, distspec, ProcessGroup, ComputeSpec, ComputePattern


def weight_split(weight: torch.Tensor, dim: int) -> torch.nn.parameter.Parameter:
    """weight_split 
    split a nn.Parameter

    Args:
        weight (torch.nn.parameter.Parameter): a torch Parameter instance
        dim (int): the dimension to be sharded along with

    Returns:
        _type_: _description_
    """
    # Append a Tensor spec to target_module.weight.shard
    # Convert to ColoTensor: colo_tensor = ColoTensor.from_torch_tensor(tensor, spec)
    assert isinstance(weight, torch.Tensor), \
        f'The type of the input tensor should be torch.nn.parameter' \
        f'Your Input tensor is {type(weight)}'

    # FIXME() I initialized a PG for this tensor. Only has TP comm group.
    # we only consider the TP-only caes.
    world_size = torch.distributed.get_world_size()
    pg = ProcessGroup(tp_degree=world_size)

    spec = ColoTensorSpec(pg, distspec.shard([dim], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    # As you has constructed a Spec, why not directly convert the tensor to ColoTensor.
    setattr(weight, "fx_attr", spec)
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
                target_module.weight = weight_split(target_module.weight, dim=0)
                if target_module.bias is not None:
                    target_module.bias.data = weight_split(target_module.bias.data, dim=0)

    gm.recompile()
    return gm


def row_shard_linear_pass(gm: torch.fx.GraphModule):
    mod_graph = gm.graph
    for node in mod_graph.nodes:
        if node.op == "call_module":
            target_module = node.graph.owning_module.get_submodule(node.target)
            if isinstance(target_module, torch.nn.Linear):
                target_module.weight = weight_split(target_module.weight, dim=-1)

                # insert input sharding node before the sharded linear node
                with mod_graph.inserting_before(node):
                    input_node_list = list(node._input_nodes.keys())
                    assert len(input_node_list) == 1, 'linear forward must have and only have one input tensor.'
                    input_node = input_node_list[0]
                    new_input_node = mod_graph.create_node('call_function', weight_split, args=(input_node, -1))
                    replace_all_uses_except_replaced(input_node, new_input_node)
    gm.recompile()
    return gm


#TODO: add elementwise op process pass, then we can try to use column and row mixed strategy.
