import torch
from torch.fx.node import map_arg
from torch.fx.node import Node
from torch.fx.passes.split_module import split_module

import colossalai
from colossalai.tensor import ColoTensor, TensorSpec, distspec, ProcessGroup, ComputeSpec, ComputePattern


def weight_split(weight: torch.nn.parameter.Parameter, dim: int) -> torch.nn.parameter.Parameter:
    """weight_split 
    split a nn.Parameter

    Args:
        weight (torch.nn.parameter.Parameter): a torch Parameter instance
        dim (int): the dimension to be sharded along with

    Returns:
        _type_: _description_
    """
    #TODO: This func temporarily works with no materialization
    # Append a Tensor spec to target_module.weight.shard
    # Convert to ColoTensor: colo_tensor = ColoTensor.from_torch_tensor(tensor, spec)
    # assert isinstance(weight, torch.nn.parameter.Parameter), \
    #     f'The type of the input tensor should be torch.nn.parameter' \
    #     f'Your Input tensor is {type(weight)}'

    # FIXME() I initialized a PG for this tensor. Only has TP comm group.
    # we only consider the TP-only caes.
    world_size = torch.distributed.get_world_size()
    pg = ProcessGroup(tp_degree=world_size)

    spec = TensorSpec(distspec.shard(pg, [dim], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    # As you has constructed a Spec, why not directly convert the tensor to ColoTensor.
    # setattr(weight, "fx_attr", spec)
    weight.data = ColoTensor(data=weight.data, spec=spec)
    return weight


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
    gm.recompile()
    return gm


#TODO: add elementwise op process pass, then we can try to use column and row mixed strategy.
