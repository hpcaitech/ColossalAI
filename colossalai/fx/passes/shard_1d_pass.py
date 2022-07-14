import torch
import operator
from torch.fx.node import map_arg
from torch.fx.node import Node
from torch.fx.passes.split_module import split_module

import colossalai
# from colossalai.tensor import ColoTensor, TensorSpec, distspec, ProcessGroup, ComputeSpec, ComputePattern

ELEMENTWISE_MODULE_OP = [torch.nn.Dropout, torch.nn.ReLU, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.AvgPool1d, torch.nn.AvgPool2d]
ELEMENTWISE_FUNC_OP = [torch.add, operator.add, torch.abs, torch.cos, torch.exp, torch.mul, operator.mul, operator.floordiv, operator.truediv, operator.neg, torch.multiply, torch.nn.functional.relu, torch.nn.functional.dropout, torch.nn.functional.conv1d, torch.nn.functional.conv2d, torch.nn.functional.conv3d, torch.nn.functional.avg_pool1d, torch.nn.functional.avg_pool2d, torch.nn.functional.avg_pool3d, torch.nn.functional.max_pool1d, torch.nn.functional.max_pool2d, torch.nn.functional.max_pool3d]

def weight_split(weight: torch.nn.parameter.Parameter, dim: int, col_normal: bool) -> torch.nn.parameter.Parameter:
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
    # world_size = torch.distributed.get_world_size()
    # pg = ProcessGroup(tp_degree=world_size)

    # spec = TensorSpec(distspec.shard(pg, [dim], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    # # As you has constructed a Spec, why not directly convert the tensor to ColoTensor.
    if col_normal:
        setattr(weight, "fx_attr", (dim, "SHARD", "TP", "col_normal"))
    else:
        setattr(weight, "fx_attr", (dim, "SHARD", "TP", "col_needs_many_outputs"))
    # weight.data = ColoTensor(data=weight.data, spec=spec)
    return weight
def column_shard_linear_pass(gm: torch.fx.GraphModule):
    mod_graph = gm.graph
    for node in mod_graph.nodes:
        if node.op == "call_module":
            target_module = node.graph.owning_module.get_submodule(node.target)
            if isinstance(target_module, torch.nn.Linear):
                target_module.weight = weight_split(target_module.weight, dim=0, col_normal=False)
                if target_module.bias is not None:
                    target_module.bias.data = weight_split(target_module.bias.data, dim=0, col_normal=False)

    gm.recompile()
    return gm


def row_shard_linear_pass(gm: torch.fx.GraphModule):
    mod_graph = gm.graph
    for node in mod_graph.nodes:
        if node.op == "call_module":
            target_module = node.graph.owning_module.get_submodule(node.target)
            if isinstance(target_module, torch.nn.Linear):
                target_module.weight = weight_split(target_module.weight, dim=-1, col_normal=False)

    gm.recompile()
    return gm

def transform_mlp_pass(gm: torch.fx.GraphModule):
    #TODO: Needs to handle special cases, like x = linear(x) + linear(x)
    mod_graph = gm.graph
    col_shard = True
    element_op = []
    all_linear_name = []
    linear_name = []
    for name, func in gm.named_children():
        if not isinstance(func, torch.nn.Linear):
            for i in ELEMENTWISE_MODULE_OP:
                if isinstance(func, i):
                    element_op.append(name)
                    break
        else:
            if name in all_linear_name:
                if name in linear_name:
                    linear_name.remove(name)
            else:
                all_linear_name.append(name)
                linear_name.append(name)
    for node in mod_graph.nodes:
        if node.target in linear_name:
            target_module = node.graph.owning_module.get_submodule(node.target)
            dim = 0 if col_shard else -1
            target_module.weight = weight_split(target_module.weight, dim=dim, col_normal=False)
            col_shard = not col_shard
        elif node.target in all_linear_name:
            target_module = node.graph.owning_module.get_submodule(node.target)
            dim = 0 if col_shard else -1
            target_module.weight = weight_split(target_module.weight, dim=dim, col_normal=True)
            col_shard = not col_shard
        else:
            if node.target not in element_op and all(node.target != i for i in ELEMENTWISE_FUNC_OP):
                col_shard = True
    gm.recompile()
    return gm

            

#TODO: add elementwise op process pass, then we can try to use column and row mixed strategy.