import torch
import operator
import colossalai

ELEMENTWISE_MODULE_OP = [torch.nn.Dropout, torch.nn.ReLU, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.AvgPool1d, torch.nn.AvgPool2d]
ELEMENTWISE_FUNC_OP = [torch.add, operator.add, torch.abs, torch.cos, torch.exp, torch.mul, operator.mul, operator.floordiv, operator.truediv, operator.neg, torch.multiply, torch.nn.functional.relu, torch.nn.functional.dropout, torch.nn.functional.conv1d, torch.nn.functional.conv2d, torch.nn.functional.conv3d, torch.nn.functional.avg_pool1d, torch.nn.functional.avg_pool2d, torch.nn.functional.avg_pool3d, torch.nn.functional.max_pool1d, torch.nn.functional.max_pool2d, torch.nn.functional.max_pool3d]

def weight_split(weight: torch.nn.parameter.Parameter, dim: int, col_normal: bool) -> torch.nn.parameter.Parameter:
    """weight_split 
    split a nn.Parameter

    Args:
        weight (torch.nn.parameter.Parameter): a torch Parameter instance
        dim (int): the dimension to be sharded along with
        col_normal(bool): col shard with gather or not
    Returns:
        _type_: _description_
    """
    if col_normal:
        setattr(weight, "fx_attr", (dim, "SHARD", "TP", "col_normal"))
    else:
        setattr(weight, "fx_attr", (dim, "SHARD", "TP", "col_needs_many_outputs"))
    return weight
def column_shard_linear_pass(gm: torch.fx.GraphModule):
    # Split all the linear module with column shard. Currently for testing only.
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
    # Split all the linear module with row shard. Currently for testing only.
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
    # Get the name of element wise module(torch.nn.ReLU)
    # Get the name of all the linear modules and repeated linear modules
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
    # If the linear modules is called multiple times, set the dist spec as col shard
    # If the module is element wise or the function/method is element wise, remains col_shard 
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