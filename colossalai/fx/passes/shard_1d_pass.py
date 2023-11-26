import operator

import torch
import torch.nn as nn

from colossalai.legacy.tensor import ProcessGroup
from colossalai.legacy.tensor.compute_spec import ComputePattern, ComputeSpec
from colossalai.legacy.tensor.distspec import ShardSpec

ELEMENTWISE_MODULE_OP = [torch.nn.Dropout, torch.nn.ReLU]
ELEMENTWISE_FUNC_OP = [
    torch.add,
    operator.add,
    torch.abs,
    torch.cos,
    torch.exp,
    torch.mul,
    operator.mul,
    operator.floordiv,
    operator.truediv,
    operator.neg,
    torch.multiply,
    torch.nn.functional.relu,
    torch.nn.functional.dropout,
]


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


def transformer_mlp_pass(graph_module: torch.fx.GraphModule, process_group: ProcessGroup):
    """
    This IR pass checks for transformer MLP like structure and annotate column and row sharding to the linear layers.
    """
    # TODO: Needs to handle special cases, like x = linear(x) + linear(x)
    graph = graph_module.graph
    world_size = process_group.world_size()

    def _traverse_and_annotate(node, start_tracking, annotation_record, world_size):
        # traverse the graph to look for consecutive linear layers
        is_linear_module = False

        if node.op == "call_module":
            # look for the linear layer
            module = node.graph.owning_module.get_submodule(node.target)
            if isinstance(module, nn.Linear):
                is_linear_module = True
                if start_tracking:
                    # when start_tracking = True
                    # it means the first linear has been found and the current module
                    # is the second linear
                    # set the current linear module to be row-sharded
                    annotation_record["row"] = module

                    for shard_type, module in annotation_record.items():
                        # add row sharding spec
                        if shard_type == "row":
                            dist_spec = ShardSpec(dims=[-1], num_partitions=[world_size])
                            comp_spec = ComputeSpec(ComputePattern.TP1D)
                            setattr(module.weight, "pg", process_group)
                            setattr(module.weight, "dist_spec", dist_spec)
                            setattr(module.weight, "comp_spec", comp_spec)
                        elif shard_type == "col":
                            weight_dist_spec = ShardSpec(dims=[0], num_partitions=[world_size])
                            weight_comp_spec = ComputeSpec(ComputePattern.TP1D)
                            weight_comp_spec.output_replicate = False
                            setattr(module.weight, "pg", process_group)
                            setattr(module.weight, "dist_spec", weight_dist_spec)
                            setattr(module.weight, "comp_spec", weight_comp_spec)

                            if module.bias is not None:
                                bias_dist_spec = ShardSpec(dims=[0], num_partitions=[world_size])
                                bias_comp_spec = ComputeSpec(ComputePattern.TP1D)
                                bias_comp_spec.output_replicate = False
                                setattr(module.bias, "pg", process_group)
                                setattr(module.bias, "dist_spec", bias_dist_spec)
                                setattr(module.bias, "comp_spec", bias_comp_spec)
                    start_tracking = False
                    annotation_record.clear()
                else:
                    # when start tracking = False
                    # it means the current layer is the first linear
                    # set the linear layer to be col-sharded
                    start_tracking = True
                    annotation_record["col"] = module

        if start_tracking and not is_linear_module:
            # check against the white list
            # if non-element wise op is found, we reset the tracking
            if node.op == "call_module":
                module = node.graph.owning_module.get_submodule(node.target)
                if module.__class__ not in ELEMENTWISE_MODULE_OP:
                    start_tracking = False
            elif node.op == "call_function" or node.op == "call_method":
                if node.target not in ELEMENTWISE_FUNC_OP:
                    start_tracking = False
            elif len(node.users.keys()) > 1:
                start_tracking = False

            if not start_tracking:
                annotation_record.clear()

        # stop tracking for consecutive linear when branch is found
        # e.g.
        # out1 = self.linear1(x)
        # out2 = self.linear2(x)
        # return out1+out2
        next_nodes = list(node.users.keys())
        if len(next_nodes) > 1:
            start_tracking = False
            annotation_record.clear()

        # traverse
        for node in next_nodes:
            _traverse_and_annotate(node, start_tracking, annotation_record, world_size)

    placeholder_node = list(graph.nodes)[0]
    annotate_record = {}
    _traverse_and_annotate(placeholder_node, False, annotate_record, world_size)

    return graph_module
