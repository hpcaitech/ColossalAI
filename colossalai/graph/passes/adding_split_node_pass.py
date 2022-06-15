import torch

from torch.fx import symbolic_trace
from torch.fx.node import Node
from torch.fx.passes.split_module import split_module


def pipe_split():
    pass


def balanced_split_pass(gm: torch.fx.GraphModule, pp_size: int):
    mod_graph = gm.graph
    total_param_amount = 0
    for param in mod_graph.owning_module.parameters():
        total_param_amount += param.numel()
    params_per_partition = total_param_amount // pp_size
    accumulate_param_amount = 0
    for node in mod_graph.nodes:
        if pp_size <= 1:
            break
        if node.op == "call_module":
            target_module = node.graph.owning_module.get_submodule(node.target)
            for param in target_module.parameters():
                accumulate_param_amount += param.numel()
        if accumulate_param_amount >= params_per_partition:
            accumulate_param_amount = 0
            pp_size -= 1
            with mod_graph.inserting_after(node):
                split_node = mod_graph.create_node('call_function', pipe_split)
    gm.recompile()
    return gm


def split_with_split_nodes_pass(annotated_gm: torch.fx.GraphModule):
    part_idx = 0

    def split_callback(n: torch.fx.Node):
        nonlocal part_idx
        if (n.op, n.target) == ('call_function', pipe_split):
            part_idx += 1
        return part_idx

    split_mod = split_module(annotated_gm, None, split_callback)
    split_submodules = []
    for name, submodule in split_mod.named_modules():
        if isinstance(submodule, torch.fx.GraphModule):
            for node in submodule.graph.nodes:
                if (node.op, node.target) == ('call_function', pipe_split):
                    submodule.graph.erase_node(node)
            submodule.recompile()
            split_submodules.append(submodule)

    return split_mod, split_submodules
