import torch
from torch.fx import symbolic_trace
from torch.fx.node import Node

from colossalai.fx.passes.split_module import split_module


def pipe_split():
    pass


def avgcompute_split_pass(gm: torch.fx.GraphModule, pp_size: int):
    """
    In avgcompute_split_pass, we split module by the fwd flops.
    """
    mod_graph = gm.graph
    # To use avgcompute_split_pass, we need run meta_info_prop interpreter first.
    # If nodes don't have meta info, this pass will fall back to normal balanced split pass.
    check_node = list(mod_graph.nodes)[0]
    if 'tensor_meta' not in check_node.meta:
        return balanced_split_pass(gm, pp_size)

    total_fwd_flop = 0
    for node in mod_graph.nodes:
        total_fwd_flop += node.fwd_flop

    partition_flop = total_fwd_flop // pp_size
    accumulate_fwd_flop = 0
    for node in mod_graph.nodes:
        if pp_size <= 1:
            break
        if 'pipe_split' in node.name:
            continue
        accumulate_fwd_flop += node.fwd_flop
        if accumulate_fwd_flop >= partition_flop:
            total_fwd_flop = total_fwd_flop - accumulate_fwd_flop
            accumulate_fwd_flop = 0
            pp_size -= 1
            partition_flop = total_fwd_flop // pp_size
            with mod_graph.inserting_after(node):
                split_node = mod_graph.create_node('call_function', pipe_split)
    gm.recompile()
    return gm


def avgnode_split_pass(gm: torch.fx.GraphModule, pp_size: int):
    """
    In avgnode_split_pass, simpliy split graph by node number.
    """
    mod_graph = gm.graph
    avg_num_node = len(mod_graph.nodes) // pp_size
    accumulate_num_node = 0
    for node in mod_graph.nodes:
        if pp_size <= 1:
            break
        accumulate_num_node += 1
        if accumulate_num_node >= avg_num_node:
            accumulate_num_node = 0
            pp_size -= 1
            if node.next.op == 'output':
                with mod_graph.inserting_before(node):
                    split_node = mod_graph.create_node('call_function', pipe_split)
            else:
                with mod_graph.inserting_after(node):
                    split_node = mod_graph.create_node('call_function', pipe_split)
    gm.recompile()
    return gm


def balanced_split_pass(gm: torch.fx.GraphModule, pp_size: int):
    """
    In balanced_split_pass, we split module by the size of parameters(weights+bias).
    """
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
            # If the next node is output node, we will insert split annotation before
            # node to make sure there is at least one node in last partition.
            if node.next.op == 'output':
                with mod_graph.inserting_before(node):
                    split_node = mod_graph.create_node('call_function', pipe_split)
            else:
                with mod_graph.inserting_after(node):
                    split_node = mod_graph.create_node('call_function', pipe_split)
    if pp_size > 1:
        node_counter = 0
        for node in mod_graph.nodes:
            if pp_size <= 1:
                break
            if node.op == 'placeholder':
                continue
            elif node_counter == 0:
                node_counter += 1
            else:
                pp_size -= 1
                node_counter = 0
                with mod_graph.inserting_before(node):
                    split_node = mod_graph.create_node('call_function', pipe_split)

    gm.recompile()
    return gm


def balanced_split_pass_v2(gm: torch.fx.GraphModule, pp_size: int):
    """
    In balanced_split_pass_v12, we split module by the size of nodes(weights+bias+outputs).
    """
    mod_graph = gm.graph
    # To use balanced_split_pass_v2, we need run meta_info_prop interpreter first.
    # If nodes don't have meta info, this pass will fall back to normal balanced split pass.
    check_node = list(mod_graph.nodes)[0]
    if 'tensor_meta' not in check_node.meta:
        return balanced_split_pass(gm, pp_size)

    total_element_size = 0
    for node in mod_graph.nodes:
        total_element_size += node.node_size

    partition_size = total_element_size // pp_size
    accumulate_node_size = 0
    for node in mod_graph.nodes:
        if pp_size <= 1:
            break
        if 'pipe_split' in node.name:
            continue
        accumulate_node_size += node.node_size
        if accumulate_node_size >= partition_size:
            total_element_size = total_element_size - accumulate_node_size
            accumulate_node_size = 0
            pp_size -= 1
            partition_size = total_element_size // pp_size
            with mod_graph.inserting_after(node):
                split_node = mod_graph.create_node('call_function', pipe_split)
    gm.recompile()
    return gm


def uniform_split_pass(gm: torch.fx.GraphModule, pp_size: int):
    mod_graph = gm.graph
    valid_children_size = 0
    valid_children = []
    for module in mod_graph.owning_module.children():
        valid_children_size += 1
        valid_children.append(module)

    if valid_children_size < pp_size:
        # If valid children is not enough to shard, we will use balanced policy instead of uniform policy.
        return balanced_split_pass(gm, pp_size)
    layers_per_partition = valid_children_size // pp_size
    accumulate_layer_amount = 0
    for node in mod_graph.nodes:
        if pp_size <= 1:
            break
        if node.op == "call_module":
            target_module = node.graph.owning_module.get_submodule(node.target)
            if target_module in valid_children:
                accumulate_layer_amount += 1
        if accumulate_layer_amount == layers_per_partition:
            accumulate_layer_amount = 0
            pp_size -= 1
            with mod_graph.inserting_after(node):
                split_node = mod_graph.create_node('call_function', pipe_split)
    gm.recompile()
    return gm


def split_with_split_nodes_pass(annotated_gm: torch.fx.GraphModule, merge_output=False):
    # TODO(lyl): use partition IR to assign partition ID to each node.
    # Currently: analyzing graph -> annotate graph by inserting split node -> use split module pass to split graph
    # In future: graph to partitions -> analyzing partition IR -> recombining partitions to get best performance -> assign partition ID to each node
    part_idx = 0

    def split_callback(n: torch.fx.Node):
        nonlocal part_idx
        if (n.op, n.target) == ('call_function', pipe_split):
            part_idx += 1
        return part_idx

    split_mod = split_module(annotated_gm, None, split_callback, merge_output)
    split_submodules = []
    for name, submodule in split_mod.named_modules():
        if isinstance(submodule, torch.fx.GraphModule):
            for node in submodule.graph.nodes:
                if (node.op, node.target) == ('call_function', pipe_split):
                    submodule.graph.erase_node(node)
            submodule.recompile()
            split_submodules.append(submodule)

    return split_mod, split_submodules
