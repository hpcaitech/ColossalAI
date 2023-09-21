import numpy as np
import torch
import tqdm

from colossalai.fx.passes.split_module import split_module


def pipe_split():
    pass


def block_split():
    pass


# Construct blocks with the condition that (block_flops / total_flops) >= limit.
def construct_blocks(gm: torch.fx.GraphModule, limit=0.01):
    total_fwd_flop = 0
    total_bwd_flop = 0
    for node in gm.graph.nodes:
        total_fwd_flop += node.fwd_flop
        total_bwd_flop += node.bwd_flop

    total_flop = total_fwd_flop + total_bwd_flop
    per_block_flop = total_flop * limit
    accumulate_fwd_flop = 0
    accumulate_bwd_flop = 0
    block_nodes = []
    for node in gm.graph.nodes:
        if "block_split" in node.name:
            continue
        accumulate_fwd_flop += node.fwd_flop
        accumulate_bwd_flop += node.bwd_flop
        if accumulate_fwd_flop + accumulate_bwd_flop >= per_block_flop:
            with gm.graph.inserting_after(node):
                block_node = gm.graph.create_node("call_function", block_split)
                setattr(block_node, "fwd_flop", accumulate_fwd_flop)
                setattr(block_node, "bwd_flop", accumulate_bwd_flop)
            accumulate_fwd_flop = 0
            accumulate_bwd_flop = 0
            block_nodes.append(block_node)

    return block_nodes


def remove_blocks(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        if (node.op, node.target) == ("call_function", block_split):
            gm.graph.erase_node(node)


def get_compute_costs(node_list):
    num_nodes = len(node_list)
    all_compute_cost = np.full((num_nodes, num_nodes), np.inf, dtype=np.float64)

    for start in tqdm.tqdm(range(num_nodes), desc="start pos", position=0):
        for end in tqdm.tqdm(range(start, num_nodes), desc="end pos", position=1, leave=False):
            selected_flops = [(node_list[i].fwd_flop + node_list[i].bwd_flop) for i in range(start, end + 1)]
            all_compute_cost[start, end] = sum(selected_flops)

    return all_compute_cost


def do_dp_split_gpipe_impl(num_nodes, num_stages, num_microbatches, compute_costs, max_compute_cost):
    """The core implementation of the DP algorithm."""
    # Adapted from Alpa DP Formulation.
    # For f, node ID start from 0
    # f[number of stages,
    #   node id that is currently being considered]

    # record time cost(assess by fwd+bwd flop now)
    f = np.full((num_stages + 1, num_nodes + 1), np.inf, dtype=np.float32)

    # record max stage compute cost among all stages in this partition.
    f_stage_max = np.full((num_stages + 1, num_nodes + 1), 0.0, dtype=np.float32)
    # record start node index for next stage in this partition
    f_argmin = np.full((num_stages + 1, num_nodes + 1), -1, dtype=np.int32)
    f[0, num_nodes] = 0
    for s in tqdm.tqdm(
        range(1, num_stages + 1), desc="stage", position=2, leave=False
    ):  # pylint: disable=too-many-nested-blocks
        for i in tqdm.tqdm(range(num_nodes - 1, -1, -1), desc="start node", position=3, leave=False):
            for k in tqdm.tqdm(range(num_nodes, i, -1), desc="mid node", position=4, leave=False):
                stage_cost = compute_costs[i, k - 1]
                new_cost = f[s - 1, k] + stage_cost
                if stage_cost <= max_compute_cost and new_cost < f[s, i]:
                    f[s, i] = new_cost
                    f_stage_max[s, i] = max(f_stage_max[s - 1, k], stage_cost)
                    f_argmin[s, i] = k

    best_total_cost = f[num_stages, 0]
    if np.isinf(best_total_cost):
        return np.inf, None

    total_cost = f[num_stages, 0] + (num_microbatches - 1) * f_stage_max[num_stages, 0]

    current_s = num_stages
    current_node = 0

    res = []
    while current_s > 0 and current_node < num_nodes:
        next_start_node = f_argmin[current_s, current_node]
        res.append((current_node, next_start_node))
        current_s -= 1
        current_node = next_start_node

    return total_cost, res


def do_dp_split_gpipe(node_list, compute_costs, num_stages: int, num_microbatches: int):
    # Ignore the memory cost profiling in Alpa's design for convenience.
    max_compute_costs = np.sort(np.unique(compute_costs))
    best_cost = np.inf
    best_solution = None
    last_max_compute_cost = 0.0
    gap = 1e6  # temporary magic number, unit: flops

    for max_compute_cost in tqdm.tqdm(max_compute_costs):
        # Pruning to reduce search space.
        if max_compute_cost * num_microbatches >= best_cost:
            break
        if max_compute_cost - last_max_compute_cost < gap:
            continue

        cost, solution = do_dp_split_gpipe_impl(
            len(node_list), num_stages, num_microbatches, compute_costs, max_compute_cost
        )

        if cost < best_cost:
            best_cost = cost
            best_solution = solution
        last_max_compute_cost = max_compute_cost
    return best_cost, best_solution


# Auto DP partition based on Alpa.
# Adapted to Gpipe Scheduler
# split_mode:
#   'node': fx_node
#   'block': many fx_nodes construct a block
def gpipe_dp_split_pass(gm: torch.fx.GraphModule, pp_size: int, num_microbatches: int, mode="block", block_limit=0.01):
    assert mode in ["node", "block"]

    # nodes or blocks will be used in partition.
    node_list = []
    if mode == "node":
        for node in gm.graph.nodes:
            node_list.append(node)
    elif mode == "block":
        node_list = construct_blocks(gm, limit=block_limit)
    else:
        pass

    compute_costs = get_compute_costs(node_list)

    best_cost, best_solution = do_dp_split_gpipe(node_list, compute_costs, pp_size, num_microbatches)

    for _, next_start_node in best_solution:
        if pp_size <= 1:
            break
        node = node_list[next_start_node]
        with gm.graph.inserting_before(node):
            split_node = gm.graph.create_node("call_function", pipe_split)
        pp_size -= 1

    # remove block node if possible
    if mode == "block":
        remove_blocks(gm)

    gm.recompile()
    return gm


def avgcompute_split_pass(gm: torch.fx.GraphModule, pp_size: int):
    """
    In avgcompute_split_pass, we split module by the fwd flops.
    """
    mod_graph = gm.graph
    # To use avgcompute_split_pass, we need run meta_info_prop interpreter first.
    # If nodes don't have meta info, this pass will fall back to normal balanced split pass.
    check_node = list(mod_graph.nodes)[0]
    if "tensor_meta" not in check_node.meta:
        return balanced_split_pass(gm, pp_size)

    total_fwd_flop = 0
    for node in mod_graph.nodes:
        total_fwd_flop += node.fwd_flop

    partition_flop = total_fwd_flop // pp_size
    accumulate_fwd_flop = 0
    for node in mod_graph.nodes:
        if pp_size <= 1:
            break
        if "pipe_split" in node.name:
            continue
        accumulate_fwd_flop += node.fwd_flop
        if accumulate_fwd_flop >= partition_flop:
            total_fwd_flop = total_fwd_flop - accumulate_fwd_flop
            accumulate_fwd_flop = 0
            pp_size -= 1
            partition_flop = total_fwd_flop // pp_size
            with mod_graph.inserting_after(node):
                split_node = mod_graph.create_node("call_function", pipe_split)
    gm.recompile()
    return gm


def avgnode_split_pass(gm: torch.fx.GraphModule, pp_size: int):
    """
    In avgnode_split_pass, simply split graph by node number.
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
            if node.next.op == "output":
                with mod_graph.inserting_before(node):
                    split_node = mod_graph.create_node("call_function", pipe_split)
            else:
                with mod_graph.inserting_after(node):
                    split_node = mod_graph.create_node("call_function", pipe_split)
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
            if node.next.op == "output":
                with mod_graph.inserting_before(node):
                    split_node = mod_graph.create_node("call_function", pipe_split)
            else:
                with mod_graph.inserting_after(node):
                    split_node = mod_graph.create_node("call_function", pipe_split)
    if pp_size > 1:
        node_counter = 0
        for node in mod_graph.nodes:
            if pp_size <= 1:
                break
            if node.op == "placeholder":
                continue
            elif node_counter == 0:
                node_counter += 1
            else:
                pp_size -= 1
                node_counter = 0
                with mod_graph.inserting_before(node):
                    split_node = mod_graph.create_node("call_function", pipe_split)

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
    if "tensor_meta" not in check_node.meta:
        return balanced_split_pass(gm, pp_size)

    total_element_size = 0
    for node in mod_graph.nodes:
        total_element_size += node.node_size

    partition_size = total_element_size // pp_size
    accumulate_node_size = 0
    for node in mod_graph.nodes:
        if pp_size <= 1:
            break
        if "pipe_split" in node.name:
            continue
        accumulate_node_size += node.node_size
        if accumulate_node_size >= partition_size:
            total_element_size = total_element_size - accumulate_node_size
            accumulate_node_size = 0
            pp_size -= 1
            partition_size = total_element_size // pp_size
            with mod_graph.inserting_after(node):
                split_node = mod_graph.create_node("call_function", pipe_split)
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
                split_node = mod_graph.create_node("call_function", pipe_split)
    gm.recompile()
    return gm


def split_with_split_nodes_pass(annotated_gm: torch.fx.GraphModule, merge_output=False):
    # TODO(lyl): use partition IR to assign partition ID to each node.
    # Currently: analyzing graph -> annotate graph by inserting split node -> use split module pass to split graph
    # In future: graph to partitions -> analyzing partition IR -> recombining partitions to get best performance -> assign partition ID to each node
    part_idx = 0

    def split_callback(n: torch.fx.Node):
        nonlocal part_idx
        if (n.op, n.target) == ("call_function", pipe_split):
            part_idx += 1
        return part_idx

    split_mod = split_module(annotated_gm, None, split_callback, merge_output)
    split_submodules = []
    for name, submodule in split_mod.named_modules():
        if isinstance(submodule, torch.fx.GraphModule):
            for node in submodule.graph.nodes:
                if (node.op, node.target) == ("call_function", pipe_split):
                    submodule.graph.erase_node(node)
            submodule.recompile()
            split_submodules.append(submodule)

    return split_mod, split_submodules
