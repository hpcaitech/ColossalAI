import inspect
from typing import Any, Callable, Dict, List, Optional

import torch
from packaging import version
from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule

from colossalai.fx.passes.adding_split_node_pass import balanced_split_pass, pipe_split
from colossalai.fx.passes.meta_info_prop import TensorMetadata
from colossalai.fx.passes.split_module import Partition


def customized_split_pass_for_gpt2(gm: torch.fx.GraphModule, pp_size: int, partition_list: List[int]):
    """
    This pass is only used to do the gpt2 performance test, it may move into adding_split_node_pass.py, and will be deprecated in future.
    """
    mod_graph = gm.graph
    valid_children_size = 0
    valid_children = []
    for node in mod_graph.nodes:
        if node.op == "call_module":
            valid_children_size += 1
            valid_children.append(node.target)
    if valid_children_size < pp_size:
        # If valid children is not enough to shard, we will use balanced policy instead of uniform policy.
        return balanced_split_pass(gm, pp_size)
    accumulate_layer_amount = 0
    list_of_part = partition_list
    part_index = 0
    for node in mod_graph.nodes:
        if pp_size <= 1:
            break
        if node.op == "call_module":
            if node.target in valid_children:
                accumulate_layer_amount += 1
        if accumulate_layer_amount == list_of_part[part_index]:
            part_index += 1
            pp_size -= 1
            with mod_graph.inserting_after(node):
                split_node = mod_graph.create_node("call_function", pipe_split)

    gm.recompile()
    return gm


def split_with_split_nodes_pass_for_gp2_test(annotated_gm: torch.fx.GraphModule):
    """
    This pass will be used in gpt2 test, only a part of changes may be added into
    split_with_split_nodes_pass, and it will be deprecated in future.
    """
    part_idx = 0

    def eliminate_unused_placeholders(gm):
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                if not len(node.users):
                    gm.graph.erase_node(node)
        gm.recompile()
        return gm

    def refill_outputs_and_placeholders(gm, next_partition_placeholders):
        """
        This method is used to eliminate the outputs in previous partition which is unused in next partition.
        In split module pass, it treats partitions as a DAG, but we need treat them as a single direction linked list in pipeline parallel.
        The difference is if a output from partition 0 is an input argument of partition 3, the DAG will not transfer it
        to partition 1 and partition 2. However, in single direction linked list, we need to do so.
        """
        output_type = None
        output_args = []
        non_output_list = []
        new_placeholder_list = []
        for node in gm.graph.nodes:
            if node.op == "output":
                if isinstance(node.args[0], (tuple, list)):
                    output_type = node.args[0].__class__
                    output_args.extend([n.name for n in node.args[0]])
                else:
                    output_args.append(node.args[0].name)
                rm_list = []
                for name in output_args:
                    if next_partition_placeholders and name not in next_partition_placeholders:
                        rm_list.append(name)
                for name in rm_list:
                    output_args.remove(name)
                gm.graph.erase_node(node)
            else:
                non_output_list.append(node.name)

        for name in next_partition_placeholders:
            if name not in output_args:
                output_args.append(name)

        for name in output_args:
            if name not in non_output_list:
                gm.graph.placeholder(name)

        # convert name to node for output_args
        for index, name in enumerate(output_args):
            for n in gm.graph.nodes:
                if n.name == name:
                    output_args[index] = n
                    continue

        # reorder the output args to make sure
        # output args has same order as next partition placeholder
        reorder_output_args = []
        if next_partition_placeholders:
            for name in next_partition_placeholders:
                for node in output_args:
                    if node.name == name:
                        reorder_output_args.append(node)
                        continue

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                new_placeholder_list.append(node.name)
        if output_type is not None:
            gm.graph.output(output_type(output_args))
        else:
            gm.graph.output(output_args)
        gm.recompile()
        return gm, new_placeholder_list

    def split_callback(n: torch.fx.Node):
        nonlocal part_idx
        if (n.op, n.target) == ("call_function", pipe_split):
            part_idx += 1
        return part_idx

    split_mod = split_module_for_gpt2_test(annotated_gm, None, split_callback)
    split_submodules = []
    for name, submodule in split_mod.named_modules():
        if isinstance(submodule, torch.fx.GraphModule):
            for node in submodule.graph.nodes:
                if (node.op, node.target) == ("call_function", pipe_split):
                    submodule.graph.erase_node(node)
            submodule.recompile()
            split_submodules.append(submodule)

    submodules = list(split_mod.children())
    placeholder_dict = {}
    for submodule in submodules:
        submodule = eliminate_unused_placeholders(submodule)
        placeholder_dict[submodule] = []
    submodules.reverse()
    for index, submodule in enumerate(submodules):
        if index == 0:
            placeholder_list = []
        else:
            placeholder_list = placeholder_dict[submodules[index - 1]]
        submodule, placeholder_dict[submodule] = refill_outputs_and_placeholders(submodule, placeholder_list)
        submodule.recompile()

    split_mod.recompile()

    return split_mod, split_submodules


@compatibility(is_backward_compatible=True)
def split_module_for_gpt2_test(
    m: GraphModule,
    root_m: torch.nn.Module,
    split_callback: Callable[[torch.fx.node.Node], int],
):
    """
    This pass will be used in gpt2 pp performance test, only a part of changes may be added into
    split_module, and it will be deprecated in future.
    """
    partitions: Dict[str, Partition] = {}
    orig_nodes: Dict[str, torch.fx.node.Node] = {}

    def _node_with_all_tensor_element(node_metadata: Any) -> int:
        """
        return whether node contains non-tensor element.
        """
        all_tensor_node = True

        if isinstance(node_metadata, TensorMetadata):
            all_tensor_node = node_metadata.is_tensor and all_tensor_node
        elif isinstance(node_metadata, dict):
            value_list = [v for _, v in node_metadata.items()]
            all_tensor_node += _node_with_all_tensor_element(value_list)
        else:
            for element in node_metadata:
                all_tensor_node += _node_with_all_tensor_element(element)

        return all_tensor_node

    def _move_all_ancestors_into_partition(node, partition_name):
        all_ancestors = set()

        def _gen_all_ancestors_set(node):
            all_ancestors.add(node)
            for n in node.all_input_nodes:
                if n in all_ancestors:
                    continue
                _gen_all_ancestors_set(n)

        _gen_all_ancestors_set(node)
        for n in list(all_ancestors):
            if n.op != "placeholder" and n._fx_partition > partition_name:
                n._fx_partition = partition_name

    def record_cross_partition_use(def_node: torch.fx.node.Node, use_node: Optional[torch.fx.node.Node]):  # noqa: B950
        def_partition_name = getattr(def_node, "_fx_partition", None)
        use_partition_name = getattr(use_node, "_fx_partition", None)
        if def_partition_name != use_partition_name:
            # if 'tensor_meta' in def_node.meta:
            #     if not _node_with_all_tensor_element(def_node.meta['tensor_meta']):
            #         _move_all_ancestors_into_partition(use_node, def_partition_name)
            #         node_process_list.extend(use_node.all_input_nodes)
            #         node_process_list.extend(list(use_node.users))
            #         node_process_list.append(use_node)

            #         return

            if def_partition_name is not None:
                def_partition = partitions[def_partition_name]
                def_partition.outputs.setdefault(def_node.name)
                if use_partition_name is not None:
                    def_partition.partition_dependents.setdefault(use_partition_name)

            if use_partition_name is not None:
                use_partition = partitions[use_partition_name]
                use_partition.inputs.setdefault(def_node.name)
                if def_partition_name is not None:
                    use_partition.partitions_dependent_on.setdefault(def_partition_name)

    node_process_list = list(m.graph.nodes)
    # split nodes into partitions
    while node_process_list:
        node = node_process_list.pop(0)
        orig_nodes[node.name] = node

        if node.op in ["placeholder"]:
            continue
        if node.op == "output":
            # partition_name = str(split_callback(node))
            # def _set_output_args_partition(n, partition_name):
            #     n._fx_partition = partition_name
            # torch.fx.graph.map_arg(node.args[0], lambda n: _set_output_args_partition(n, partition_name))
            torch.fx.graph.map_arg(node.args[0], lambda n: record_cross_partition_use(n, None))
            continue
        partition_name = str(split_callback(node))

        # add node to partitions
        partition = partitions.get(partition_name)
        if partition is None:
            partitions[partition_name] = partition = Partition(partition_name)

        partition.node_names.append(node.name)
        origin_partition_name = getattr(node, "_fx_partition", None)
        if origin_partition_name is None:
            node._fx_partition = partition_name

        torch.fx.graph.map_arg(node.args, lambda def_node: record_cross_partition_use(def_node, node))
        torch.fx.graph.map_arg(node.kwargs, lambda def_node: record_cross_partition_use(def_node, node))  # noqa: B950

    # find partitions with no dependencies
    root_partitions: List[str] = []
    for partition_name, partition in partitions.items():
        if not len(partition.partitions_dependent_on):
            root_partitions.append(partition_name)

    # check partitions for circular dependencies and create topological partition ordering
    sorted_partitions: List[str] = []
    while root_partitions:
        root_partition = root_partitions.pop()
        sorted_partitions.append(root_partition)
        for dependent in partitions[root_partition].partition_dependents:
            partitions[dependent].partitions_dependent_on.pop(root_partition)
            if not partitions[dependent].partitions_dependent_on:
                root_partitions.append(dependent)
    if len(sorted_partitions) != len(partitions):
        raise RuntimeError("cycle exists between partitions!")

    # add placeholders to partitions
    for partition_name in sorted_partitions:
        partition = partitions[partition_name]
        for input in partition.inputs:
            placeholder = partition.graph.placeholder(input)
            placeholder.meta = orig_nodes[input].meta.copy()
            partition.environment[orig_nodes[input]] = placeholder

    # Transform nodes and collect targets for partition's submodule
    for node in m.graph.nodes:
        if hasattr(node, "_fx_partition"):
            partition = partitions[node._fx_partition]

            # swap out old graph nodes in kw/args with references to new nodes in this submodule
            environment = partition.environment
            gathered_args = torch.fx.graph.map_arg(node.args, lambda n: environment[n])
            gathered_kwargs = torch.fx.graph.map_arg(node.kwargs, lambda n: environment[n])

            if node.op not in ["call_module", "get_attr"]:
                target = node.target
            else:
                target_atoms = node.target.split(".")
                target_attr = m
                for atom in target_atoms:
                    if not hasattr(target_attr, atom):
                        raise RuntimeError(f"Operator target {node.target} not found!")
                    target_attr = getattr(target_attr, atom)
                # target = target_atoms[-1]
                target = "_".join(target_atoms)
                partition.targets[target] = target_attr

            assert isinstance(gathered_args, tuple)
            assert isinstance(gathered_kwargs, dict)
            new_node = partition.graph.create_node(
                op=node.op, target=target, args=gathered_args, kwargs=gathered_kwargs, name=node.name
            )
            new_node.meta = node.meta.copy()
            partition.environment[node] = new_node

    # Set up values to construct base module
    base_mod_env: Dict[str, torch.fx.node.Node] = {}
    base_mod_graph: torch.fx.graph.Graph = torch.fx.graph.Graph()
    base_mod_attrs: Dict[str, torch.fx.graph_module.GraphModule] = {}
    for node in m.graph.nodes:
        if node.op == "placeholder":
            if version.parse(torch.__version__) < version.parse("1.11.0"):
                base_mod_env[node.name] = base_mod_graph.placeholder(node.name, type_expr=node.type)
            else:
                default_value = node.args[0] if len(node.args) > 0 else inspect.Signature.empty
                base_mod_env[node.name] = base_mod_graph.placeholder(
                    node.name, type_expr=node.type, default_value=default_value
                )
            base_mod_env[node.name].meta = node.meta.copy()

    # Do some things iterating over the partitions in topological order again:
    # 1) Finish off submodule Graphs by setting corresponding outputs
    # 2) Construct GraphModules for each submodule
    # 3) Construct the base graph by emitting calls to those submodules in
    #    topological order

    for partition_name in sorted_partitions:
        partition = partitions[partition_name]

        # Set correct output values
        output_vals = tuple(partition.environment[orig_nodes[name]] for name in partition.outputs)
        output_vals = output_vals[0] if len(output_vals) == 1 else output_vals  # type: ignore[assignment]
        partition.graph.output(output_vals)

        # Construct GraphModule for this partition
        submod_name = f"submod_{partition_name}"
        base_mod_attrs[submod_name] = torch.fx.graph_module.GraphModule(
            partition.targets, partition.graph
        )  # noqa: B950

        # Emit call in base graph to this submodule
        output_val = base_mod_graph.call_module(submod_name, tuple(base_mod_env[name] for name in partition.inputs))
        if len(partition.outputs) > 1:
            # Unpack multiple return values from submodule
            output_val_proxy = torch.fx.proxy.Proxy(output_val)
            for i, output_name in enumerate(partition.outputs):
                base_mod_env[output_name] = output_val_proxy[i].node  # type: ignore[index]
        else:
            if not partition.outputs:
                continue
            base_mod_env[list(partition.outputs)[0]] = output_val

    for node in m.graph.nodes:
        if node.op == "output":
            base_mod_graph.output(torch.fx.graph.map_arg(node.args[0], lambda n: base_mod_env[n.name]))  # noqa: B950

    return torch.fx.graph_module.GraphModule(base_mod_attrs, base_mod_graph)
