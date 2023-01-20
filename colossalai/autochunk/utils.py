from typing import Any, Callable, Dict, Iterable, List, Tuple

from torch.fx.node import Node


def flat_list(inputs: Any) -> List:
    """
    flat a list by recursion
    """
    if not (isinstance(inputs, list) or isinstance(inputs, set) or isinstance(inputs, tuple)):
        return [inputs]
    res = []
    for i in inputs:
        if isinstance(i, list) or isinstance(i, set) or isinstance(i, tuple):
            res.extend(flat_list(i))
        else:
            res.append(i)
    return res


def find_first_tensor_arg(node: Node) -> Node:
    """
    Find the first input tensor arg for a node
    """
    for arg in node.args:
        if type(arg) == type(node):
            return arg
    raise RuntimeError()


def is_non_compute_node(node: Node) -> bool:
    if any(i in node.op for i in ["placeholder", "get_attr", "output"]) or any(i in node.name for i in ["getattr"]):
        return True
    if "getitem" in node.name:
        node_args = flat_list(node.args[1:])
        for node_arg in node_args:
            if any(i == str(node_arg) for i in ["None", "Ellipsis"]):
                return False
            if "slice" in str(node_arg):
                return False
        return True
    return False


def get_node_shape(node: Node) -> List:
    if hasattr(node.meta["tensor_meta"], "shape"):
        return node.meta["tensor_meta"].shape
    return None


def is_non_memory_node(node: Node) -> bool:
    if "getitem" in node.name:
        return True
    if "output" in node.op:
        return True
    return is_non_compute_node(node)


def is_non_compute_node_except_placeholder(node):
    if "placeholder" in node.op:
        return False
    return is_non_compute_node(node)


def is_non_compute_node_except_placeholder_output(node):
    if "output" in node.op:
        return False
    return is_non_compute_node_except_placeholder(node)


def find_idx_by_name(name, nodes_list):
    for idx, node in enumerate(nodes_list):
        if node.name == name:
            return idx
    raise RuntimeError("name %s not found in node list" % name)


def delete_free_var_from_last_use(user_to_last_uses):
    for key, value in user_to_last_uses.items():
        for n in value:
            if n.op == "placeholder":
                user_to_last_uses[key].remove(n)


def find_chunk_all_input_nodes(nodes: List[Node]):
    """
    Find non-compute input and output node names.
    input nodes are nodes used in the list
    output nodes are nodes will use nodes in the list
    """
    input_nodes = []
    for node in nodes:
        for input_node in node._input_nodes.keys():
            if input_node not in nodes and input_node not in input_nodes:
                input_nodes.append(input_node)
    return input_nodes


def find_chunk_compute_input_and_output_nodes(nodes: List[Node]):
    """
    Find non-compute input and output node names.
    input nodes are nodes used in the list
    output nodes are nodes will use nodes in the list
    """
    input_nodes = []
    output_nodes = []

    # if a node has an input node which is not in the node list
    # we treat that input node as the input of the checkpoint function
    for node in nodes:
        for input_node in node._input_nodes.keys():
            if (input_node not in nodes and input_node not in input_nodes
                    and not is_non_compute_node_except_placeholder(input_node)):
                input_nodes.append(input_node)

    # if a node has a user node which is not in the node list
    # we treat that user node as the node receiving the current node output
    for node in nodes:
        for output_node in node.users.keys():
            if (output_node not in nodes and node not in output_nodes
                    and not is_non_compute_node_except_placeholder_output(output_node)):
                output_nodes.append(node)

    return input_nodes, output_nodes
