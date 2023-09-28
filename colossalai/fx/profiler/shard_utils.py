import torch
from torch.fx import Node

from .._compatibility import compatibility, is_compatible_with_meta
from .memory_utils import activation_size

if is_compatible_with_meta():
    from .constants import OUTPUT_SAVED_MOD, OUTPUT_SAVED_OPS

__all__ = ["calculate_fwd_in", "calculate_fwd_tmp", "calculate_fwd_out"]


@compatibility(is_backward_compatible=False)
def calculate_fwd_in(n: Node) -> int:
    """A helper function to calculate `fwd_in` (with sharding spec)

    Args:
        n (Node): a node from the graph

    Returns:
        fwd_in (int): the result of `fwd_in`
    """
    # TODO(super-dainiu): should divide the memory by sharding spec
    return activation_size(n.meta["fwd_in"])


@compatibility(is_backward_compatible=False)
def calculate_fwd_tmp(n: Node) -> int:
    """A helper function to calculate `fwd_tmp` (with sharding spec)
    Currently, `torch.nn.ReLU` behaves weirdly, so we have to patch it for accuracy.

    Args:
        n (Node): a node from the graph

    Returns:
        fwd_tmp (int): the result of `fwd_tmp`
    """

    # TODO(super-dainiu): should divide the memory by sharding spec
    def is_relu_like_node(n: Node) -> bool:
        """Check if a node is a ReLU-like node.
        ReLU-like nodes have the following properties:
        - They are either `call_function` or `call_module`
        - Their output tensors are directly saved for backward
        - Their input tensors are not saved for backward

        An example is `torch.nn.functional.softmax` which has (forward + backward):
        def forward(self, input_2):
            _softmax_default = torch.ops.aten._softmax.default(input_2, None, None);  input_2 = None
            zeros_like_default = torch.ops.aten.zeros_like.default(_softmax_default, dtype = None, layout = None, device = None, pin_memory = None)
            detach_default = torch.ops.aten.detach.default(_softmax_default);  _softmax_default = None
            _softmax_backward_data_default = torch.ops.aten._softmax_backward_data.default(zeros_like_default, detach_default, None, None);  zeros_like_default = detach_default = None
            detach_default_1 = torch.ops.aten.detach.default(_softmax_backward_data_default);  _softmax_backward_data_default = None
            detach_default_2 = torch.ops.aten.detach.default(detach_default_1);  detach_default_1 = None

        Args:
            n (Node): A node from the graph

        Returns:
            bool: Whether the node is a ReLU-like node
        """
        if n.op == "call_function":
            return n.target in OUTPUT_SAVED_OPS
        elif n.op == "call_module":
            return type(n.graph.owning_module.get_submodule(n.target)) in OUTPUT_SAVED_MOD
        return False

    if not is_relu_like_node(n):
        return activation_size(n.meta["fwd_tmp"])
    return 0


@compatibility(is_backward_compatible=False)
def calculate_fwd_out(n: Node) -> int:
    """A helper function to calculate `fwd_out` (with sharding spec)

    Args:
        n (Node): a node from the graph

    Returns:
        fwd_out (int): the result of `fwd_out`
    """

    # TODO(super-dainiu): should divide the memory by sharding spec
    def intersect(a, b):
        return {k: a[k] for k in a if k in b}

    fwd_in = dict()
    for u in n.users:
        fwd_in.update({x.data_ptr(): x for x in u.meta["fwd_in"] if isinstance(x, torch.Tensor)})
    fwd_out = {x.data_ptr(): x for x in n.meta["fwd_out"] if isinstance(x, torch.Tensor)}
    return activation_size(intersect(fwd_in, fwd_out))


def calculate_fwd_time(n: Node) -> float:
    """A helper function to calculate `fwd_time` (with sharding spec)
    Args:
        n (Node): a node from the graph
    Returns:
        fwd_time (float): the result of `fwd_time`
    """
    # TODO(super-dainiu): should divide the time by the number of GPUs as well as TFLOPs
    return n.meta["fwd_time"]


def calculate_bwd_time(n: Node) -> float:
    """A helper function to calculate `bwd_time` (with sharding spec)
    Args:
        n (Node): a node from the graph
    Returns:
        bwd_time (float): the result of `bwd_time`
    """
    # TODO(super-dainiu): should divide the time by the number of GPUs as well as TFLOPs
    return n.meta["bwd_time"]
