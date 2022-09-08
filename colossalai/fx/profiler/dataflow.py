from typing import Tuple
from torch.fx import Graph, Node
import torch


def is_forward(n: Node):
    assert hasattr(n, 'stage'), f'Node {n} has no attribute `stage`!'
    return getattr(n, 'stage') == 'f'


def is_loss(n: Node):
    assert hasattr(n, 'stage'), f'Node {n} has no attribute `stage`!'
    return getattr(n, 'stage') == 'l'


def is_backward(n: Node):
    assert hasattr(n, 'stage'), f'Node {n} has no attribute `stage`!'
    return getattr(n, 'stage') == 'b'


def autograd_graph_analysis(graph: Graph) -> Tuple[int, int, int, int]:
    """Analyze the autograd node dependencies and find out the memory usage.
    Basically the input graph should have all nodes marked 'f' (forward), 'l' (loss), 'b' (backward).
    Nodes should have attribute `_out_tensors` indicating the output of each node.

    Args:
        graph (Graph): The autograd graph with nodes marked 'f' (forward), 'l' (loss), 'b' (backward)

    Returns:
        fwd_tmp (int): Intermediate memory encountered through forward pass. These tensors are not supposed to be freed unless checkpointed.
        fwd_out (int): The output of the entire forward pass.
        bwd_tmp (int): Intermediate memory (or peak memory) encountered through backward pass. These tensors can be freed as long as it is not required for its users. We will use liveness analysis to detect the peak memory usage.
        bwd_out (int): 
    """
    pass


def _peak_memory_analysis(nodes: Tuple[Node, ...]) -> int:
    """Apply liveness analysis to a list of nodes in topological order and calculate the peak memory.

    Args:
        nodes (Tuple[Node, ...]): A list of nodes in topological order.

    Returns:
        memory_peak (int): Peak memory encountered during the execution.
    """
    pass
