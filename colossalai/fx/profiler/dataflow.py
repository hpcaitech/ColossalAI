from typing import Tuple
from torch.fx import Graph, Node
import torch
from .memory import activation_size


def is_forward(n: Node):
    assert 'stage' in n.meta, f'Node meta of {n} has no key `stage`!'
    return n.meta['stage'] == 'f'


def is_loss(n: Node):
    assert 'stage' in n.meta, f'Node meta of {n} has no key `stage`!'
    return n.meta['stage'] == 'l'


def is_backward(n: Node):
    assert 'stage' in n.meta, f'Node meta of {n} has no key `stage`!'
    return n.meta['stage'] == 'b'


def autograd_graph_analysis(graph: Graph) -> Tuple[int, int, int, int]:
    """Analyze the autograd node dependencies and find out the memory usage.
    Basically the input graph should have all nodes marked 'f' (forward), 'l' (loss), 'b' (backward) for keyword `stage`.
    Nodes should have attribute `out_tensors` indicating the output of each node.
    ============================================================================
                    p           o    <---- We need to keep track of grad out
                    |\________  | 
                    ↓         ↘|
                    f --------> b
                    |\ \_____   ↑
                    | \      ↘ /
                    f  f ----> b      <---- Not every forward result needs to be saved for backward
                    |   \____  ↑
                    ↘       ↘| 
                      f ----> b       <---- Backward can be freed as soon as it is required no more.
                         ↘ ↗
                           l
    =============================================================================                     
    Args:
        graph (Graph): The autograd graph with nodes marked 'f' (forward), 'l' (loss), 'b' (backward) for keyword `stage`.

    Returns:
        fwd_tmp (int): Intermediate memory encountered through forward pass. These tensors are not supposed to be freed unless checkpointed.
        fwd_out (int): The output of the entire forward pass.
        bwd_tmp (int): Intermediate memory (or peak memory) encountered through backward pass. These tensors can be freed as long as it is not required for its users. We will use liveness analysis to detect the peak memory usage.
        bwd_out (int): The output of the entire backward pass.
    """
    # deps is used to track all the memory dependencies of the graph.
    deps = {}

    fwd_tmp = 0
    fwd_out = 0
    bwd_tmp = 0
    bwd_out = 0

    for n in graph.nodes:
        n: Node
        if is_forward(n):
            if any(map(is_backward, n.users)):
                fwd_tmp += activation_size(n.meta['out_tensors'])
            if any(map(is_loss, n.users)):
                fwd_out += activation_size(n.meta['out_tensors'])
        elif is_backward(n):
            if not len(n.users):
                bwd_out += activation_size(n.meta['out_tensors'])


def _peak_memory_analysis(nodes: Tuple[Node, ...]) -> int:
    """Apply liveness analysis to a list of nodes in topological order and calculate the peak memory.

    Args:
        nodes (Tuple[Node, ...]): A list of nodes in topological order.

    Returns:
        memory_peak (int): Peak memory encountered during the execution.
    """
    pass
