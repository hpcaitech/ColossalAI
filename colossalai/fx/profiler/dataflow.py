from typing import Dict, Tuple
from torch.fx import Graph, Node
from .memory import INPLACE_ATEN, activation_size


def is_forward(n: Node):
    assert 'stage' in n.meta, f'Node meta of {n} has no key `stage`!'
    return n.meta['stage'] == 'f'


def is_loss(n: Node):
    assert 'stage' in n.meta, f'Node meta of {n} has no key `stage`!'
    return n.meta['stage'] == 'l'


def is_placeholder(n: Node):
    assert 'stage' in n.meta, f'Node meta of {n} has no key `stage`!'
    return n.meta['stage'] == 'p'


def is_backward(n: Node):
    assert 'stage' in n.meta, f'Node meta of {n} has no key `stage`!'
    return n.meta['stage'] == 'b'


def autograd_graph_analysis(graph: Graph) -> Tuple[int, int, int, int]:
    """Analyze the autograd node dependencies and find out the memory usage.
    Basically the input graph should have all nodes marked 'f' (forward), 'l' (loss), 'b' (backward) for keyword `stage`.
    Nodes should have attribute `out` indicating the output of each node.
    ============================================================================
    Placeholder ---->   p           o     <---- We need to keep track of grad out
                        |\________  | 
                        ↓         ↘|
                        f --------> b
                        |\ \_____   ↑
                        | \      ↘ /
                        f  f ----> b      <---- Not every forward result needs to be saved for backward
                        |   \____  ↑
                         ↘      ↘| 
                           f ----> b      <---- Backward can be freed as soon as it is required no more.
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

    def _peak_memory(deps: Dict[Node, int]):
        bwd_tmp = 0
        for k, v in deps.items():
            if v > 0:
                bwd_tmp += activation_size(k.meta['out'])
        return bwd_tmp

    # deps is used to track all the memory dependencies of the graph.
    deps = {}

    fwd_in = 0
    fwd_tmp = 0
    bwd_tmp = 0
    bwd_out = 0

    for n in graph.nodes:
        n: Node
        if is_placeholder(n):
            # a placeholder node who has any backward node users will have to be kept in memory until released
            if any(map(is_backward, n.users)) and not any(map(is_loss, n.users)):
                # but if its users are all inplace methods in forward pass, it should not have activations
                fwd_in += activation_size(n.meta['out'])
        if is_forward(n):
            # a forward node who has any backward node users will have to be kept in memory until released
            if any(map(is_backward, n.users)) and not any(map(is_loss, n.users)):
                # but if its users are all inplace methods in forward pass, it should not have activations
                fwd_tmp += activation_size(n.meta['out'])
        elif is_backward(n):
            if len(n.users):
                # liveness analysis is only used in backward
                deps[n] = len(n.users)
                bwd_tmp = max(bwd_tmp, _peak_memory(deps))
                for input_n in n.all_input_nodes:
                    if input_n in deps:
                        deps[input_n] -= 1
            else:
                # basically a backward node without user is a `grad_out` node
                bwd_out += activation_size(n.meta['out'])
    return fwd_in, fwd_tmp, bwd_tmp, bwd_out
