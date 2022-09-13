from typing import Dict
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


def autograd_graph_analysis(graph: Graph) -> Dict[str, int]:
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
        meta (Dict): Meta information for the dataflow.
    """

    def _peak_memory(deps: Dict[Node, int]):
        bwd_tmp = 0
        for k, v in deps.items():
            if v > 0:
                bwd_tmp += activation_size(k.meta['out'])
        return bwd_tmp

    # deps is used to track all the memory dependencies of the graph.
    deps = {}
    meta = {
        'fwd_in': 0,
        'fwd_tmp': 0,
        'bwd_tmp': 0,
        'bwd_out': 0,
    }

    for n in graph.nodes:
        n: Node
        if n.meta['save'] and not any(map(is_loss, n.users)):
            # A forward tensor who is marked `save` but is not
            # an input to `loss` should be saved during forward.
            # If the tensor is a placeholder, then it belongs to `fwd_in`.
            # Any `fwd_in` should be kept in memory even this function
            # is checkpointed.
            # Otherwise, the tensor belongs to `fwd_tmp`. If we checkpoint
            # the node, `fwd_tmp` can be freed.
            if is_placeholder(n):
                meta['fwd_in'] += activation_size(n.meta['out'])
            if is_forward(n):
                meta['fwd_tmp'] += activation_size(n.meta['out'])
        elif is_backward(n):
            if len(n.users):
                # liveness analysis is only used in backward
                deps[n] = len(n.users)
                meta['bwd_tmp'] = max(meta['bwd_tmp'], _peak_memory(deps))
                for input_n in n.all_input_nodes:
                    if input_n in deps:
                        deps[input_n] -= 1
            else:
                # basically a backward node without user is a `grad_out` node
                meta['bwd_out'] += activation_size(n.meta['out'])
    return meta
