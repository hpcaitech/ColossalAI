from dataclasses import dataclass
from enum import Enum
from typing import Dict
from torch.fx import Graph, Node
from .memory import activation_size


class Stage(Enum):
    F = 0
    L = 1
    B = 2
    P = 3


@dataclass
class GraphInfo:
    """
    GraphInfo is a dataclass for the dataflow analysis.
    The dataflow analysis is conducted on a single node of the FX graph.
    ============================================================================
                            -------------------------------
                            |            Node             |
    [fwd_in] are       ---> | [fwd_in]          [bwd_out] |    <----- [bwd_out] is marks the memory for `grad_out`
    placeholders saved for  |     | \__________     |     |
    backward.               |     |            \    |     |
                            | [fwd_tmp] ------> [bwd_tmp] |    <-----
                            |     |  \_________     |     |    [bwd_tmp] marks the peak memory 
                            |    / \           \    |     |    in backward pass.
    [x] is not counted ---> | [x]  [fwd_tmp] -> [bwd_tmp] |    <-----
    in [fwd_tmp] because    |  |       |  \_____    |     |
    it is not saved for     |  |       |        \   |     |
    backward.               -------------------------------
    ============================================================================
    Attributes:
        fwd_in (int): See the above illustration.
        fwd_tmp (int): See the above illustration.
        bwd_tmp (int): See the above illustration.
        bwd_out (int): See the above illustration.
    """
    fwd_in: int = 0
    fwd_tmp: int = 0
    bwd_tmp: int = 0
    bwd_out: int = 0


def is_forward(n: Node):
    assert 'stage' in n.meta, f'Node meta of {n} has no key `stage`!'
    return n.meta['stage'] == Stage.F


def is_loss(n: Node):
    assert 'stage' in n.meta, f'Node meta of {n} has no key `stage`!'
    return n.meta['stage'] == Stage.L


def is_placeholder(n: Node):
    assert 'stage' in n.meta, f'Node meta of {n} has no key `stage`!'
    return n.meta['stage'] == Stage.P


def is_backward(n: Node):
    assert 'stage' in n.meta, f'Node meta of {n} has no key `stage`!'
    return n.meta['stage'] == Stage.B


def is_saved(n: Node):
    return n.meta.get('saved', False)


def autograd_graph_analysis(graph: Graph) -> GraphInfo:
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
        graphinfo (GraphInfo): Meta information for the dataflow.
    """

    def _peak_memory(deps: Dict[Node, int]):
        bwd_tmp = 0
        for k, v in deps.items():
            if v > 0:
                bwd_tmp += activation_size(k.meta['out'])
        return bwd_tmp

    # deps is used to track all the memory dependencies of the graph.
    deps = {}
    graph_info = GraphInfo()

    for n in graph.nodes:
        n: Node
        if is_saved(n) and not any(map(is_loss, n.users)):
            # A forward tensor who is marked `save` but is not
            # an input to `loss` should be saved during forward.
            # If the tensor is a placeholder, then it belongs to `fwd_in`.
            # Any `fwd_in` should be kept in memory even this function
            # is checkpointed.
            # Otherwise, the tensor belongs to `fwd_tmp`. If we checkpoint
            # the node, `fwd_tmp` can be freed.
            if is_placeholder(n):
                graph_info.fwd_in += activation_size(n.meta['out'])
                # print(activation_size(n.meta['out']))
            if is_forward(n):
                graph_info.fwd_tmp += activation_size(n.meta['out'])
                # print(activation_size(n.meta['out']))
        elif is_backward(n):
            if len(n.users):
                # liveness analysis is only used in backward
                deps[n] = len(n.users)
                graph_info.bwd_tmp = max(graph_info.bwd_tmp, _peak_memory(deps))
                for input_n in n.all_input_nodes:
                    if input_n in deps:
                        deps[input_n] -= 1
            else:
                # basically a backward node without user is a `grad_out` node
                graph_info.bwd_out += activation_size(n.meta['out'])
    return graph_info
