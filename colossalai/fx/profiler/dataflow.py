from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Dict
from torch.fx import Graph, Node
from .memory import activation_size


class Phase(Enum):
    FORWARD = 0
    LOSS = 1
    BACKWARD = 2
    PLACEHOLDER = 3


@dataclass
class GraphInfo:
    """
    GraphInfo is a dataclass for MetaInfo, which measures
    the execution memory cost and FLOPs with `MetaTensor`.
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
        fwd_flop (int): The forward FLOPs of a certain node
        bwd_flop (int): The backward FLOPs of a certain node.
        fwd_mem_in (int): See the above illustration.
        fwd_mem_tmp (int): See the above illustration.
        bwd_mem_tmp (int): See the above illustration.
        bwd_mem_out (int): See the above illustration.
    """
    fwd_flop: int = 0
    bwd_flop: int = 0
    fwd_mem_in: int = 0
    fwd_mem_tmp: int = 0
    bwd_mem_tmp: int = 0
    bwd_mem_out: int = 0


def is_phase(n: Node, phase: Phase) -> bool:
    assert 'phase' in n.meta, f'Node meta of {n} has no key `phase`!'
    return n.meta['phase'] == phase


def is_saved(n: Node):
    return n.meta.get('saved', False)


def autograd_graph_analysis(graph: Graph) -> GraphInfo:
    """Analyze the autograd node dependencies and find out the memory usage.
    Basically the input graph should have all nodes marked for keyword `phase`.
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
        graph (Graph): The autograd graph with nodes marked for keyword `phase`.

    Returns:
        graph_info (GraphInfo): Meta information for the dataflow.
    """

    def _peak_memory(deps: Dict[Node, int]):
        peak_mem = 0
        for k, v in deps.items():
            if v > 0:
                peak_mem += activation_size(k.meta['out'])
        return peak_mem

    # deps is used to track all the memory dependencies of the graph.
    deps = {}
    graph_info = GraphInfo()

    for n in graph.nodes:
        n: Node
        if is_saved(n) and not any(map(partial(is_phase, phase=Phase.LOSS), n.users)):
            # A forward tensor who is marked `save` but is not
            # an input to `loss` should be saved during forward.
            # If the tensor is a placeholder, then it belongs to `fwd_mem_in`.
            # Any `fwd_mem_in` should be kept in memory even this function
            # is checkpointed.
            # Otherwise, the tensor belongs to `fwd_mem_tmp`. If we checkpoint
            # the node, `fwd_mem_tmp` can be freed.
            if is_phase(n, Phase.PLACEHOLDER):
                graph_info.fwd_mem_in += activation_size(n.meta['out'])
            if is_phase(n, Phase.FORWARD):
                graph_info.fwd_mem_tmp += activation_size(n.meta['out'])
        elif is_phase(n, Phase.BACKWARD):
            if len(n.users):
                # liveness analysis is only used in backward
                deps[n] = len(n.users)
                graph_info.bwd_mem_tmp = max(graph_info.bwd_mem_tmp, _peak_memory(deps))
                for input_n in n.all_input_nodes:
                    if input_n in deps:
                        deps[input_n] -= 1
            else:
                # basically a backward node without user is a `grad_out` node
                graph_info.bwd_mem_out += activation_size(n.meta['out'])
    return graph_info
