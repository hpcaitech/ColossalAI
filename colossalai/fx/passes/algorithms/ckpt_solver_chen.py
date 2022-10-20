import math
from typing import List, Set, Tuple

import torch
from torch.fx import GraphModule, Node

from colossalai.fx.profiler import calculate_fwd_in, calculate_fwd_tmp

__all__ = ['chen_greedy']
CKPT_OP = ['call_module', 'call_method', 'call_function', 'get_attr']


def _all_potential_ckpt_nodes(gm: GraphModule) -> List:
    """
    In most existing frameworks of activation checkpoint, the forward graph is assumed to be linearized.
    """

    def is_sink():
        """
        If we can free all memories when executing a certain node, it is a sink.
        """
        return not sum((v for k, v in deps.items()))

    deps = {}
    ckpt_nodes = []
    for n in gm.graph.nodes:
        for n_par in n._input_nodes:
            deps[n_par] -= 1    # free memory and dependencies

        # We can only put act_ckpt on these nodes
        if n.op in CKPT_OP and is_sink():
            ckpt_nodes.append(n)
        deps[n] = len(n.users)    # add dependencies for future executions
    return ckpt_nodes


def chen_greedy(gm: GraphModule) -> GraphModule:
    """
    This is the simple implementation of Algorithm 3 in https://arxiv.org/abs/1604.06174.
    Note that this algorithm targets at memory optimization only, using techniques in appendix A.

    Usage:
        model = resnet18()
        input_sample = torch.rand(4, 3, 224, 224)
        gm = symbolic_trace(model)
        MetaInfoProp(gm).run(input_sample)
        gm = chen_greedy(gm)

    Args:
        gm (GraphModule): The module to add checkpoints
    """

    def grid_search(num_grids: int = 6) -> Set:
        """
        Search ckpt strategy with b = 0, then run the allocation algorithm again with b = √xy.
        Grid search over [√2/2 b, √2 b] for ckpt_opt over num_grids as in appendix A.
        """
        _, b_approx = run_chen_greedy(0)
        b_min, b_max = math.floor(b_approx / math.sqrt(2)), math.ceil(b_approx * math.sqrt(2))
        b_opt = math.inf
        for b in range(b_min, b_max, (b_max - b_min) // num_grids):
            ckpt_intv, b_approx = run_chen_greedy(b)
            if b_approx < b_opt:
                b_opt = b_approx
                ckpt_opt = ckpt_intv
        return ckpt_opt

    def run_chen_greedy(b: int = 0) -> Tuple[Set, int]:
        """
        This is the simple implementation of Algorithm 3 in https://arxiv.org/abs/1604.06174.
        """
        ckpt_nodes = _all_potential_ckpt_nodes(gm)
        ckpt_intv = []
        temp = 0
        x = 0
        y = 0
        prev_idx = 2
        for (idx, n) in enumerate(gm.graph.nodes):
            n: Node
            temp += calculate_fwd_in(n) + calculate_fwd_tmp(n)
            y = max(y, temp)
            if temp > b and n in ckpt_nodes:
                x += calculate_fwd_in(n)
                temp = 0
                ckpt_intv.append((prev_idx, idx + 1))
                prev_idx = idx + 1
        return ckpt_intv, math.floor(math.sqrt(x * y))

    gm.graph.lint()    # make sure nodes are in topological order
    ckpt = grid_search(num_grids=6)
    node_list = list(gm.graph.nodes)
    for i, seg in enumerate(ckpt):
        for idx in range(*seg):
            n = node_list[idx]
            if n.op in CKPT_OP:
                setattr(n, 'activation_checkpoint', i)
    gm.recompile()
    return gm
