from typing import List, Set, Tuple
import torch
from torch.fx import GraphModule
import math

__all__ = ['chen_greedy', 'chen_sqrtn']


def _all_potential_ckpt_nodes(gm: GraphModule) -> List:
    ckpt_nodes = []
    for n in gm.graph.nodes:
        if n.op == 'call_module':
            ckpt_nodes.append(n)
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
            temp += getattr(n, 'activation_size')
            y = max(y, temp)
            if temp > b and n in ckpt_nodes:
                x += getattr(n, 'activation_size')
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
            if n.op in ['call_module', 'call_method', 'call_function']:
                setattr(n, 'activation_checkpoint', str(i))
    gm.recompile()
    return gm


def chen_sqrtn(gm: GraphModule) -> GraphModule:
    """
    This is the theoretical optimal strategy in https://arxiv.org/abs/1604.06174.

    Usage:
        model = resnet18()
        input_sample = torch.rand(4, 3, 224, 224)
        gm = symbolic_trace(model)
        MetaInfoProp(gm).run(input_sample)
        gm = chen_sqrtn(gm)

    Args:
        gm (GraphModule): The module to add checkpoints
    """
    gm.graph.lint()    # make sure nodes are in topological order
    k = int(len(gm.graph.nodes)**0.5)    # take approximately sqrt(n) checkpoints
    for idx, n in enumerate(gm.graph.nodes):
        # We should not add act_ckpt to the placeholder
        # The last segment should not be checkpointed
        if n.op != 'placeholder' and (idx + 1) // k < k:
            setattr(n, 'activation_checkpoint', str((idx + 1) // k))
    gm.recompile()
    return gm
