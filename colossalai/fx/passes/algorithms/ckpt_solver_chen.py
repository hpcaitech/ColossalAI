from typing import Set, Tuple
import torch
from torch.fx import GraphModule
import math

__all__ = ['chen_greedy', 'chen_sqrtn']


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
        Search ckpt strategy with B = 0, then run the allocation algorithm again with B = √xy.
        Grid search over [√2/2 B, √2 B] for ckpt_opt over num_grids as in appendix A.
        """
        _, B_approx = run_chen_greedy(0)
        B_min, B_max = math.floor(B_approx / math.sqrt(2)), math.ceil(B_approx * math.sqrt(2))
        B_opt = math.inf
        for B in range(B_min, B_max, (B_max - B_min) // num_grids):
            ckpt, B_approx = run_chen_greedy(B)
            if B_approx < B_opt:
                B_opt = B_approx
                ckpt_opt = ckpt
        return ckpt_opt

    def run_chen_greedy(B: int = 0) -> Tuple[Set, int]:
        """
        This is the simple implementation of Algorithm 3 in https://arxiv.org/abs/1604.06174.
        """
        ckpt = set()
        temp = 0
        x = 0
        y = 0
        for (idx, n) in enumerate(gm.graph.nodes):
            temp += getattr(n, 'activation_size')
            y = max(y, temp)
            if temp > B:
                x += getattr(n, 'activation_size')
                temp = 0
                ckpt.add(idx)
        return ckpt, math.floor(math.sqrt(x * y))

    gm.graph.lint()    # make sure nodes are in topological order
    ckpt = grid_search(num_grids=6)
    i = 0
    for idx, n in enumerate(gm.graph.nodes):
        if idx in ckpt:
            setattr(n, 'activation_checkpoint', str(i))
            i += 1
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
        if (idx + 1) % k == 0:
            setattr(n, 'activation_checkpoint', str((idx + 1) // k))
    gm.recompile()
    return gm
