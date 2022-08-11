import torch
from torch.fx import GraphModule

__all__ = ['chen_greedy', 'chen_sqrtn']


def chen_greedy(gm: GraphModule, B: int):
    """
    This is the simple implementation of Algorithm 3 in https://arxiv.org/abs/1604.06174.

    Usage:
        B = 5 * 1024 * 1024 * 1024  # An approximate memory budget of 5GB
        model = resnet18()
        input_sample = torch.rand(4, 3, 224, 224)
        gm = symbolic_trace(model)
        MetaInfoProp(gm).run(input_sample)
        gm = chen_greedy(gm, B)

    Args:
        gm (GraphModule): The module to add checkpoints
        B (int): The approximate memory budget for this module.
    """
    gm.graph.lint()    # make sure nodes are in topological order
    temp = 0
    x = 0
    idx = 0
    budget = B
    for n in gm.graph.nodes:
        B -= getattr(n, 'param_size')
        assert B > 0, f'The memory budget {budget / 1024 ** 3:.2f} GB is not enough for model parameters of {gm}'
    for n in gm.graph.nodes:
        temp += getattr(n, 'activation_size')
        if temp > B:
            x += getattr(n, 'activation_size')
            temp = x
            setattr(n, 'activation_checkpoint', str(idx))
            idx += 1
    gm.recompile()
    return gm


def chen_sqrtn(gm: GraphModule):
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
