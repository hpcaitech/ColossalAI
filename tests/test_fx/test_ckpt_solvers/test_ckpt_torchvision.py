from ctypes import Union
from typing import Callable
from colossalai.fx.passes.algorithms import chen_greedy, chen_sqrtn
import torch
import torchvision.models as tm
from colossalai.fx import ColoTracer
from torch.fx import GraphModule
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
import pytest

SOLVERS = [chen_greedy, chen_sqrtn]


def _is_activation_checkpoint_available(gm: GraphModule):
    for n in gm.graph.nodes:
        if hasattr(n, 'activation_checkpoint') and getattr(n, 'activation_checkpoint') is not None:
            return True


def _is_all_gradient_close(m: torch.nn.Module, gm: GraphModule):
    for m_p, gm_p in zip(m.parameters(), gm.parameters()):
        if not torch.allclose(m_p.grad, gm_p.grad):
            return False
    return True


def check_backward_consistency(m: torch.nn.Module, gm: GraphModule, solver: Callable[[GraphModule], GraphModule],
                               model_cls: Callable[[], torch.nn.Module]):
    criterion = torch.nn.MSELoss()
    data = torch.rand(2, 3, 24, 24)
    label = torch.rand(2, 5)
    loss = criterion(m(data), label)
    loss.backward()
    loss = criterion(gm(data), label)
    loss.backward()
    assert _is_all_gradient_close(m, gm), f'Solver {solver} did not work correctly in backward pass on {model_cls}'


def test_ckpt_solver():
    MODEL_LIST = [tm.resnet18]

    torch.backends.cudnn.deterministic = True

    tracer = ColoTracer()

    data = torch.rand(2, 3, 24, 24)
    for solver in SOLVERS:
        for model_cls in MODEL_LIST:
            model = model_cls(num_classes=5)
            graph = tracer.trace(root=model)
            gm = GraphModule(model, graph, model.__class__.__name__)
            MetaInfoProp(gm).run(data)
            gm = solver(gm)
            assert _is_activation_checkpoint_available(
                gm), f"Solver {solver} did not annotate {model_cls} with any activation checkpoints"
            check_backward_consistency(model, gm, solver, model_cls)


if __name__ == '__main__':
    test_ckpt_solver()
