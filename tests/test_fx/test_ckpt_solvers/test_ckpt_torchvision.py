from colossalai.fx.passes.algorithms import chen_greedy, chen_sqrtn
import torch
import torchvision.models as tm
from colossalai.fx import ColoTracer
from torch.fx import GraphModule
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from functools import partial
import pytest

SOLVERS = [partial(chen_greedy, B=1024 * 1024 * 64), chen_sqrtn]


def _is_activation_checkpoint_available(gm: GraphModule):
    for n in gm.graph.nodes:
        if hasattr(n, 'activation_checkpoint') and getattr(n, 'activation_checkpoint') is not None:
            return True


def test_ckpt_solver():
    MODEL_LIST = [tm.resnet18, tm.densenet121]

    torch.backends.cudnn.deterministic = True

    tracer = ColoTracer()
    data = torch.rand(1, 3, 224, 224)

    for solver in SOLVERS:
        for model_cls in MODEL_LIST:
            model = model_cls()
            graph = tracer.trace(root=model)
            gm = GraphModule(model, graph, model.__class__.__name__)
            MetaInfoProp(gm).run(data)
            gm = solver(gm)
            assert _is_activation_checkpoint_available(
                gm), f"Solver {solver} did not annotate {model_cls} with any activation checkpoints"
            assert torch.allclose(gm(data), model(data))


if __name__ == '__main__':
    test_ckpt_solver()
