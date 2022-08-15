from typing import Callable
import copy
import torch
import torch.multiprocessing as mp
import torchvision.models as tm
from torch.fx import GraphModule
import colossalai
from colossalai.fx import ColoTracer
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.fx.passes.algorithms import chen_greedy, chen_sqrtn
from colossalai.utils import free_port
from colossalai.core import global_context as gpc
import pytest

try:
    from colossalai.fx.codegen import ActivationCheckpointCodeGen
    with_codegen = True
except:
    # fall back to older pytorch version
    from colossalai.fx.codegen import python_code_with_activation_checkpoint
    with_codegen = False

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
    data = torch.rand(2, 3, 32, 32)
    label = torch.rand(2, 5)
    loss = criterion(m(data), label)
    loss.backward()
    loss = criterion(gm(data), label)
    loss.backward()
    assert _is_all_gradient_close(m, gm), f'Solver {solver} did not work correctly in backward pass on {model_cls}'


def _run_ckpt_solver(rank):
    colossalai.launch(config={}, rank=rank, world_size=1, host='localhost', port=free_port(), backend='nccl')
    MODEL_LIST = [tm.resnet18, tm.densenet121]

    torch.backends.cudnn.deterministic = True

    tracer = ColoTracer(trace_act_ckpt=False)

    data = torch.rand(2, 3, 32, 32)
    for solver in SOLVERS:
        for model_cls in MODEL_LIST:
            m = model_cls(num_classes=5)
            graph = tracer.trace(root=m)
            gm = GraphModule(copy.deepcopy(m), graph, m.__class__.__name__)
            MetaInfoProp(gm).run(data)
            codegen = ActivationCheckpointCodeGen()
            gm.graph.set_codegen(codegen)
            gm = solver(gm)
            assert _is_activation_checkpoint_available(
                gm), f"Solver {solver} did not annotate {model_cls} with any activation checkpoints"
            check_backward_consistency(m, gm, solver, model_cls)


@pytest.mark.skip
@pytest.mark.skipif(not with_codegen, reason='torch version is lower than 1.12.0')
def test_ckpt_solver():
    mp.spawn(_run_ckpt_solver, nprocs=1)


def _run_ckpt_solver_torch11(rank):
    colossalai.launch(config={}, rank=rank, world_size=1, host='localhost', port=free_port(), backend='nccl')
    MODEL_LIST = [tm.resnet18, tm.densenet121]

    torch.backends.cudnn.deterministic = True

    tracer = ColoTracer(trace_act_ckpt=False)

    data = torch.rand(2, 3, 32, 32)
    for solver in SOLVERS:
        for model_cls in MODEL_LIST:
            m = model_cls(num_classes=5)
            graph = tracer.trace(root=m)
            gm = GraphModule(copy.deepcopy(m), graph, m.__class__.__name__)
            MetaInfoProp(gm).run(data)
            gm.graph._python_code = python_code_with_activation_checkpoint.__get__(graph)
            gm = solver(gm)
            assert _is_activation_checkpoint_available(
                gm), f"Solver {solver} did not annotate {model_cls} with any activation checkpoints"
            check_backward_consistency(m, gm, solver, model_cls)


@pytest.mark.skip
@pytest.mark.skipif(with_codegen, reason='torch version is equal to or higher than 1.12.0')
def test_ckpt_solver_torch11():
    mp.spawn(_run_ckpt_solver_torch11, nprocs=1)


if __name__ == '__main__':
    test_ckpt_solver()
    test_ckpt_solver_torch11()
