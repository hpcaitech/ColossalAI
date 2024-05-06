import copy
import re
from typing import Callable

import pytest
import torch
import torchvision.models as tm
from torch.fx import GraphModule

import colossalai
from colossalai.fx import ColoTracer
from colossalai.fx._compatibility import is_compatible_with_meta
from colossalai.fx.graph_module import ColoGraphModule

# from colossalai.fx.passes.algorithms import chen_greedy, solver_rotor
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.legacy.core import global_context as gpc
from colossalai.testing import rerun_if_address_is_in_use, spawn

if is_compatible_with_meta():
    from colossalai.fx.profiler.tensor import MetaTensor

try:
    from colossalai.fx.codegen import ActivationCheckpointCodeGen

    with_codegen = True
except:
    # fall back to older pytorch version
    from colossalai.fx.codegen import python_code_with_activation_checkpoint

    with_codegen = False

# SOLVERS = [chen_greedy, solver_rotor]
SOLVERS = []


def _is_activation_checkpoint_available(gm: GraphModule):
    for n in gm.graph.nodes:
        if hasattr(n, "activation_checkpoint") and getattr(n, "activation_checkpoint") is not None:
            return True


def _is_all_gradient_close(m: torch.nn.Module, gm: GraphModule):
    for m_p, gm_p in zip(m.parameters(), gm.parameters()):
        if not torch.allclose(m_p.grad, gm_p.grad):
            return False
    return True


def _is_graph_linearized(gm: GraphModule):
    code = gm.code
    # find patterns like r'      return output_1, output_2', which is not expected on a linearized graph
    pattern = re.compile(r"     return [a-zA-Z0-9_]+(, [a-zA-Z0-9_]+)+")
    if pattern.findall(code):
        return False
    else:
        return True


def check_backward_consistency(
    m: torch.nn.Module,
    gm: GraphModule,
    solver: Callable[[GraphModule], GraphModule],
    model_cls: Callable[[], torch.nn.Module],
):
    criterion = torch.nn.MSELoss()
    m.cuda()
    data = torch.rand(2, 3, 32, 32).cuda()
    label = torch.rand(2, 5).cuda()
    loss = criterion(m(data), label)
    loss.backward()
    loss = criterion(gm(data), label)
    loss.backward()
    assert _is_all_gradient_close(m, gm), f"Solver {solver} did not work correctly in backward pass on {model_cls}"


def _run_ckpt_solver(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    MODEL_LIST = [tm.densenet121]

    torch.backends.cudnn.deterministic = True

    tracer = ColoTracer(trace_act_ckpt=False)

    data = torch.rand(8, 3, 224, 224, device="meta")
    for solver in SOLVERS:
        for model_cls in MODEL_LIST:
            m = model_cls(num_classes=5)
            graph = tracer.trace(root=m)
            gm = ColoGraphModule(copy.deepcopy(m), graph, m.__class__.__name__)
            MetaInfoProp(gm.cuda()).run(MetaTensor(data).cuda())
            codegen = ActivationCheckpointCodeGen()
            gm.graph.set_codegen(codegen)
            if solver == solver_rotor:
                gm = solver(gm, data, mem_limit=500 * 1024 * 1024, mem_slots=500)
            else:
                gm = solver(gm)
            assert _is_graph_linearized(gm), f"Solver {solver} did not solve {model_cls} in a linearized manner."
            assert _is_activation_checkpoint_available(
                gm
            ), f"Solver {solver} did not annotate {model_cls} with any activation checkpoints"
            check_backward_consistency(m, gm, solver, model_cls)
    gpc.destroy()


@pytest.mark.skip("TODO(super-dainiu): refactor all tests.")
@pytest.mark.skipif(not with_codegen, reason="torch version is lower than 1.12.0")
@rerun_if_address_is_in_use()
def test_ckpt_solver():
    spawn(_run_ckpt_solver, 1)


def _run_ckpt_solver_torch11(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    MODEL_LIST = [tm.densenet121]

    torch.backends.cudnn.deterministic = True

    tracer = ColoTracer(trace_act_ckpt=False)

    data = torch.rand(8, 3, 32, 32, device="meta")
    for solver in SOLVERS:
        for model_cls in MODEL_LIST:
            m = model_cls(num_classes=5)
            graph = tracer.trace(root=m)
            gm = ColoGraphModule(copy.deepcopy(m), graph, m.__class__.__name__)
            MetaInfoProp(gm).run(data)
            gm.graph._python_code = python_code_with_activation_checkpoint.__get__(graph)
            if solver == solver_rotor:
                gm = solver(gm, data, mem_limit=500 * 1024 * 1024, mem_slots=500, force_python=True)
            else:
                gm = solver(gm)
            assert _is_graph_linearized(gm), f"Solver {solver} did not solve {model_cls} in a linearized manner."
            assert _is_activation_checkpoint_available(
                gm
            ), f"Solver {solver} did not annotate {model_cls} with any activation checkpoints"
            check_backward_consistency(m, gm, solver, model_cls)
    gpc.destroy()


@pytest.mark.skipif(with_codegen, reason="torch version is equal to or higher than 1.12.0")
@pytest.mark.skip(reason="currently torch11 ColoGraphModule is not done")
@rerun_if_address_is_in_use()
def test_ckpt_solver_torch11():
    spawn(_run_ckpt_solver_torch11, 1)


if __name__ == "__main__":
    _run_ckpt_solver(rank=0)
    test_ckpt_solver()
    test_ckpt_solver_torch11()
