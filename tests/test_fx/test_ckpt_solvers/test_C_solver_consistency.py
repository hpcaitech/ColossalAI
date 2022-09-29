import copy
import torch
import torch.multiprocessing as mp
import torchvision.models as tm
import torch.fx
import colossalai
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.fx.passes.algorithms import solver_rotor
from colossalai.fx.passes.algorithms.operation import Sequence
from colossalai.core import global_context as gpc
from colossalai.utils import free_port
import pytest
from colossalai import META_COMPATIBILITY
if META_COMPATIBILITY:
    from colossalai.fx.profiler.tensor import MetaTensor

try:
    from colossalai.fx.codegen import ActivationCheckpointCodeGen
    withcodegen = True
except:
    from colossalai.fx.codegen import python_code_with_activation_checkpoint
    withcodegen = False


def _run_C_solver_consistency_test(rank=0):
    colossalai.launch(config={}, rank=rank, world_size=1, host='localhost', port=free_port(), backend='nccl')

    for M, mem_budget in [(tm.resnet18, 2000), (tm.resnet50, 8000)]:
        model = M()
        data = torch.rand(128, 3, 224, 224, device='meta')

        tracer = ColoTracer()
        graph = tracer.trace(model, meta_args={"x": data})
        graph.set_codegen(ActivationCheckpointCodeGen())
        gm = ColoGraphModule(model, graph, model.__class__.__name__)
        if META_COMPATIBILITY:
            data_meta = MetaTensor(data, fake_device=next(gm.parameters()).device)
        MetaInfoProp(gm).run(data_meta)

        # python solver
        gm = solver_rotor(gm, data_meta, mem_budget * 1024 * 1024, force_python=True)
        sequence_python: Sequence = copy.deepcopy(gm.__sequence__)

        # C solver
        gm = solver_rotor(gm, data_meta, mem_budget * 1024 * 1024)
        sequence_C: Sequence = copy.deepcopy(gm.__sequence__)

        sequence_python = sequence_python.list_operations()
        sequence_C = sequence_C.list_operations()

        # make sure the solutions are the same
        assert len(sequence_python) == len(sequence_C) and \
        all(python_op.__repr__() == C_op.__repr__() for (python_op, C_op) in zip(sequence_python, sequence_C))

    gpc.destroy()


@pytest.mark.skipif(not withcodegen, reason="torch version is less than 1.12.0")
def test_C_solver_consistency():
    mp.spawn(_run_C_solver_consistency_test, nprocs=1)


if __name__ == '__main__':
    _run_C_solver_consistency_test(rank=0)
