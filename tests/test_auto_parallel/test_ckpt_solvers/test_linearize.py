import pytest
import torch
import torchvision.models as tm

from colossalai.fx import ColoTracer
from colossalai.fx._compatibility import is_compatible_with_meta
from colossalai.fx.graph_module import ColoGraphModule

# from colossalai.fx.passes.algorithms import linearize, solver_rotor
# from colossalai.fx.passes.algorithms.operation import (ForwardCheck, ForwardEnable, ForwardNograd, Loss)
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.testing import clear_cache_before_run

if is_compatible_with_meta():
    from colossalai.fx.profiler.tensor import MetaTensor

try:
    from colossalai.fx.codegen import ActivationCheckpointCodeGen

    with_codegen = True
except:
    # fall back to older pytorch version
    from colossalai.fx.codegen import python_code_with_activation_checkpoint

    with_codegen = False


@pytest.mark.skip(reason="TODO: modify the logger")
@pytest.mark.skip("TODO(lyl): refactor all tests.")
@pytest.mark.skipif(not with_codegen, reason="torch version is lower than 1.12.0")
@clear_cache_before_run()
def test_linearize():
    MODEL_DICT = {tm.resnet18: [2100, 3000], tm.densenet121: [8100, 17000]}
    tracer = ColoTracer()
    for M, budgets in MODEL_DICT.items():
        for budget in budgets:
            model = M()
            graph = tracer.trace(model)
            graph.set_codegen(ActivationCheckpointCodeGen())
            gm = ColoGraphModule(model, graph, model.__class__.__name__)
            MetaInfoProp(gm).run(MetaTensor(torch.rand(128, 3, 224, 224, device="meta"), fake_device="cpu"))
            node_list = linearize(gm)
            gm = solver_rotor(gm, data=torch.rand(128, 3, 224, 224, device="meta"), mem_limit=budget * 1024**2)
            op_list = gm.__sequence__.list_operations()
            loss_op = next(op for op in op_list if isinstance(op, Loss))
            op_list = op_list[: op_list.index(loss_op)]
            in_ckpt = False
            ckpt_idx = 0
            for idx, op in enumerate(op_list):
                if in_ckpt:
                    if isinstance(op, ForwardNograd):
                        for n in node_list[idx]:
                            assert hasattr(n, "activation_checkpoint"), f"{n} is not annotated!"
                            assert (
                                n.activation_checkpoint[0] == ckpt_idx
                            ), f"{n} ckpt_idx {n.activation_checkpoint[0]} wrong, should be {ckpt_idx}!"

                        continue

                    if isinstance(op, ForwardEnable):
                        for n in node_list[idx]:
                            assert getattr(n, "activation_checkpoint", None) == None, f"{n} should not be annotated!"
                            in_ckpt = False

                        ckpt_idx += 1
                        continue

                    if isinstance(op, ForwardCheck):
                        ckpt_idx += 1
                        for n in node_list[idx]:
                            assert hasattr(n, "activation_checkpoint"), f"{n} is not annotated!"
                            assert (
                                n.activation_checkpoint[0] == ckpt_idx
                            ), f"{n} ckpt_idx {n.activation_checkpoint[0]} wrong, should be {ckpt_idx}!"

                        continue

                else:
                    if isinstance(op, ForwardCheck):
                        in_ckpt = True
                        for n in node_list[idx]:
                            assert hasattr(n, "activation_checkpoint"), f"{n} is not annotated!"
                            assert (
                                n.activation_checkpoint[0] == ckpt_idx
                            ), f"{n} ckpt_idx {n.activation_checkpoint[0]} wrong, should be {ckpt_idx}!"

            del model
            del gm
            del node_list


@pytest.mark.skip("TODO(lyl): refactor all tests.")
@pytest.mark.skip(reason="torch11 meta tensor not implemented")
@pytest.mark.skipif(with_codegen, reason="torch version is equal to or higher than 1.12.0")
@clear_cache_before_run()
def test_linearize_torch11():
    MODEL_DICT = {tm.resnet18: [2100, 3000], tm.densenet121: [8100, 17000]}
    tracer = ColoTracer()
    for M, budgets in MODEL_DICT.items():
        for budget in budgets:
            model = M()
            graph = tracer.trace(model)
            gm = ColoGraphModule(model, graph, model.__class__.__name__)
            gm.graph._python_code = python_code_with_activation_checkpoint.__get__(graph)
            node_list = linearize(gm)
            gm = solver_rotor(gm, data=torch.rand(128, 3, 224, 224, device="meta"), mem_limit=budget * 1024**2)
            op_list = gm.__sequence__.list_operations()
            loss_op = next(op for op in op_list if isinstance(op, Loss))
            op_list = op_list[: op_list.index(loss_op)]
            in_ckpt = False
            ckpt_idx = 0
            for idx, op in enumerate(op_list):
                if in_ckpt:
                    if isinstance(op, ForwardNograd):
                        for n in node_list[idx]:
                            assert hasattr(n, "activation_checkpoint"), f"{n} is not annotated!"
                            assert n.activation_checkpoint == ckpt_idx, f"{n} ckpt_idx wrong, should be {ckpt_idx}!"

                        continue

                    if isinstance(op, ForwardEnable):
                        for n in node_list[idx]:
                            assert getattr(n, "activation_checkpoint", None) == None, f"{n} should not be annotated!"
                            in_ckpt = False

                        ckpt_idx += 1
                        continue

                    if isinstance(op, ForwardCheck):
                        ckpt_idx += 1
                        for n in node_list[idx]:
                            assert hasattr(n, "activation_checkpoint"), f"{n} is not annotated!"
                            assert n.activation_checkpoint == ckpt_idx, f"{n} ckpt_idx wrong, should be {ckpt_idx}!"

                        continue

                else:
                    if isinstance(op, ForwardCheck):
                        in_ckpt = True
                        for n in node_list[idx]:
                            assert hasattr(n, "activation_checkpoint"), f"{n} is not annotated!"
                            assert n.activation_checkpoint == ckpt_idx, f"{n} ckpt_idx wrong, should be {ckpt_idx}!"

            del model
            del gm
            del node_list


if __name__ == "__main__":
    test_linearize()
