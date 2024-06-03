import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import colossalai
from colossalai.fx import ColoTracer
from colossalai.fx.graph_module import ColoGraphModule
from colossalai.legacy.core import global_context as gpc
from colossalai.testing import rerun_if_address_is_in_use, spawn

try:
    from colossalai.fx.codegen import ActivationCheckpointCodeGen

    with_codegen = True
except:
    # fall back to older pytorch version
    from colossalai.fx.codegen import python_code_with_activation_checkpoint

    with_codegen = False


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 4)
        self.linear2 = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.linear1(x), self.linear2(x)


class relu(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x)


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = MLP()
        self.relu = relu()
        self.linear2 = torch.nn.Linear(4, 4)

    def ckpt2(self, x):
        return F.relu(x, inplace=True)

    def ckpt3(self, x, y):
        return self.linear2(x) + self.linear2(y)

    def forward(self, x, y):
        y1, y2 = checkpoint(self.mlp1, x)
        y3 = checkpoint(self.relu, x)

        y4 = checkpoint(self.ckpt2, y)
        y5 = checkpoint(self.ckpt3, y, y4)
        y6 = self.linear2(y4)
        return y1 + y2 + y3 + y4 + y5 + y6


def _run_act_ckpt_codegen(rank, world_size, port):
    # launch colossalai to make sure we could execute colossalai.utils.checkpoint currently
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")

    # build model and run forward
    model = MyModule()
    data1 = torch.rand(4, 4)
    data2 = torch.rand(4, 4)

    # copy model to cuda
    model = model.to(device="cuda")
    data1 = data1.to(device="cuda")
    data2 = data2.to(device="cuda")

    non_fx_out = model(data1, data2)

    # trace the module and replace codegen
    tracer = ColoTracer(trace_act_ckpt=True)
    graph = tracer.trace(model)
    codegen = ActivationCheckpointCodeGen()
    graph.set_codegen(codegen)

    # check ops are annotated with ckpt
    # also annotate the selected node for offloading
    ckpt_nodes = ["mlp1_linear1", "mlp1_linear2", "relu_relu", "relu"]
    offload_starts = ["mlp1_linear1"]
    for node in graph.nodes:
        if node.name in ckpt_nodes:
            assert "activation_checkpoint" in node.meta

            # annotate the selected node for offload
            if node.name in offload_starts:
                node.meta["activation_offload"] = True

    gm = ColoGraphModule(model, graph)
    gm.recompile()

    # assert checkpoint function will be generated and
    # the offload option is correct
    code = graph.python_code("self").src
    assert (
        "colossalai.utils.activation_checkpoint.checkpoint(self.checkpoint_0, True, x, use_reentrant=False)" in code
        and "colossalai.utils.activation_checkpoint.checkpoint(self.checkpoint_1, False, x, use_reentrant=False)"
        in code
        and "colossalai.utils.activation_checkpoint.checkpoint(self.checkpoint_2, False, y, use_reentrant=False)"
        in code
        and "colossalai.utils.activation_checkpoint.checkpoint(self.checkpoint_3, False, y, relu, use_reentrant=True)"
        in code
    )

    # recompile and verify the outputs are consistent
    fx_out = gm(data1, data2)
    assert torch.equal(non_fx_out, fx_out)

    gpc.destroy()


@pytest.mark.skipif(not with_codegen, reason="torch version is lower than 1.12.0")
@rerun_if_address_is_in_use()
def test_act_ckpt_codegen():
    spawn(_run_act_ckpt_codegen, 1)


def _run_act_ckpt_python_code_torch11(rank, world_size, port):
    # launch colossalai to make sure we could execute colossalai.utils.checkpoint currently
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")

    # build model and run forward
    model = MyModule()
    data1 = torch.rand(4, 4)
    data2 = torch.rand(4, 4)

    # copy model to cuda
    data1 = data1.to(device="cuda")
    data2 = data2.to(device="cuda")

    non_fx_out = model(data1, data2)

    # trace the module and replace codegen
    tracer = ColoTracer(trace_act_ckpt=True)
    graph = tracer.trace(model)

    # replace a bound method of an object
    graph._python_code = python_code_with_activation_checkpoint.__get__(graph)

    # check ops are annotated with ckpt
    ckpt_nodes = ["mlp1_linear1", "mlp1_linear2", "relu_relu", "relu"]
    offload_starts = ["mlp1_linear1"]
    for node in graph.nodes:
        if node.name in ckpt_nodes:
            assert "activation_checkpoint" in node.meta

            # annotate the selected node for offload
            if node.name in offload_starts:
                node.meta["activation_offload"] = True

    gm = ColoGraphModule(model, graph)
    gm.recompile()
    # assert checkpoint function will be generated and
    # the offload option is correct
    code = graph.python_code("self").src
    assert (
        "colossalai.utils.activation_checkpoint.checkpoint(self.checkpoint_0, True, x, use_reentrant=False)" in code
        and "colossalai.utils.activation_checkpoint.checkpoint(self.checkpoint_1, False, x, use_reentrant=False)"
        in code
        and "colossalai.utils.activation_checkpoint.checkpoint(self.checkpoint_2, False, y, use_reentrant=False)"
        in code
        and "colossalai.utils.activation_checkpoint.checkpoint(self.checkpoint_3, False, y, relu, use_reentrant=True)"
        in code
    )

    # recompile and verify the outputs are consistent
    fx_out = gm(data1, data2)
    assert torch.equal(non_fx_out, fx_out)

    gpc.destroy()


@pytest.mark.skipif(with_codegen, reason="torch version is equal to or higher than 1.12.0")
@pytest.mark.skip(reason="currently torch11 ColoGraphModule is not done")
@rerun_if_address_is_in_use()
def test_act_ckpt_python_code_torch11():
    spawn(_run_act_ckpt_python_code_torch11, 1)


if __name__ == "__main__":
    _run_act_ckpt_codegen(rank=0)
