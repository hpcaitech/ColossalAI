import copy

import pytest
import torch
from torch.fx import GraphModule

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


class MyNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear0 = torch.nn.Linear(4, 4)
        self.linear1 = torch.nn.Linear(4, 4)
        self.linear2 = torch.nn.Linear(4, 4)
        self.linear3 = torch.nn.Linear(4, 4)
        self.linear4 = torch.nn.Linear(4, 4)
        self.linear5 = torch.nn.Linear(4, 4)
        self.linear6 = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = self.linear0(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.linear6(x)
        return x


def _is_all_gradient_close(m: torch.nn.Module, gm: GraphModule) -> bool:
    for m_p, gm_p in zip(m.parameters(), gm.parameters()):
        if not torch.allclose(m_p.grad, gm_p.grad):
            return False
    return True


def _test_fwd_and_bwd(model: torch.nn.Module, gm: ColoGraphModule, data: torch.Tensor):
    # test forward
    non_fx_out = model(data)
    fx_out = gm(data)
    assert torch.equal(non_fx_out, fx_out), "fx_out doesn't comply with original output"

    # test backward
    loss0 = non_fx_out.sum()
    loss0.backward()
    loss1 = fx_out.sum()
    loss1.backward()
    assert _is_all_gradient_close(model, gm), "gm doesn't have the same gradient as original one"


def _run_offload_codegen(rank, world_size, port):
    # launch colossalai to make sure we could execute colossalai.utils.checkpoint currently
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")

    # build model and input
    model = MyNet().cuda()
    data = torch.rand(4, 4).cuda()

    # trace the module and replace codegen
    tracer = ColoTracer(trace_act_ckpt=True)
    graph = tracer.trace(model)
    codegen = ActivationCheckpointCodeGen()
    graph.set_codegen(codegen)

    # annotate the activation offload part
    # also annotate the activation_checkpoint so we could test both types
    # of input offload
    for node in graph.nodes:
        if node.name == "linear0":
            node.meta["activation_offload"] = [0, True, False]
        if node.name == "linear1":
            node.meta["activation_offload"] = [0, True, False]
        if node.name == "linear2":
            node.meta["activation_offload"] = [1, True, True]
        if node.name == "linear4":
            node.meta["activation_offload"] = [2, False, True]
        if node.name == "linear5":
            node.meta["activation_checkpoint"] = [0]
            node.meta["activation_offload"] = True

    gm = ColoGraphModule(copy.deepcopy(model), graph)
    gm.recompile()

    # assert we have all the components
    code = graph.python_code("self").src
    assert (
        "def pack_hook_input(self, x):" in code
        and "def unpack_hook(self, packed):" in code
        and "def pack_hook_no_input(self, x):" in code
        and "setattr(x, 'offload', True)" in code
        and "setattr(linear3, 'offload', False)" in code
        and "with torch.autograd.graph.saved_tensors_hooks(self.pack_hook_input, self.unpack_hook):" in code
        and "with torch.autograd.graph.save_on_cpu(pin_memory=True):" in code
        and "with torch.autograd.graph.saved_tensors_hooks(self.pack_hook_no_input, self.unpack_hook):" in code
        and "colossalai.utils.activation_checkpoint.checkpoint(self.checkpoint_0, True, linear4, use_reentrant=False)"
        in code
    )

    _test_fwd_and_bwd(model, gm, data)
    gpc.destroy()


@pytest.mark.skipif(not with_codegen, reason="torch version is lower than 1.12.0")
@rerun_if_address_is_in_use()
def test_act_ckpt_codegen():
    spawn(_run_offload_codegen, 1)


def _run_offload_codegen_torch11(rank, world_size, port):
    # launch colossalai to make sure we could execute colossalai.utils.checkpoint currently
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")

    # build model and input
    model = MyNet().cuda()
    data = torch.rand(4, 4).cuda()

    # trace the module and replace codegen
    tracer = ColoTracer(trace_act_ckpt=True)
    graph = tracer.trace(model)

    # replace a bound method of an object
    graph._python_code = python_code_with_activation_checkpoint.__get__(graph)

    # annotate the activation offload part
    # also annotate the activation_checkpoint so we could test both types
    # of input offload
    for node in graph.nodes:
        if node.name == "linear0":
            node.meta["activation_offload"] = [0, True, False]
        if node.name == "linear1":
            node.meta["activation_offload"] = [0, True, False]
        if node.name == "linear2":
            node.meta["activation_offload"] = [1, True, True]
        if node.name == "linear4":
            node.meta["activation_offload"] = [2, False, True]
        if node.name == "linear5":
            node.meta["activation_checkpoint"] = [0]
            node.meta["activation_offload"] = True

    gm = ColoGraphModule(copy.deepcopy(model), graph)
    gm.recompile()

    # assert we have all the components
    code = graph.python_code("self").src
    assert (
        "def pack_hook_input(self, x):" in code
        and "def unpack_hook(self, packed):" in code
        and "def pack_hook_no_input(self, x):" in code
        and "setattr(x, 'offload', True)" in code
        and "setattr(linear3, 'offload', False)" in code
        and "with torch.autograd.graph.saved_tensors_hooks(self.pack_hook_input, self.unpack_hook):" in code
        and "with torch.autograd.graph.save_on_cpu(pin_memory=True):" in code
        and "with torch.autograd.graph.saved_tensors_hooks(self.pack_hook_no_input, self.unpack_hook):" in code
        and "colossalai.utils.activation_checkpoint.checkpoint(self.checkpoint_0, True, linear4, use_reentrant=False)"
        in code
    )

    _test_fwd_and_bwd(model, gm, data)
    gpc.destroy()


@pytest.mark.skip(reason="currently torch11 ColoGraphModule is not implemented")
@rerun_if_address_is_in_use()
def test_act_ckpt_python_code_torch11():
    spawn(_run_offload_codegen_torch11, 1)


if __name__ == "__main__":
    _run_offload_codegen(0)
