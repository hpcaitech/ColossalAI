import copy
import torch
import torch.nn.functional as F
import pytest
import torch.multiprocessing as mp
from torch.fx import GraphModule
from colossalai.fx import ColoTracer
import colossalai
from colossalai.utils import free_port
from colossalai.core import global_context as gpc
from colossalai.fx.graph_module import ColoGraphModule

try:
    from chunk_codegen import ChunkCodeGen
    with_codegen = True
except:
    # fall back to older pytorch version
    from chunk_codegen import python_code_with_activation_checkpoint
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

    # test barckward
    loss0 = non_fx_out.sum()
    loss0.backward()
    loss1 = fx_out.sum()
    loss1.backward()
    assert _is_all_gradient_close(model, gm), "gm doesn't have the same gradient as original one"


def _run_offload_codegen(rank):
    # launch colossalai to make sure we could execute colossalai.utils.checkpoint currectly
    colossalai.launch(config={}, rank=rank, world_size=1, host='localhost', port=free_port(), backend='nccl')

    # build model and input
    model = MyNet().cuda()
    data = torch.rand(4, 4).cuda()

    # trace the module and replace codegen
    tracer = ColoTracer(trace_act_ckpt=True)
    graph = tracer.trace(model)
    codegen = ChunkCodeGen()
    graph.set_codegen(codegen)

    # annotate the activation offload part
    # also annotate the activation_checkpoint so we could test both types
    # of input offload
    for node in graph.nodes:
        if node.name == "linear0":
            setattr(node, "activation_offload", [0, True, False])
        if node.name == "linear1":
            setattr(node, "activation_offload", [0, True, False])
        if node.name == "linear2":
            setattr(node, "activation_offload", [1, True, True])
        if node.name == "linear4":
            setattr(node, "activation_offload", [2, False, True])
        if node.name == "linear5":
            setattr(node, "activation_checkpoint", [0])
            setattr(node, "activation_offload", True)

    gm = ColoGraphModule(copy.deepcopy(model), graph)
    gm.recompile()

    # assert we have all the components
    code = graph.python_code("self").src
    print(code)

    _test_fwd_and_bwd(model, gm, data)
    gpc.destroy()


@pytest.mark.skipif(not with_codegen, reason='torch version is lower than 1.12.0')
def test_act_ckpt_codegen():
    mp.spawn(_run_offload_codegen, nprocs=1)


if __name__ == "__main__":
    _run_offload_codegen(0)
