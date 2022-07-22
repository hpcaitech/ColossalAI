import torch
import pytest
from torch.utils.checkpoint import checkpoint
from torch.fx import GraphModule
from colossalai.fx import ColoTracer

try:
    from colossalai.fx.codegen import ActivationCheckpointCodeGen
except:
    pass


class MLP(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 4)
        self.linear2 = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.linear1(x), self.linear1(x)


class MyModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mlp1 = MLP()
        self.mlp2 = MLP()
        self.linear3 = torch.nn.Linear(4, 4)

    def forward(self, x):
        y1, y2 = checkpoint(self.mlp1, x)
        y3, y4 = checkpoint(self.mlp2, x)
        return y1 + y2 + y3 + y4


@pytest.mark.skip("torch 1.12 is required")
def test_act_ckpt_codegen():
    # build model and run forward
    model = MyModule()
    data = torch.rand(4, 4)
    non_fx_out = model(data)

    # trace the module and replace codegen
    tracer = ColoTracer(trace_act_ckpt=True)
    graph = tracer.trace(model)
    codegen = ActivationCheckpointCodeGen()
    graph.set_codegen(codegen)

    # check ops are annotated with ckpt
    ckpt_nodes = ['mlp1_linear1', 'mlp1_linear1_1', 'mlp2_linear1', 'mlp2_linear1_1']
    for node in graph.nodes:
        if node.name in ckpt_nodes:
            assert hasattr(node, 'activation_checkpoint')

    # assert checkpoint function will be generated
    code = graph.python_code('self').src
    assert 'checkpoint_0' in code and 'checkpoint_1' in code

    # recompile and verify the outputs are consistent
    gm = GraphModule(model, graph)
    gm.recompile()
    fx_out = gm(data)
    assert torch.equal(non_fx_out, fx_out)


if __name__ == '__main__':
    test_act_ckpt_codegen()
