import torch
import torch.nn as nn
from colossalai.fx import ColoTracer
from torch.fx import GraphModule
from torch.utils.checkpoint import checkpoint


class MLP(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 4)
        self.linear2 = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


# Simple module for demonstration
class MyModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mlp = MLP()
        self.output = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = checkpoint(self.mlp, x)
        x = self.output(x)
        return x


def test_activation_checkpoint_annotation():
    module = MyModule()

    # test tracing with activation checkpoint
    tracer = ColoTracer(trace_act_ckpt=True)
    graph = tracer.trace(module)
    gm = GraphModule(module, graph)

    # print([f"{node.name} cktp = {getattr(node, 'ckpt', False)}" for node in gm.graph.nodes])

    for node in gm.graph.nodes:
        if node.name in ['mlp_linear1', 'mlp_linear2']:
            assert getattr(node, 'activation_checkpoint', False)

    tracer = ColoTracer(trace_act_ckpt=False)
    graph = tracer.trace(module)
    gm = GraphModule(module, graph)

    for node in gm.graph.nodes:
        assert not getattr(node, 'activation_checkpoint', False)


if __name__ == '__main__':
    test_activation_checkpoint_annotation()
