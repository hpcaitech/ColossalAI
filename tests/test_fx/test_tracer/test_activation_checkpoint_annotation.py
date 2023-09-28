import torch
from torch.fx import GraphModule
from torch.utils.checkpoint import checkpoint

from colossalai.fx import ColoTracer
from colossalai.testing import clear_cache_before_run


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
        self.mlp_1 = MLP()
        self.mlp_2 = MLP()
        self.output = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = checkpoint(self.mlp_1, x)
        x = checkpoint(self.mlp_2, x)
        x = self.output(x)
        return x


@clear_cache_before_run()
def test_activation_checkpoint_annotation():
    module = MyModule()

    # test tracing with activation checkpoint
    tracer = ColoTracer(trace_act_ckpt=True)
    graph = tracer.trace(module)
    gm = GraphModule(module, graph)

    for node in gm.graph.nodes:
        if node.name in ["mlp_1_linear1", "mlp_1_linear2"]:
            assert node.meta.get("activation_checkpoint", -1) == 0

    for node in gm.graph.nodes:
        if node.name in ["mlp_2_linear1", "mlp_2_linear2"]:
            assert node.meta.get("activation_checkpoint", -1) == 1

    tracer = ColoTracer(trace_act_ckpt=False)
    graph = tracer.trace(module)
    gm = GraphModule(module, graph)

    for node in gm.graph.nodes:
        assert not hasattr(node, "activation_checkpoint")


if __name__ == "__main__":
    test_activation_checkpoint_annotation()
