import torch
import torch.nn as nn
from torch.fx import GraphModule
from colossalai.fx import ColoTracer as Tracer


class ControlFlowModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 10)

    def forward(self, x, y):
        x1 = self.linear1(x)
        y1 = self.linear2(y)

        if x1.dim() == 2:
            return x1 + y1
        else:
            return x1 - y1


def test_control_flow():
    model = ControlFlowModel()
    tracer = Tracer()
    graph = tracer.trace(model,
                         meta_args={
                             'x': torch.rand(4, 10, device='meta'),
                             'y': torch.rand(4, 10, device='meta')
                         })
    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()

    x = torch.rand(4, 10)
    y = torch.rand(4, 10)

    assert torch.all(model(x, y) == gm(x, y))


if __name__ == '__main__':
    test_control_flow()
