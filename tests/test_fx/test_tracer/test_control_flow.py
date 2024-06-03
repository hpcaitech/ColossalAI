import torch
import torch.nn as nn
from torch.fx import GraphModule

from colossalai.fx import ColoTracer as Tracer
from colossalai.testing import clear_cache_before_run


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


@clear_cache_before_run()
def test_control_flow():
    model = ControlFlowModel()
    tracer = Tracer()
    graph_branch_true = tracer.trace(
        model, meta_args={"x": torch.rand(4, 10, device="meta"), "y": torch.rand(4, 10, device="meta")}
    )
    graph_branch_false = tracer.trace(
        model, meta_args={"x": torch.rand(10, device="meta"), "y": torch.rand(4, 10, device="meta")}
    )

    gm_branch_true = GraphModule(model, graph_branch_true, model.__class__.__name__)
    gm_branch_false = GraphModule(model, graph_branch_false, model.__class__.__name__)
    gm_branch_true.recompile()
    gm_branch_false.recompile()

    # test the true branch
    x = torch.rand(4, 10)
    y = torch.rand(4, 10)
    assert torch.all(model(x, y) == gm_branch_true(x, y))
    assert torch.all(gm_branch_false(x, y) != gm_branch_true(x, y))

    # test the true branch
    x = torch.rand(10)
    y = torch.rand(4, 10)
    assert torch.all(model(x, y) == gm_branch_false(x, y))
    assert torch.all(gm_branch_false(x, y) != gm_branch_true(x, y))


if __name__ == "__main__":
    test_control_flow()
