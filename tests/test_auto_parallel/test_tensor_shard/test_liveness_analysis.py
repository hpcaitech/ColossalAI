import torch
import torch.nn as nn

from colossalai.auto_parallel.tensor_shard.solver import GraphAnalyser
from colossalai.fx import ColoGraphModule, ColoTracer


class LinearModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(4, 4)

    def forward(self, x1, x2):
        x1 = x1 * 2
        x1 = self.linear1(x1)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        out = x1 + x2
        return out


def test_liveness_analysis():
    model = LinearModel()
    tracer = ColoTracer()
    graph = tracer.trace(model,
                         meta_args={
                             'x1': torch.rand(4, 4, device='meta'),
                             'x2': torch.rand(4, 4, device='meta')
                         })
    gm = ColoGraphModule(root=model, graph=graph, class_name=model.__class__.__name__)

    graph_analyser = GraphAnalyser(gm)
    liveness_list = graph_analyser.liveness_analysis()
    stage_count = len(liveness_list)

    # if a LiveStage is covered by another LiveStage, we just keep the larger one.
    assert stage_count == 1

    # a variable named `relu` must exist
    # and this live var must have inplace = True
    assert liveness_list[0].all_live_vars.exists('relu')
    relu_var = liveness_list[0].all_live_vars.get('relu')
    assert relu_var.is_inplace

    # the unique vars must be fewer than the all vars since in-place ops exist
    all_live_vars = liveness_list[0].all_live_vars
    unique_live_vars = liveness_list[0].unique_live_vars
    assert len(unique_live_vars) + 1 == len(all_live_vars)


if __name__ == '__main__':
    test_liveness_analysis()
