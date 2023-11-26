import pytest
import torch
import torch.nn as nn

from colossalai._analyzer.fx.graph_module import ColoGraphModule
from colossalai._analyzer.fx.passes import shape_prop_pass
from colossalai._analyzer.fx.tracer.tracer import ColoTracer
from colossalai.auto_parallel.tensor_shard.solver import GraphAnalyser
from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.testing import clear_cache_before_run


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


@pytest.mark.skip("meta tensor has some bugs in 1.11")
@clear_cache_before_run()
def test_liveness_analysis():
    model = LinearModel()
    tracer = ColoTracer(bias_addition_split=True)
    meta_args = {"x1": torch.rand(4, 4, device="meta"), "x2": torch.rand(4, 4, device="meta")}
    graph = tracer.trace(model, meta_args=meta_args)
    gm = ColoGraphModule(root=model, graph=graph, class_name=model.__class__.__name__)
    shape_prop_pass(gm, *meta_args.values())

    graph_analyser = GraphAnalyser(gm)
    liveness_list = graph_analyser.liveness_analysis()
    stage_count = len(liveness_list)

    # if a LiveStage is covered by another LiveStage, we just keep the larger one.
    assert stage_count == 1

    # a variable named `relu` must exist
    # and this live var must have inplace = True
    assert liveness_list[0].all_live_vars.exists("relu")
    relu_var = liveness_list[0].all_live_vars.get("relu")
    assert relu_var.is_inplace

    # the unique vars must be fewer than the all vars since in-place ops exist
    all_live_vars = liveness_list[0].all_live_vars
    unique_live_vars = liveness_list[0].unique_live_vars
    assert len(unique_live_vars) + 1 == len(all_live_vars)


if __name__ == "__main__":
    test_liveness_analysis()
