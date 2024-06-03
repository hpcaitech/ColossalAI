import torch
from torch.fx import symbolic_trace

from colossalai.fx.passes.adding_split_node_pass import (
    balanced_split_pass,
    balanced_split_pass_v2,
    split_with_split_nodes_pass,
    uniform_split_pass,
)
from colossalai.testing import clear_cache_before_run

MODEL_DIM = 16
BATCH_SIZE = 8
PIPELINE_SIZE = 2


class MLP(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim)
        self.linear2 = torch.nn.Linear(dim, dim)
        self.linear3 = torch.nn.Linear(dim, dim)
        self.linear4 = torch.nn.Linear(dim, dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x


def pipeline_pass_test_helper(model, data, pass_func):
    origin_output = model(data)
    symbolic_traced = symbolic_trace(model)
    annotated_model = pass_func(symbolic_traced, PIPELINE_SIZE)
    split_model, split_submodules = split_with_split_nodes_pass(annotated_model)
    output = split_model(data)
    assert output.equal(origin_output)


@clear_cache_before_run()
def test_pipeline_passes():
    model = MLP(MODEL_DIM)
    data = torch.rand(BATCH_SIZE, MODEL_DIM)
    pipeline_pass_test_helper(model, data, balanced_split_pass)
    pipeline_pass_test_helper(model, data, balanced_split_pass_v2)
    pipeline_pass_test_helper(model, data, uniform_split_pass)


if __name__ == "__main__":
    test_pipeline_passes()
