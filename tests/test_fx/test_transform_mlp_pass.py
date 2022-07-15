import torch
import torch.nn as nn
import pytest
import colossalai
from colossalai.fx import ColoTracer
from colossalai.fx.passes.shard_1d_pass import transform_mlp_pass
CONFIG = dict(parallel=dict(tensor=dict(size=2, mode='1d')))

class MLP(torch.nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim)
        self.linear2 = torch.nn.Linear(dim, dim)
        self.linear3 = torch.nn.Linear(dim, dim)
        self.linear4 = torch.nn.Linear(dim, dim)
        self.dropout = torch.nn.Dropout()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(self.relu(self.linear2(x)))
        x = self.linear3(x)
        x = torch.nn.functional.relu(self.linear4(x))
        return x

def test_out_acc():
    model = MLP(16).cuda()
    model.eval()
    input_tensor = torch.rand(2, 16).cuda()
    output = model(input_tensor)
    tracer = ColoTracer()
    graph = tracer.trace(model, meta_args={'x': torch.randn((2, 16), device="meta")})
    gm = torch.fx.GraphModule(model, graph, model.__class__.__name__)
    splitted_gm = transform_mlp_pass(gm)
    new_output = splitted_gm(input_tensor)
    assert output.equal(new_output)

def test_linear_acc():
    input_tensor = torch.rand(2, 16).cuda()
    model = MLP(16).cuda()
    tracer = ColoTracer()
    graph = tracer.trace(model, meta_args={'x': torch.randn((2, 16), device="meta")})
    gm = torch.fx.GraphModule(model, graph, model.__class__.__name__)
    splitted_gm = transform_mlp_pass(gm)
    col_shard = True
    for node in splitted_gm.graph.nodes:
        if node.op == "call_module" and isinstance(node.graph.owning_module.get_submodule(node.target), torch.nn.Linear):
            target_module = node.graph.owning_module.get_submodule(node.target)
            dim = 0 if col_shard else -1
            assert target_module.weight.fx_attr == (dim, "SHARD", "TP", "col_needs_many_outputs")
            col_shard = not col_shard

if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # colossalai.launch_from_torch(config=CONFIG)
    test_out_acc()
    test_linear_acc()
