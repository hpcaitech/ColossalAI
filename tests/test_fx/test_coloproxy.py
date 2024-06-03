import torch
import torch.nn as nn
from torch.fx import GraphModule

from colossalai.fx.proxy import ColoProxy
from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.testing import clear_cache_before_run


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.shape[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


@clear_cache_before_run()
def test_coloproxy():
    tracer = ColoTracer()
    model = Conv1D(3, 3)
    input_sample = {"x": torch.rand(3, 3).to("meta")}

    graph = tracer.trace(root=model, meta_args=input_sample)
    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()
    node = list(gm.graph.nodes)[0]

    proxy = ColoProxy(node=node, tracer=tracer)
    proxy.meta_data = torch.empty(4, 2, device="meta")

    assert len(proxy) == 4
    assert proxy.shape[0] == 4 and proxy.shape[1] == 2
    assert proxy.dim() == 2
    assert proxy.dtype == torch.float32
    assert proxy.size(0) == 4


if __name__ == "__main__":
    test_coloproxy()
