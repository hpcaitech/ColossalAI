import torch
from colossalai.fx.proxy import ColoProxy
import pytest


@pytest.mark.skip
def test_coloproxy():
    # create a dummy node only for testing purpose
    model = torch.nn.Linear(10, 10)
    gm = torch.fx.symbolic_trace(model)
    node = list(gm.graph.nodes)[0]

    # create proxy
    proxy = ColoProxy(node=node)
    proxy.meta_data = torch.empty(4, 2, device='meta')

    assert len(proxy) == 4
    assert proxy.shape[0] == 4 and proxy.shape[1] == 2
    assert proxy.dim() == 2
    assert proxy.dtype == torch.float32
    assert proxy.size(0) == 4


if __name__ == '__main__':
    test_coloproxy()
