from colossalai.fx.proxy import ColoProxy
import torch
from tests.components_to_test.registry import non_distributed_component_funcs


def test_coloproxy():
    model_builder, *_ = non_distributed_component_funcs.get_callable('simple_net')()
    model = model_builder()

    gm = torch.fx.symbolic_trace(model)
    node = list(gm.graph.nodes)[0]
    proxy = ColoProxy(node=node)
    proxy.meta_tensor = torch.empty(4, 2, device='meta')

    assert len(proxy) == 4
    assert proxy.shape[0] == 4 and proxy.shape[1] == 2
    assert proxy.dim() == 2
    assert proxy.dtype == torch.float32
    assert proxy.size(0) == 4


if __name__ == '__main__':
    test_coloproxy()