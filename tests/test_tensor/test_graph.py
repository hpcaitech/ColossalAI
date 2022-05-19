import pytest
from torch import nn
import torch
from colossalai.tensor import ColoTensor
from colossalai.tensor.graph import GraphContext
import gc


class SimpleNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.proj1 = nn.Linear(4, 8)
        self.proj2 = nn.Linear(8, 4)
        self.proj3 = nn.Linear(4, 4)
        self.proj4 = nn.Linear(4, 4)

    def forward(self, x):
        x = self.proj1(x)
        x = self.proj2(x)
        x = self.proj3(x)
        x = self.proj4(x)
        return x


def _visit_graph(start_node):
    if start_node is None:
        return

    start_node.print()

    post_node_list = start_node.post_nodes
    for node in post_node_list:
        _visit_graph(node)


def _get_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                yield obj
        except Exception as e:
            print('A trivial exception occured: {}'.format(e))


def _count_tensors():
    cnt = 0
    for t in _get_tensors():
        cnt += 1
    return cnt


def count_tensors(use_colossal):
    model = SimpleNet()

    model.eval()
    with torch.no_grad():
        if use_colossal:
            colo_input = ColoTensor.from_torch_tensor(torch.randn(4))
            graph_ctx = GraphContext()
            with graph_ctx:
                output = model(colo_input)
            output = model(colo_input)
            ret = _count_tensors()

            _visit_graph(graph_ctx.graph_nodes[0])

            del graph_ctx
            return ret
        else:
            input_t = torch.randn(4)
            output = model(input_t)
            output = model(input_t)
            return _count_tensors()


@pytest.mark.skip
# FIXME(ver217)
def test_check_activation_tensors():
    assert count_tensors(False) == count_tensors(True)


if __name__ == "__main__":
    count_tensors(True)
