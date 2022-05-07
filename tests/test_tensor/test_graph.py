from torch import nn
import torch
from colossalai.tensor import ColoTensor
from colossalai.tensor.graph import GraphContext


class SimpleNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.proj1 = nn.Linear(4, 8)
        # self.ln1 = nn.LayerNorm(8)
        self.proj2 = nn.Linear(8, 4)
        self.proj3 = nn.Linear(4, 4)
        self.proj4 = nn.Linear(4, 4)
        # self.ln2 = nn.LayerNorm(4)

    def forward(self, x):
        # x = self.embed(x)
        x = self.proj1(x)
        # x = self.ln1(x)
        x = self.proj2(x)
        x = self.proj3(x)
        x = self.proj4(x)
        # x = self.ln2(x)
        return x


def _visit_graph(start_node):
    if start_node is None:
        return

    start_node.print()

    post_node_list = start_node.post_nodes
    for node in post_node_list:
        _visit_graph(node)


def test_graph():
    model = SimpleNet()

    colo_input = ColoTensor.init_from_torch_tensor(torch.randn(4))
    with GraphContext():
        output = model(colo_input)
    output = model(colo_input)
    print(colo_input._graph_node)
    _visit_graph(colo_input._graph_node)
    # print(output)


if __name__ == "__main__":
    test_graph()
