import torch

from colossalai.fx import ColoTracer
from colossalai.fx.passes.utils import assign_bfs_level_to_nodes, get_leaf, get_top
from colossalai.testing import clear_cache_before_run


class MLP(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim)
        self.linear2 = torch.nn.Linear(dim, dim)
        self.linear3 = torch.nn.Linear(dim, dim)
        self.linear4 = torch.nn.Linear(dim, dim)
        self.linear5 = torch.nn.Linear(dim, dim)

    def forward(self, x):
        l1 = self.linear1(x)
        l2 = self.linear2(x)
        l3 = self.linear3(l1)
        l4 = self.linear4(l2)
        l5 = self.linear5(l3)
        return l4, l5


@clear_cache_before_run()
def test_graph_manipulation():
    model = MLP(4)
    tracer = ColoTracer()
    graph = tracer.trace(model)
    nodes = list(graph.nodes)
    x, l1, l2, l3, l4, l5, output = nodes

    leaf_nodes = set(get_leaf(graph))
    top_nodes = set(get_top(graph))
    compare_dict = {x: None, l1: 0, l2: 0, l3: 1, l4: 1, l5: 2, output: None}
    assign_bfs_level_to_nodes(graph)

    assert leaf_nodes == set([l4, l5])
    assert top_nodes == set([l1, l2])
    for node in graph.nodes:
        if node.op in ("placeholder", "output"):
            assert not hasattr(node, "bfs_level")
        else:
            assert node.bfs_level == compare_dict[node]


if __name__ == "__main__":
    test_graph_manipulation()
