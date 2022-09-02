from typing import List
from torch.fx import GraphModule, Node


def linearize(gm: GraphModule) -> List[List[Node]]:
    """Linearizing the graph

    Args:
        gm (GraphModule): GraphModule derived by tracing

    Returns:
        List[List[Node]]: List of list, each inside list of Node presents
        the actual 'node' in linearized manner.
    """

    def _is_sink() -> bool:
        """Check if we can free all dependencies

        Returns:
            bool
        """

        return not sum([v for _, v in deps.items()])

    deps = {}
    linearized_nodes = []
    region = []

    for n in gm.graph.nodes:
        for n_par in n._input_nodes:
            deps[n_par] -= 1
        region.append(n)

        # if the node could free all dependencies in graph
        # we could begin a new node
        if _is_sink():
            linearized_nodes.append(region)
            region = []

        deps[n] = len(n.users)

    # Remove input
    linearized_nodes = linearized_nodes[1:-1]
    return linearized_nodes
