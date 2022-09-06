from typing import List
from torch.fx import GraphModule, Node


def linearize(gm: GraphModule, cnode: List[str] = None) -> List[List[Node]]:
    """Linearizing the graph

    Args:
        gm (GraphModule): GraphModule derived by tracing
        cnode (List[str], optional): common node List, should be the subset of input. Default to None.

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

    # make sure that item in cnode is valid
    if cnode:
        for name in cnode:
            try:
                assert next(node for node in gm.graph.nodes if node.name == name).op == "placeholder", \
                f"common node {name} is not an input of the model"
            except StopIteration:
                raise ValueError(f"common node name {name} not in graph")

    else:
        cnode = []

    deps = {}
    linearized_nodes = []
    region = []

    for n in gm.graph.nodes:
        if n.op != "placeholder" and n.op != "output":
            for n_par in n._input_nodes:
                if n_par.op != "placeholder" and n_par.name not in cnode:
                    deps[n_par] -= 1
            region.append(n)

            # if the node could free all dependencies in graph
            # we could begin a new node
            if _is_sink():
                linearized_nodes.append(region)
                region = []

            # propagate common node attr if possible
            if len(n._input_nodes) == len([node for node in n._input_nodes if node.name in cnode]):
                cnode.append(n.name)
            else:
                deps[n] = len([user for user in n.users if user.op != "output"])

    return linearized_nodes
