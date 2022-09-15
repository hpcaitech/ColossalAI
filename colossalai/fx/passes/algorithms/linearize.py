from typing import List, Any
from torch.fx import GraphModule, Node
from colossalai.fx.profiler import is_inplace

# Common nodes are type of nodes that could be seen as attributes and remain
# unchanged throughout the whole model, it will be used several times by
# different blocks of model, so that it is hard for us to linearize the graph
# when we encounter those kinds of nodes. We let users to annotate some of the
# input as common node, such as attention mask, and the followings are some of
# the ops that could actually be seen as common nodes. With our common node prop,
# we could find some of the "real" common nodes (e.g. the real attention mask
# used in BERT and GPT), the rule is simple, for node who's parents are all common
# nodes or it's op belongs to the following operations, we view this node as a
# newly born common node.
# List of target name that could be seen as common node
COPS = ["getattr", "getitem", "size"]


def _is_cop(target: Any) -> bool:
    """Check if an op could be seen as common node

    Args:
        target (Any): node target

    Returns:
        bool
    """

    if isinstance(target, str):
        return target in COPS
    else:
        return target.__name__ in COPS


def linearize(gm: GraphModule, cnode: List[str] = None) -> List[List[Node]]:
    """Linearizing the graph

    Args:
        gm (GraphModule): GraphModule derived by tracing
        cnode (List[str], optional): common node List, should be the subset of input. Default to None.

    Returns:
        List[List[Node]]: List of list, each inside list of Node presents
        the actual 'node' in linearized manner.

    Remarks:
        We merge the inplace ops into the previous node.
    """

    def _is_sink() -> bool:
        """Check if we can free all dependencies

        Returns:
            bool
        """

        return not sum([v for _, v in deps.items()]) and not any(map(is_inplace, n.users))

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
            if len(n._input_nodes) == len([node for node in n._input_nodes if node.name in cnode]) or _is_cop(n.target):
                cnode.append(n.name)
            else:
                deps[n] = len([user for user in n.users if user.op != "output"])

    return linearized_nodes
