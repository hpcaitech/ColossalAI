from colossalai.tensor import ColoTensor
from colossalai.context.singleton_meta import SingletonMeta


class GraphGlobalEnv(metaclass=SingletonMeta):

    def __init__(self) -> None:
        self.graph_building = False
        self.graph_node_list = []
        self.node_id = -1

    def get_node_id(self):
        self.node_id += 1
        return self.node_id

    def add_graph_node(self, node):
        self.graph_node_list.append(node)


class GraphContext():
    """
    
    Building the computing graph under the context

    >>> with GraphContext():
    >>>     output = model(colo_input_tensor)
    """
    graph_nodes = []

    def __enter__(self):
        GraphGlobalEnv().graph_building = True
        GraphGlobalEnv().graph_node_list = []

    def __exit__(self, *exc_info):
        GraphGlobalEnv().graph_building = False
        GraphGlobalEnv().node_id = -1
        self.graph_nodes = GraphGlobalEnv().graph_node_list


class GraphNode(object):

    def __init__(self) -> None:
        self.prev_nodes = []
        self.post_nodes = []
        self.id = GraphGlobalEnv().get_node_id()

    def add_prev_node(self, node):
        if GraphGlobalEnv().graph_building:
            self.prev_nodes.append(node)

    def add_post_node(self, node):
        if GraphGlobalEnv().graph_building:
            self.post_nodes.append(node)

    def post_node_empty(self) -> bool:
        return len(self.post_nodes) == 0


class GraphOpNode(GraphNode):

    def __init__(self, op_type, param_list) -> None:
        super().__init__()
        self._op_type = op_type
        self._param_list = param_list
        GraphGlobalEnv().add_graph_node(self)

    def add_prev_tensor(self, colo_tensor: ColoTensor):
        r"""
        Link the current graph op node to previous graph op.
        Op1 <- Activation (colo_tensor) Op2
        Op1 <- Op2
        """
        if GraphGlobalEnv().graph_building:
            assert isinstance(colo_tensor, ColoTensor)
            if colo_tensor._graph_node is None:
                colo_tensor._graph_node = GraphNode()

            prev_ops = colo_tensor._graph_node.prev_nodes
            for op_node in prev_ops:
                self.add_prev_node(op_node)
                op_node.add_post_node(self)

    def add_post_tensor(self, colo_tensor: ColoTensor):
        """
        Op <- Activation (colo_tensor)
        """
        if GraphGlobalEnv().graph_building:
            assert isinstance(colo_tensor, ColoTensor)
            if colo_tensor._graph_node is None:
                colo_tensor._graph_node = GraphNode()

            colo_tensor._graph_node.add_prev_node(self)

    def print(self):
        print(
            f'GraphOpNode {self._op_type} {self.id}, post nodes {[node.id for node in self.post_nodes]}, prev node number {[node.id for node in self.prev_nodes]}'
        )
