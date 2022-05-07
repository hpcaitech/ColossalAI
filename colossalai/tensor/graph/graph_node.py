from colossalai.tensor import ColoTensor
from colossalai.context.singleton_meta import SingletonMeta


class GraphGlobalEnv(metaclass=SingletonMeta):

    def __init__(self) -> None:
        self.graph_building = False


class GraphContext():
    """
    
    Building the computing graph under the context

    >>> with GraphContext():
    >>>     output = model(colo_input_tensor)
    """

    def __enter__(self):
        GraphGlobalEnv().graph_building = True

    def __exit__(self, *exc_info):
        GraphGlobalEnv().graph_building = False


class GraphNode(object):

    def __init__(self) -> None:
        self.prev_nodes = []
        self.post_nodes = []

    def add_prev_node(self, node):
        if GraphGlobalEnv().graph_building:
            self.prev_nodes.append(node)

    def add_post_node(self, node):
        if GraphGlobalEnv().graph_building:
            self.post_nodes.append(node)

    def post_node_empty(self) -> bool:
        if len(self.post_nodes) == 0:
            return True
        else:
            return False

    def print(self):
        print(f'GraphNode')


class GraphOpNode(GraphNode):

    def __init__(self, op_type, param_list) -> None:
        super().__init__()
        self._op_type = op_type
        self._param_list = param_list

    def add_prev_tensor(self, colo_tensor: ColoTensor):
        if GraphGlobalEnv().graph_building:
            assert isinstance(colo_tensor, ColoTensor)
            if colo_tensor._graph_node is None:
                colo_tensor._graph_node = GraphNode()

            self.add_prev_node(colo_tensor._graph_node)
            colo_tensor._graph_node.add_post_node(self)

    def add_post_tensor(self, colo_tensor: ColoTensor):
        if GraphGlobalEnv().graph_building:
            assert isinstance(colo_tensor, ColoTensor)
            if colo_tensor._graph_node is None:
                colo_tensor._graph_node = GraphNode()

            self.add_post_node(colo_tensor._graph_node)
            colo_tensor._graph_node.add_prev_node(self)

    def print(self):
        print(
            f'GraphOpNode {self._op_type}, post node number {len(self.post_nodes)}, prev node number {len(self.prev_nodes)}'
        )
