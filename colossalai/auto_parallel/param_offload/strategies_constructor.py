from torch.fx import Graph, Node

from colossalai.auto_parallel.param_offload.offload_strategy import OffloadStrategiesVector
from colossalai.auto_parallel.param_offload.strategy_generator import StrategyGenerator
from colossalai.auto_parallel.param_offload.util import ModelParameters, NodeInfo

class OffloadStrategiesConstructor:
    """
    OffloadStrategiesConstructor is used to construct the offload plan for the model execution.

    Args:
        graph (Graph): a Graph object used for analysis and strategy generation.
        solver_option (SolverOption): a SolverOptions object which specifies the preferences for plan searching.
    """

    def __init__(self, graph: Graph, solver_option = None):
        self.graph = graph
        assert graph.owning_module is not None, 'The given graph is not associated with a owning_module'
        self.root_module = self.graph.owning_module
        self.nodes = list(graph.nodes)
        self.leaf_strategies = []
        self.solver_option = solver_option
        self.no_strategy_nodes = []

    def build_strategies_and_cost(self):
        """
        This method is to build the strategy vector for each node in the computation graph.
        """

        def _check_no_strategy_for_node(node: Node):
            label = False
            if node.op in ('placeholder', 'get_attr', 'call_method', 'output'):
                label = True

            elif node.op == "call_module":
                target = node.target
                submod = self.root_module.get_submodule(target)
                if (
                        len(list(submod.named_parameters(recurse=False))) == 0
                        and len(list(submod.named_buffers(recurse=False))) == 0
                ):
                    label = True

            elif node.op == "call_function":
                label = True
                for inp_node in list(node._input_nodes.keys()):
                    if inp_node.op == "get_attr":
                        label = False
                        break

            return label

        def _set_params_info_for_node(node: Node):

            assert node.op in ['call_function', 'call_module']
            assert hasattr(node, "node_info") and isinstance(node.node_info, NodeInfo)

            node_info = node.node_info
            node_info.has_param = True
            if node_info.param_indices is None:
                node_info.param_indices = []

            if node.op == 'call_module':
                target = node.target
                submod = self.root_module.get_submodule(target)
                for p in list(submod.parameters(recurse=False)):
                    node_info.param_indices.append(ModelParameters.param_idx)
                    node_info.param_size += p.data.numel() * p.data.element_size()
                    ModelParameters.fp16_params.append(p)
                    ModelParameters.fp32_master_params.append(p.detach().clone().float())
                    ModelParameters.param_idx += 1

            elif node.op == 'call_function':
                for inp_node in list(node._input_nodes.keys()):
                    if inp_node.op == "get_attr":
                        attr_itr = self.root_module
                        atoms = inp_node.target.split(".")
                        for atom in atoms:
                            attr_itr = getattr(attr_itr, atom)

                        node_info.param_indices.append(ModelParameters.param_idx)
                        node_info.param_size += attr_itr.data.numel() * attr_itr.data.element_size()
                        ModelParameters.fp16_params.append(attr_itr)
                        ModelParameters.fp32_master_params.append(attr_itr.detach().clone().float())
                        ModelParameters.param_idx += 1

        for node in self.nodes:
            setattr(node, "node_info", NodeInfo())
            strategies_vector = OffloadStrategiesVector(node)

            if _check_no_strategy_for_node(node):
                self.no_strategy_nodes.append(node)
                continue

            _set_params_info_for_node(node)
            generator = StrategyGenerator(node, self.graph)
            strategies_vector.extend(generator.generate())
            setattr(node, 'strategies_vector', strategies_vector)
            self.leaf_strategies.append(strategies_vector)
