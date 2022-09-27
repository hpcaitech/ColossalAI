from abc import ABC, abstractmethod
from torch.fx.node import Node
from colossalai.device.device_mesh import DeviceMesh
from typing import Dict, List
from ..sharding_strategy import ShardingStrategy_V2, StrategiesVector, OperationData
from ..strategy import StrategyGenerator_V2


class NodeHandler(ABC):
    '''
    The NodeHandler is an abstract class used to generate every possible strategies for an operator node.

    Args:
        node (Node): the input node in node argument list.
        device_mesh (DeviceMesh): A logical view of a physical mesh.
        strategies_vector (StrategiesVector): all the strategies generated in this handler will be recorded into the strategies_vector.
    '''

    def __init__(
        self,
        node: Node,
        device_mesh: DeviceMesh,
        strategies_vector: StrategiesVector,
    ) -> None:
        self.node = node
        self.predecessor_node = list(node._input_nodes.keys())
        self.successor_node = list(node.users.keys())
        self.device_mesh = device_mesh
        self.strategies_vector = strategies_vector

    def register_strategy(self) -> StrategiesVector:
        """
        Register different sharding strategies for the current node.
        """
        strategy_generators = self.get_strategy_generator()
        for generator in strategy_generators:
            strategies = generator.generate()
            self.strategies_vector.extend(strategies)

        strategies_vector = map(self.post_process, self.strategies_vector)
        self.strategies_vector = list(strategies_vector)
        return self.strategies_vector

    def post_process(self, strategy: ShardingStrategy_V2):
        # tranform the strategy generated
        # e.g. to process the sharding strategy for the transposed weights
        return strategy

    @abstractmethod
    def get_strategy_generator(self) -> List[StrategyGenerator_V2]:
        """
        Define which generators should be used by this NodeHandler object.
        """
        pass

    @abstractmethod
    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        """
        Returns the mapping between the logical operation data to its physical data.
        A logical operation data is a data associated with an operation, which can be input and output. It is 
        defined by the strategy generator, for example, a matrix multiplication operation has two operands "input" 
        and "other" and one result "output". For a nn.Linear module, the physical operand for "input" is
        the module input, the physical operand for "other" is the module weight, and the physical result for "output"
        is the module output.
        Note that the operand name is specified by the StrategyGenerator object.

        For example:

            # for a linear layer
            mapping = {
                "input": Operand(name=str(self.node.args[0]), type=OperationDataType.ARG, data=self.node.args[0]._meta_data),
                "other": Operand(name="weight", type=OperationDataType.PARAM, data=self.named_parameters['weight']),
                "bias": Operand(name="bias", type=OperationDataType.PARAM, data=self.named_parameters['bias']),
                "output": Operand(name=str(self.node), type=OperationDataType.OUTPUT, data=self.node._meta_data),
            }
        """
        pass


class ModuleHandler(NodeHandler):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        print("created")

        # set attributes to access module parameters for convenience
        assert self.node.graph.owning_module is not None, \
            f'The graph is not associated with a module, please make sure it can be used to instantiate a GraphModule object.'
        module = self.node.graph.owning_module.get_submodule(self.node.target)
        named_parameters = list(module.named_parameters(recurse=False))
        # convert named parameters from list to dict
        named_parameters = {k: v for k, v in named_parameters}
        self.module = module
        self.named_parameters = named_parameters
