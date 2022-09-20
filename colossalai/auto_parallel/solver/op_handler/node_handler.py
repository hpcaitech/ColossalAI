from abc import ABC, abstractmethod
from torch.fx.node import Node
from colossalai.device.device_mesh import DeviceMesh
from typing import Dict, List
from ..sharding_strategy import StrategiesVector, Operand, StrategyGenerator_V2


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
        self.strategy_generator = self.register_strategy_generator()

    def register_strategy(self) -> StrategiesVector:
        """
        Register different sharding strategies for the current node.
        """
        operand_mapping = self.get_operand_mapping()
        for generator in self.strategy_generator:
            strategies = generator.generate(operand_mapping)
            self.strategies_vector.extend(strategies)
        return self.strategies_vector

    @abstractmethod
    def register_strategy_generator(self) -> List[StrategyGenerator_V2]:
        """
        Define which generators should be used by this NodeHandler object.
        """
        pass

    @abstractmethod
    def get_operand_mapping(self) -> Dict[str, Operand]:
        """
        Returns the mapping between the logical operand name to its physical operands.
        A logical operand is defined by the strategy generator, for example, a matrix multiplication 
        operation has two operands "input" and "other". For a nn.Linear module, the physical operand for "input" is
        the module input and the physical operand for "other" is the module weight.
        Note that the operand name is specified by the StrategyGenerator object.

        For example:

            # for a linear layer
            mapping = {
                "input": Operand(name=str(self.node.args[0]), type=OperandType.ARG),
                "other": Operand(name="weight", type=OperandType.PARAM),
                "bias": Operand(name="bias", type=OperandType.PARAM)
            }
        """
        pass
