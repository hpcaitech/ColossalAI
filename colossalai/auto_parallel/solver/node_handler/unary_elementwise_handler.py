import torch
from .node_handler import NodeHandler
from ..sharding_strategy import ShardingStrategy, OperationDataType, OperationData, StrategiesVector
from ..strategy import UnaryElementwiseGenerator, StrategyGenerator
from typing import List, Dict
from .registry import operator_registry
import operator

__all__ = ['UnaryElementwiseHandler']


@operator_registry.register(torch.abs)
@operator_registry.register(torch.nn.ReLU)
class UnaryElementwiseHandler(NodeHandler):
    """
    A UnaryElementwiseHandler which deals with the sharding strategies for UnaryElementwise Op.
    """

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(UnaryElementwiseGenerator(op_data_mapping, self.device_mesh, self.node.args[0]))
        return generators

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        # use transposed shape for strategies
        # the strategies will be transformed back to its original shape in self.post_process
        physical_input_operand = OperationData(name=str(self.node.args[0]),
                                               type=OperationDataType.ARG,
                                               data=self.node.args[0]._meta_data)
        physical_output = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=self.node._meta_data)

        mapping = {"input": physical_input_operand, "output": physical_output}

        return mapping
