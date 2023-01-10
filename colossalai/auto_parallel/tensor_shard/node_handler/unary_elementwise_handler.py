from typing import Dict, List

import torch

from ..sharding_strategy import OperationData, OperationDataType
from .node_handler import MetaInfoNodeHandler, NodeHandler
from .registry import operator_registry
from .strategy import StrategyGenerator, UnaryElementwiseGenerator

__all__ = ['UnaryElementwiseHandler']


@operator_registry.register(torch.Tensor.to)
@operator_registry.register(torch.Tensor.type)
@operator_registry.register(torch.abs)
@operator_registry.register(torch.nn.ReLU)
@operator_registry.register(torch.nn.Tanh)
@operator_registry.register(torch.tanh)
@operator_registry.register(torch.nn.modules.dropout.Dropout)
@operator_registry.register(torch.Tensor.contiguous)
@operator_registry.register(torch.nn.functional.dropout)
class UnaryElementwiseHandler(MetaInfoNodeHandler):
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
