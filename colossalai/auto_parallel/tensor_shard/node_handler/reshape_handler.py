from typing import Dict, List

import torch

from ..sharding_strategy import OperationData, OperationDataType
from .node_handler import NodeHandler
from .registry import operator_registry
from .strategy import ReshapeGenerator, StrategyGenerator

__all__ = ['ReshapeHandler']


@operator_registry.register(torch.reshape)
@operator_registry.register(torch.flatten)
@operator_registry.register(torch.Tensor.permute)
@operator_registry.register(torch.Tensor.view)
@operator_registry.register(torch.nn.AdaptiveAvgPool2d)
class ReshapeHandler(NodeHandler):
    """
    A ReshapeHandler which deals with the sharding strategies for Reshape Op, such as torch.reshape.
    """

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(ReshapeGenerator(op_data_mapping, self.device_mesh, self.node.args[0]))
        return generators

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        # use transposed shape for strategies
        # the strategies will be transformed back to its original shape in self.post_process

        # check if the input operand is a parameter
        if isinstance(self.node.args[0]._meta_data, torch.nn.parameter.Parameter):
            data_type = OperationDataType.PARAM
        else:
            data_type = OperationDataType.ARG

        physical_input_operand = OperationData(name=str(self.node.args[0]),
                                               type=data_type,
                                               data=self.node.args[0]._meta_data)
        physical_output = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=self.node._meta_data)

        mapping = {"input": physical_input_operand, "output": physical_output}

        return mapping
