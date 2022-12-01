from typing import Dict, List

import torch

from ...sharding_strategy import OperationData, OperationDataType
from ..node_handler import NodeHandler
from ..registry import operator_registry
from ..strategy import StrategyGenerator
from .reshape_generator import ViewGenerator

__all__ = ['ViewHandler']


@operator_registry.register(torch.Tensor.reshape)
@operator_registry.register(torch.reshape)
@operator_registry.register(torch.Tensor.view)
class ViewHandler(NodeHandler):
    """
    A ViewHandler which deals with the sharding strategies for Reshape Op, such as torch.reshape.
    """

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(ViewGenerator(op_data_mapping, self.device_mesh, self.node.args[0]))
        return generators

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        # use transposed shape for strategies
        # the strategies will be transformed back to its original shape in self.post_process

        # check if the input operand is a parameter
        if isinstance(self.node.args[0]._meta_data, torch.nn.parameter.Parameter):
            data_type = OperationDataType.PARAM
        else:
            data_type = OperationDataType.ARG

        input_data = self.node.args[0]._meta_data
        physical_input_operand = OperationData(name=str(self.node.args[0]), type=data_type, data=input_data)

        target_shape = self.node._meta_data.shape
        physical_shape_operand = OperationData(name='tgt_shape', type=OperationDataType.ARG, data=target_shape)

        output_data = self.node._meta_data
        physical_output_operand = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=output_data)

        mapping = {
            "input": physical_input_operand,
            "tgt_shape": physical_shape_operand,
            "output": physical_output_operand
        }

        return mapping
