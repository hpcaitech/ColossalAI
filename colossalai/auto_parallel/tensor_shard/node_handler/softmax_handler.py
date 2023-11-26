from typing import Dict, List

import torch

from ..sharding_strategy import OperationData, OperationDataType
from .node_handler import NodeHandler
from .registry import operator_registry
from .strategy import SoftmaxGenerator, StrategyGenerator

__all__ = ["SoftmaxHandler"]


@operator_registry.register(torch.nn.Softmax)
@operator_registry.register(torch.nn.functional.softmax)
class SoftmaxHandler(NodeHandler):
    """
    A SoftmaxHandler which deals with the sharding strategies for
    torch.nn.Softmax or torch.nn.functional.softmax.
    """

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(SoftmaxGenerator(op_data_mapping, self.device_mesh, self.node.args[0]))
        return generators

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        # check if the input operand is a parameter
        if isinstance(self.node.args[0]._meta_data, torch.nn.parameter.Parameter):
            data_type = OperationDataType.PARAM
        else:
            data_type = OperationDataType.ARG

        input_data = self.node.args[0]._meta_data
        physical_input_operand = OperationData(name=str(self.node.args[0]), type=data_type, data=input_data)

        softmax_dim = self.node.kwargs["dim"]

        num_dims = self.node.args[0]._meta_data.dim()
        # recover negative value to positive
        if softmax_dim < 0:
            softmax_dim += num_dims

        physical_dim_operand = OperationData(name="softmax_dim", type=OperationDataType.ARG, data=softmax_dim)

        output_data = self.node._meta_data
        physical_output_operand = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=output_data)

        mapping = {
            "input": physical_input_operand,
            "softmax_dim": physical_dim_operand,
            "output": physical_output_operand,
        }

        return mapping
