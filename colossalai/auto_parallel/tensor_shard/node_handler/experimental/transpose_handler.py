from typing import Dict, List

import torch

from ...sharding_strategy import OperationData, OperationDataType
from ..node_handler import NodeHandler
from ..registry import operator_registry
from ..strategy import StrategyGenerator
from .reshape_generator import TransposeGenerator

__all__ = ['TransposeHandler']


@operator_registry.register(torch.Tensor.transpose)
@operator_registry.register(torch.transpose)
class TransposeHandler(NodeHandler):
    """
    A TransposeHandler which deals with the sharding strategies for torch.permute or torch.transpose.
    """

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(TransposeGenerator(op_data_mapping, self.device_mesh, self.node.args[0]))
        return generators

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        # check if the input operand is a parameter
        if isinstance(self.node.args[0]._meta_data, torch.nn.parameter.Parameter):
            data_type = OperationDataType.PARAM
        else:
            data_type = OperationDataType.ARG

        input_data = self.node.args[0]._meta_data
        physical_input_operand = OperationData(name=str(self.node.args[0]), type=data_type, data=input_data)

        transpose_dims = []
        # torch.transpose (input, dim0, dim1)
        for arg in self.node.args:
            if isinstance(arg, torch.fx.Node):
                if isinstance(arg._meta_data, int):
                    transpose_dims.append(arg._meta_data)
            else:
                transpose_dims.append(arg)

        num_dims = self.node._meta_data.dim()
        for i in range(2):
            # recover negative value to positive
            if transpose_dims[i] < 0:
                transpose_dims[i] += num_dims

        physical_shape_operand = OperationData(name='transpose_dims',
                                               type=OperationDataType.ARG,
                                               data=list(transpose_dims))

        output_data = self.node._meta_data
        physical_output_operand = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=output_data)

        mapping = {
            "input": physical_input_operand,
            "transpose_dims": physical_shape_operand,
            "output": physical_output_operand
        }

        return mapping
