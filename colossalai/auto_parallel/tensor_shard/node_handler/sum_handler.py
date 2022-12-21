from typing import Dict, List

import torch

from ..sharding_strategy import OperationData, OperationDataType
from .node_handler import NodeHandler
from .registry import operator_registry
from .strategy import StrategyGenerator, SumGenerator

__all__ = ['SumHandler']


@operator_registry.register(torch.Tensor.sum)
@operator_registry.register(torch.sum)
class SumHandler(NodeHandler):
    """
    A SumHandler which deals with the sharding strategies for torch.sum or torch.Tensor.sum.
    """

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(SumGenerator(op_data_mapping, self.device_mesh, self.node.args[0]))
        return generators

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        # check if the input operand is a parameter
        if isinstance(self.node.args[0]._meta_data, torch.nn.parameter.Parameter):
            data_type = OperationDataType.PARAM
        else:
            data_type = OperationDataType.ARG

        input_data = self.node.args[0]._meta_data
        physical_input_operand = OperationData(name=str(self.node.args[0]), type=data_type, data=input_data)

        if len(self.node.args) > 1:
            sum_dims = self.node.args[1]
        else:
            sum_dims = tuple(range(self.node.args[0]._meta_data.dim()))

        if isinstance(sum_dims, int):
            sum_dims = (sum_dims,)

        # recover negative value to positive
        num_dims = self.node.args[0]._meta_data.dim()
        for i in range(len(sum_dims)):
            if sum_dims[i] < 0:
                sum_dims[i] += num_dims

        # mapping the input dims to output dims
        # For examples:
        #   input: torch.rand(2, 3, 4, 5)
        #   output: torch.sum(input, (0, 2))
        #   sum_mapping_dict = {1: 0, 3: 1}
        #   sum_mapping_dict[1] = 0 means the 0th dim of output is the 1st dim of input
        #   sum_mapping_dict[3] = 1 means the 1st dim of output is the 3rd dim of input
        sum_mapping_dict = {}
        if 'keepdim' in self.node.kwargs and self.node.kwargs['keepdim']:
            for i in range(num_dims):
                sum_mapping_dict.update({i: i})
        else:
            output_index = 0
            for i in range(num_dims):
                if i not in sum_dims:
                    sum_mapping_dict.update({i: output_index})
                    output_index += 1
            assert output_index == self.node._meta_data.dim()

        sum_info = (sum_dims, sum_mapping_dict)
        physical_shape_operand = OperationData(name='sum_info', type=OperationDataType.ARG, data=sum_info)

        output_data = self.node._meta_data
        physical_output_operand = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=output_data)

        mapping = {
            "input": physical_input_operand,
            "sum_info": physical_shape_operand,
            "output": physical_output_operand
        }

        return mapping
