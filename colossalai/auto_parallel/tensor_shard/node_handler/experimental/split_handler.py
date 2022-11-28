from typing import Dict, List

import torch

from ...sharding_strategy import OperationData, OperationDataType
from ..node_handler import NodeHandler
from ..registry import operator_registry
from ..strategy import StrategyGenerator
from .reshape_generator import SplitGenerator

__all__ = ['SplitHandler']


@operator_registry.register(torch.Tensor.split)
@operator_registry.register(torch.split)
class SplitHandler(NodeHandler):
    """
    A SplitHandler which deals with the sharding strategies for torch.permute or torch.split.
    """

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(SplitGenerator(op_data_mapping, self.device_mesh, self.node.args[0]))
        return generators

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        # check if the input operand is a parameter
        if isinstance(self.node.args[0]._meta_data, torch.nn.parameter.Parameter):
            data_type = OperationDataType.PARAM
        else:
            data_type = OperationDataType.ARG

        input_data = self.node.args[0]._meta_data
        physical_input_operand = OperationData(name=str(self.node.args[0]), type=data_type, data=input_data)
        split_size = self.node.args[1]
        if len(self.node.args) == 3:
            # (input, split_size, split_dim)
            split_dim = self.node.args[2]
        else:
            if self.node.kwargs:
                split_dim = self.node.kwargs['dim']
            else:
                split_dim = 0

        num_dims = self.node.args[0]._meta_data.dim()
        # recover negative value to positive
        if split_dim < 0:
            split_dim += num_dims

        split_info = (split_size, split_dim)
        physical_shape_operand = OperationData(name='split_info', type=OperationDataType.ARG, data=split_info)

        output_data = self.node._meta_data
        physical_output_operand = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=output_data)

        mapping = {
            "input": physical_input_operand,
            "split_info": physical_shape_operand,
            "output": physical_output_operand
        }

        return mapping
