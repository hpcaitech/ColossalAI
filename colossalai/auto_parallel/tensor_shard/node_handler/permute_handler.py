from typing import Dict, List

import torch

from ..sharding_strategy import OperationData, OperationDataType
from .node_handler import NodeHandler
from .registry import operator_registry
from .strategy import PermuteGenerator, StrategyGenerator

__all__ = ['PermuteHandler']


@operator_registry.register(torch.Tensor.permute)
@operator_registry.register(torch.permute)
class PermuteHandler(NodeHandler):
    """
    A PermuteHandler which deals with the sharding strategies for torch.permute or torch.transpose.
    """

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(PermuteGenerator(op_data_mapping, self.device_mesh, self.node.args[0]))
        return generators

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        # check if the input operand is a parameter
        if isinstance(self.node.args[0]._meta_data, torch.nn.parameter.Parameter):
            data_type = OperationDataType.PARAM
        else:
            data_type = OperationDataType.ARG

        input_data = self.node.args[0]._meta_data
        physical_input_operand = OperationData(name=str(self.node.args[0]), type=data_type, data=input_data)

        permute_dims = []
        if self.node.op == 'call_method':
            # torch.Tensor.permute (input, *dims)
            for arg in self.node.args:
                if isinstance(arg, torch.fx.Node):
                    if isinstance(arg._meta_data, int):
                        permute_dims.append(arg._meta_data)
                else:
                    assert isinstance(arg, int), 'The argument in permute node should be either type of Node or int.'
                    permute_dims.append(arg)
        else:
            # torch.permute (input, dims)
            for arg in self.node.args:
                if isinstance(arg, torch.fx.Node):
                    if isinstance(arg._meta_data, (tuple, list)):
                        permute_dims.extend(arg._meta_data)
                else:
                    assert isinstance(
                        arg,
                        (tuple, list)), 'The argument in permute node should be type of Node, Tuple[int] or List[int].'
                    permute_dims.extend(arg)

        num_dims = self.node._meta_data.dim()
        for i in range(num_dims):
            # recover negative value to positive
            if permute_dims[i] < 0:
                permute_dims[i] += num_dims

        physical_shape_operand = OperationData(name='permute_dims', type=OperationDataType.ARG, data=list(permute_dims))

        output_data = self.node._meta_data
        physical_output_operand = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=output_data)

        mapping = {
            "input": physical_input_operand,
            "permute_dims": physical_shape_operand,
            "output": physical_output_operand
        }

        return mapping
