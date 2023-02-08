from typing import Dict, List

import torch

from ..sharding_strategy import OperationData, OperationDataType
from .node_handler import MetaInfoNodeHandler, NodeHandler
from .registry import operator_registry
from .strategy import DefaultReshapeGenerator, StrategyGenerator

__all__ = ['DefaultReshapeHandler']


@operator_registry.register(torch.flatten)
@operator_registry.register(torch.Tensor.unsqueeze)
@operator_registry.register(torch.nn.AdaptiveAvgPool2d)
class DefaultReshapeHandler(MetaInfoNodeHandler):
    """
    A DefaultReshapeHandler which deals with the sharding strategies for Reshape Op, such as torch.reshape.
    """

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(DefaultReshapeGenerator(op_data_mapping, self.device_mesh, self.node.args[0]))
        return generators

    def infer_logical_shape(self, data):
        """
        This function is used to infer logical shape for operands.

        Notes: This function is only used for the operands whose data are not only in type of tensor,
                such as tuple of tensor.
        """
        if isinstance(data, torch.Tensor):
            return data.shape
        else:
            assert isinstance(data, tuple), "input_data should be a tuple of tensor or a tensor."
            logical_shape = []
            for tensor in data:
                assert isinstance(tensor, torch.Tensor), "input_data should be a tuple of tensor or a tensor."
                logical_shape.append(tensor.shape)
            logical_shape = tuple(logical_shape)
            return logical_shape

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        # use transposed shape for strategies
        # the strategies will be transformed back to its original shape in self.post_process

        # check if the input operand is a parameter
        if isinstance(self.node.args[0]._meta_data, torch.nn.parameter.Parameter):
            data_type = OperationDataType.PARAM
        else:
            data_type = OperationDataType.ARG

        input_data = self.node.args[0]._meta_data
        input_logical_shape = self.infer_logical_shape(input_data)
        physical_input_operand = OperationData(name=str(self.node.args[0]),
                                               type=data_type,
                                               data=input_data,
                                               logical_shape=input_logical_shape)

        output_data = self.node._meta_data
        output_logical_shape = self.infer_logical_shape(output_data)
        physical_output = OperationData(name=str(self.node),
                                        type=OperationDataType.OUTPUT,
                                        data=output_data,
                                        logical_shape=output_logical_shape)

        mapping = {"input": physical_input_operand, "output": physical_output}

        return mapping
