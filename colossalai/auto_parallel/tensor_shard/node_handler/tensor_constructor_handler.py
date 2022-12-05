from typing import Dict, List

import torch

from ..sharding_strategy import OperationData, OperationDataType
from .node_handler import NodeHandler
from .registry import operator_registry
from .strategy import StrategyGenerator
from .strategy.tensor_constructor_generator import TensorConstructorGenerator

__all__ = ['TensorConstructorHandler']


@operator_registry.register(torch.arange)
class TensorConstructorHandler(NodeHandler):
    """
    A TensorConstructorHandler which deals with the sharding strategies for tensor constructor operations, such as torch.arange.
    """

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(TensorConstructorGenerator(op_data_mapping, self.device_mesh))
        return generators

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        output_data = self.node._meta_data
        physical_output_operand = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=output_data)

        mapping = {"output": physical_output_operand}

        return mapping
