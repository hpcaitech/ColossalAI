import operator
from typing import Dict, List

import torch

from ..sharding_strategy import OperationData, OperationDataType
from .node_handler import NodeHandler
from .registry import operator_registry
from .strategy import StrategyGenerator, TensorStrategyGenerator, TensorTupleStrategyGenerator

__all__ = ["GetItemHandler"]


@operator_registry.register(operator.getitem)
class GetItemHandler(NodeHandler):
    """
    A GetItemHandler which deals with the sharding strategies for operator.getitem.
    """

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        if isinstance(op_data_mapping["input"].data, torch.Tensor):
            generators.append(TensorStrategyGenerator(op_data_mapping, self.device_mesh, self.node.args[0]))
        else:
            generators.append(TensorTupleStrategyGenerator(op_data_mapping, self.device_mesh, self.node.args[0]))

        return generators

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        # use transposed shape for strategies
        # the strategies will be transformed back to its original shape in self.post_process
        physical_input_operand = OperationData(
            name=str(self.node.args[0]), type=OperationDataType.ARG, data=self.node.args[0]._meta_data
        )
        physical_other_operand = OperationData(name="index", type=OperationDataType.ARG, data=self.node.args[1])
        physical_output = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=self.node._meta_data)

        mapping = {"input": physical_input_operand, "index": physical_other_operand, "output": physical_output}

        return mapping
