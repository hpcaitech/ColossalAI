from typing import Dict, List

import torch

from ..sharding_strategy import OperationData, OperationDataType
from .node_handler import NodeHandler
from .registry import operator_registry
from .strategy import BatchedMatMulStrategyGenerator, StrategyGenerator


@operator_registry.register(torch.bmm)
@operator_registry.register(torch.Tensor.bmm)
class BMMFunctionHandler(NodeHandler):

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        physical_input_operand = OperationData(name=str(self.node.args[0]),
                                               type=OperationDataType.ARG,
                                               data=self.node.args[0]._meta_data)

        physical_other_operand = OperationData(name=str(self.node.args[1]),
                                               type=OperationDataType.ARG,
                                               data=self.node.args[1]._meta_data)
        physical_output = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=self.node._meta_data)

        mapping = {"input": physical_input_operand, "other": physical_other_operand, "output": physical_output}
        return mapping

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        generators = []
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(BatchedMatMulStrategyGenerator(op_data_mapping, self.device_mesh))
        return generators
