from typing import Dict, List

import torch

from ..sharding_strategy import OperationData, OperationDataType
from .node_handler import NodeHandler
from .strategy import OutputGenerator, StrategyGenerator

__all__ = ['OuputHandler']


class OuputHandler(NodeHandler):
    """
    A OuputHandler which deals with the sharding strategies for Output Node.
    """

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(OutputGenerator(op_data_mapping, self.device_mesh, self.predecessor_node))
        return generators

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        # use transposed shape for strategies
        # the strategies will be transformed back to its original shape in self.post_process
        dummy_output = torch.empty(1,).to("meta")
        physical_output = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=dummy_output)

        mapping = {"output": physical_output}
        for index, input_node in enumerate(self.predecessor_node):
            if not hasattr(input_node, "_meta_data"):
                print(input_node.name)
            physical_inputs = OperationData(name=str(input_node),
                                            type=OperationDataType.ARG,
                                            data=input_node._meta_data)
            name_key = f'input_{index}'
            mapping[name_key] = physical_inputs

        return mapping
