from typing import Dict, List

import torch

from colossalai.device.device_mesh import DeviceMesh

from ..sharding_strategy import OperationData, OperationDataType, StrategiesVector
from .node_handler import NodeHandler
from .strategy import OutputGenerator, StrategyGenerator

__all__ = ['OutputHandler']


class OutputHandler(NodeHandler):
    """
    A OutputHandler which deals with the sharding strategies for Output Node.
    """

    def __init__(self, node: torch.fx.node.Node, device_mesh: DeviceMesh, strategies_vector: StrategiesVector,
                 output_option: str) -> None:
        super().__init__(node, device_mesh, strategies_vector)
        self.output_option = output_option

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(OutputGenerator(op_data_mapping, self.device_mesh, self.predecessor_node, self.output_option))
        return generators

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        # use transposed shape for strategies
        # the strategies will be transformed back to its original shape in self.post_process
        mapping = {}
        output_meta_data = []
        for index, input_node in enumerate(self.predecessor_node):
            input_meta_data = input_node._meta_data
            physical_inputs = OperationData(name=str(input_node), type=OperationDataType.ARG, data=input_meta_data)
            name_key = f'input_{index}'
            mapping[name_key] = physical_inputs
            output_meta_data.append(input_meta_data)

        assert len(output_meta_data) > 0, f'Output node {self.node} has no input node.'
        if len(output_meta_data) == 1:
            output_meta_data = output_meta_data[0]
        else:
            output_meta_data = tuple(output_meta_data)

        self.node._meta_data = output_meta_data
        physical_output = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=self.node._meta_data)

        mapping["output"] = physical_output
        return mapping
