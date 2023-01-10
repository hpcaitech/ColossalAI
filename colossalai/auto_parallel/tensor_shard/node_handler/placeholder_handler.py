from typing import Dict, List

from torch.fx.node import Node

from colossalai.device.device_mesh import DeviceMesh

from ..sharding_strategy import OperationData, OperationDataType, StrategiesVector
from .node_handler import NodeHandler
from .strategy import PlaceholderGenerator, StrategyGenerator

__all__ = ['PlaceholderHandler']


class PlaceholderHandler(NodeHandler):
    """
    A PlaceholderHandler which deals with the sharding strategies for Placeholder Node.
    """

    def __init__(self, node: Node, device_mesh: DeviceMesh, strategies_vector: StrategiesVector,
                 placeholder_option: str) -> None:
        super().__init__(node, device_mesh, strategies_vector)
        self.placeholder_option = placeholder_option

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(
            PlaceholderGenerator(op_data_mapping, self.device_mesh, placeholder_option=self.placeholder_option))
        return generators

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        # use transposed shape for strategies
        # the strategies will be transformed back to its original shape in self.post_process
        physical_output = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=self.node._meta_data)

        mapping = {"output": physical_output}

        return mapping
