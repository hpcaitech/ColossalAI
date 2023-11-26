import copy
from typing import Dict, List

import torch

from ..sharding_strategy import OperationData, OperationDataType, ShardingStrategy
from ..utils import recover_sharding_spec_for_broadcast_shape
from .node_handler import NodeHandler
from .registry import operator_registry
from .strategy import StrategyGenerator, WhereGenerator

__all__ = ["WhereHandler"]


@operator_registry.register(torch.where)
class WhereHandler(NodeHandler):
    """
    A WhereHandler which deals with the sharding strategies for torch.where.
    """

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        logical_op_data_mapping, _ = self.get_operation_data_mapping()
        generators = []
        generators.append(WhereGenerator(logical_op_data_mapping, self.device_mesh))
        return generators

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        # use transposed shape for strategies
        # the strategies will be transformed back to its original shape in self.post_process
        physical_condition_operand = OperationData(
            name=str(self.node.args[0]), type=OperationDataType.ARG, data=self.node.args[0]._meta_data
        )
        physical_x_operand = OperationData(
            name=str(self.node.args[1]), type=OperationDataType.ARG, data=self.node.args[1]._meta_data
        )
        physical_y_operand = OperationData(
            name=str(self.node.args[2]), type=OperationDataType.ARG, data=self.node.args[2]._meta_data
        )
        physical_output = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=self.node._meta_data)
        physical_mapping = {
            "condition": physical_condition_operand,
            "x": physical_x_operand,
            "y": physical_y_operand,
            "output": physical_output,
        }
        logical_shape_for_all = self.node._meta_data.shape
        logical_mapping = {}
        for key, physical_operand in physical_mapping.items():
            logical_mapping[key] = self.convert_physical_operand_to_logical_operand(
                physical_operand, logical_shape_for_all
            )

        return logical_mapping, physical_mapping

    def convert_physical_operand_to_logical_operand(self, physical_operand, target_shape):
        logical_operand = copy.deepcopy(physical_operand)
        logical_operand.logical_shape = target_shape
        return logical_operand

    def post_process(self, strategy: ShardingStrategy):
        logical_op_data_mapping, physical_op_data_mapping = self.get_operation_data_mapping()
        for key in logical_op_data_mapping.keys():
            logical_sharding_spec = strategy.sharding_specs[logical_op_data_mapping[key]]
            logical_shape = logical_op_data_mapping[key].logical_shape
            physical_shape = physical_op_data_mapping[key].logical_shape
            physical_sharding_spec, removed_dims = recover_sharding_spec_for_broadcast_shape(
                logical_sharding_spec, logical_shape, physical_shape
            )
            strategy.sharding_specs.pop(logical_op_data_mapping[key])
            strategy.sharding_specs[physical_op_data_mapping[key]] = physical_sharding_spec
        strategy.name = f"{strategy.sharding_specs[physical_op_data_mapping['output']].sharding_sequence} = {strategy.sharding_specs[physical_op_data_mapping['condition']].sharding_sequence} x {strategy.sharding_specs[physical_op_data_mapping['x']].sharding_sequence} x {strategy.sharding_specs[physical_op_data_mapping['y']].sharding_sequence}"
        return strategy
