from typing import Dict, List, Union

import torch
from torch.fx.node import Node

from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    CommAction,
    CommType,
    OperationData,
    OperationDataType,
    ShardingStrategy,
)
from colossalai.tensor.shape_consistency import CollectiveCommPattern, CommSpec, ShapeConsistencyManager

from ..constants import BCAST_FUNC_OP
from ..utils import comm_actions_for_oprands, recover_sharding_spec_for_broadcast_shape
from .node_handler import NodeHandler
from .registry import operator_registry
from .strategy import BinaryElementwiseStrategyGenerator, StrategyGenerator

__all__ = ['BinaryElementwiseHandler']


@operator_registry.register(BCAST_FUNC_OP)
class BinaryElementwiseHandler(NodeHandler):
    """
    An BinaryBcastOpHandler is a node handler which deals with operations which have two
    operands and broadcasting occurs such as torch.add.
    """

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        bcast_shape = self.node._meta_data.shape

        def _get_op_data_type(tensor):
            if isinstance(tensor, torch.nn.parameter.Parameter):
                return OperationDataType.PARAM
            else:
                return OperationDataType.ARG

        def _get_arg_value(idx):
            if isinstance(self.node.args[idx], Node):
                meta_data = self.node.args[idx]._meta_data
            else:
                # this is in fact a real data like int 1
                # but we can deem it as meta data
                # as it won't affect the strategy generation
                assert isinstance(self.node.args[idx], (int, float))
                meta_data = torch.Tensor([self.node.args[idx]]).to('meta')
            return meta_data

        input_meta_data = _get_arg_value(0)
        other_meta_data = _get_arg_value(1)
        output_meta_data = self.node._meta_data

        input_op_data = OperationData(name=str(self.node.args[0]),
                                      type=_get_op_data_type(input_meta_data),
                                      data=input_meta_data,
                                      logical_shape=bcast_shape)
        other_op_data = OperationData(name=str(self.node.args[1]),
                                      type=_get_op_data_type(other_meta_data),
                                      data=other_meta_data,
                                      logical_shape=bcast_shape)
        output_op_data = OperationData(name=str(self.node),
                                       type=OperationDataType.OUTPUT,
                                       data=output_meta_data,
                                       logical_shape=bcast_shape)

        mapping = {'input': input_op_data, 'other': other_op_data, 'output': output_op_data}
        return mapping

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(BinaryElementwiseStrategyGenerator(op_data_mapping, self.device_mesh))
        return generators

    def post_process(self, strategy: ShardingStrategy) -> Union[ShardingStrategy, List[ShardingStrategy]]:
        # convert bias from its logical sharding spec to its physical sharding spec
        op_data_mapping = self.get_operation_data_mapping()

        for op_name, op_data in op_data_mapping.items():
            if not isinstance(op_data.data, torch.Tensor):
                # remove the sharding spec if the op_data is not a tensor, e.g. torch.pow(tensor, 2)
                strategy.sharding_specs.pop(op_data)
            else:
                # convert the logical sharding spec to physical sharding spec if broadcast
                # e.g. torch.rand(4, 4) + torch.rand(4)
                physical_shape = op_data.data.shape
                logical_shape = op_data.logical_shape
                sharding_spec = strategy.get_sharding_spec_by_name(op_data.name)
                sharding_spec, removed_dims = recover_sharding_spec_for_broadcast_shape(
                    sharding_spec, logical_shape, physical_shape)

                strategy.sharding_specs[op_data] = sharding_spec
                if len(removed_dims) > 0:
                    comm_action = comm_actions_for_oprands(node=self.node,
                                                           removed_dims=removed_dims,
                                                           op_data=op_data,
                                                           sharding_spec=sharding_spec)
                    strategy.communication_actions[op_data] = comm_action

        return strategy
