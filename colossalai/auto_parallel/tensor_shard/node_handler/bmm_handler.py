from typing import Dict, List, Union

import torch

from colossalai.tensor.shape_consistency import CollectiveCommPattern, CommSpec, ShapeConsistencyManager

from ..sharding_strategy import CommAction, CommType, OperationData, OperationDataType, ShardingStrategy
from ..utils import comm_actions_for_oprands, recover_sharding_spec_for_broadcast_shape
from .node_handler import NodeHandler
from .registry import operator_registry
from .strategy import BatchedMatMulStrategyGenerator, StrategyGenerator

__all__ = ['BMMFunctionHandler', 'AddBMMFunctionHandler']


def _get_data_mapping_for_bmm_op(node, input_idx, other_idx, bias_idx=None):
    """
    This function is a helper function which extracts the common logic for both `bmm` and `addbmm`
    node handler to reduce code redundancy.
    """
    # input operand
    physical_input_operand = OperationData(name=str(node.args[input_idx]),
                                           type=OperationDataType.ARG,
                                           data=node.args[input_idx]._meta_data)

    # other operand
    physical_other_operand = OperationData(name=str(node.args[other_idx]),
                                           type=OperationDataType.ARG,
                                           data=node.args[other_idx]._meta_data)

    # output
    physical_output = OperationData(name=str(node), type=OperationDataType.OUTPUT, data=node._meta_data)
    mapping = {"input": physical_input_operand, "other": physical_other_operand, "output": physical_output}

    if bias_idx is not None:
        # bias physical shape
        bias_logical_shape = node._meta_data.shape
        physical_bias_operand = OperationData(name=str(node.args[bias_idx]),
                                              type=OperationDataType.ARG,
                                              data=node.args[bias_idx]._meta_data,
                                              logical_shape=bias_logical_shape)
        mapping['bias'] = physical_bias_operand
    return mapping


@operator_registry.register(torch.bmm)
@operator_registry.register(torch.Tensor.bmm)
class BMMFunctionHandler(NodeHandler):
    """
    This is a NodeHandler class which deals with the batched matrix multiplication operation in PyTorch.
    Such operations including `torch.bmm` and `torch.Tensor.bmm` require the tensor to be 3D, thus, there is
    no logical-physical shape conversion in this handler.
    """

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        mapping = _get_data_mapping_for_bmm_op(node=self.node, input_idx=0, other_idx=1)
        return mapping

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(BatchedMatMulStrategyGenerator(op_data_mapping, self.device_mesh))
        return generators


@operator_registry.register(torch.addbmm)
@operator_registry.register(torch.Tensor.addbmm)
class AddBMMFunctionHandler(NodeHandler):
    """
    This is a NodeHandler class which deals with the addition + batched matrix multiplication operation in PyTorch.
    Such operations including `torch.addbmm` and `torch.Tensor.addbmm` require the two matmul tensor to be 3D. However, due to the
    addition, logical-physical shape conversion is required for the bias term.

    As the addbmm operation will reduce the batch dimension, the bias is maximum 2D.
    """

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        mapping = _get_data_mapping_for_bmm_op(node=self.node, input_idx=1, other_idx=2, bias_idx=0)
        return mapping

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(BatchedMatMulStrategyGenerator(op_data_mapping, self.device_mesh))
        return generators

    def post_process(self, strategy: ShardingStrategy) -> Union[ShardingStrategy, List[ShardingStrategy]]:
        # convert bias from its logical sharding spec to its physical sharding spec
        op_data_mapping = self.get_operation_data_mapping()

        if 'bias' in op_data_mapping:
            bias_op_data = op_data_mapping['bias']
            bias_physical_shape = bias_op_data.data.shape
            bias_logical_shape = bias_op_data.logical_shape
            bias_sharding_spec = strategy.get_sharding_spec_by_name(bias_op_data.name)
            bias_sharding_spec, removed_dims = recover_sharding_spec_for_broadcast_shape(
                bias_sharding_spec, bias_logical_shape, bias_physical_shape)
            strategy.sharding_specs[bias_op_data] = bias_sharding_spec

            if len(removed_dims) > 0:
                comm_action = comm_actions_for_oprands(node=self.node,
                                                       removed_dims=removed_dims,
                                                       op_data=bias_op_data,
                                                       sharding_spec=bias_sharding_spec)
                strategy.communication_actions[bias_op_data] = comm_action

        return strategy
