from typing import Dict, List, Union

import torch

from ..sharding_strategy import OperationData, OperationDataType, ShardingStrategy
from ..utils import comm_actions_for_oprands, recover_sharding_spec_for_broadcast_shape
from .node_handler import NodeHandler
from .registry import operator_registry
from .strategy import LinearProjectionStrategyGenerator, StrategyGenerator

__all__ = ["ADDMMFunctionHandler"]


@operator_registry.register(torch.addmm)
@operator_registry.register(torch.Tensor.addmm)
class ADDMMFunctionHandler(NodeHandler):
    """
    This is a NodeHandler class which deals with the batched matrix multiplication operation in PyTorch.
    Such operations including `torch.bmm` and `torch.Tensor.bmm` require the tensor to be 3D, thus, there is
    no logical-physical shape conversion in this handler.
    """

    def _infer_op_data_type(self, tensor: torch.Tensor) -> OperationDataType:
        if isinstance(tensor, torch.nn.parameter.Parameter):
            data_type = OperationDataType.PARAM
        else:
            data_type = OperationDataType.ARG
        return data_type

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        # input operand
        input_data = self.node.args[1]._meta_data
        physical_input_operand = OperationData(
            name=str(self.node.args[1]), type=self._infer_op_data_type(input_data), data=input_data
        )

        # other operand
        other_data = self.node.args[2]._meta_data
        physical_other_operand = OperationData(
            name=str(self.node.args[2]), type=self._infer_op_data_type(other_data), data=other_data
        )
        # bias physical shape
        bias_logical_shape = self.node._meta_data.shape
        bias_data = self.node.args[0]._meta_data
        physical_bias_operand = OperationData(
            name=str(self.node.args[0]),
            type=self._infer_op_data_type(bias_data),
            data=bias_data,
            logical_shape=bias_logical_shape,
        )

        # output
        physical_output = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=self.node._meta_data)

        mapping = {
            "input": physical_input_operand,
            "other": physical_other_operand,
            "output": physical_output,
            "bias": physical_bias_operand,
        }

        return mapping

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(
            LinearProjectionStrategyGenerator(op_data_mapping, self.device_mesh, linear_projection_type="addmm")
        )
        return generators

    def post_process(self, strategy: ShardingStrategy) -> Union[ShardingStrategy, List[ShardingStrategy]]:
        # convert bias from its logical sharding spec to its physical sharding spec
        op_data_mapping = self.get_operation_data_mapping()

        bias_op_data = op_data_mapping["bias"]
        bias_physical_shape = bias_op_data.data.shape
        bias_logical_shape = bias_op_data.logical_shape
        bias_sharding_spec = strategy.get_sharding_spec_by_name(bias_op_data.name)
        bias_sharding_spec, removed_dims = recover_sharding_spec_for_broadcast_shape(
            bias_sharding_spec, bias_logical_shape, bias_physical_shape
        )
        strategy.sharding_specs[bias_op_data] = bias_sharding_spec

        if len(removed_dims) > 0:
            comm_action = comm_actions_for_oprands(
                node=self.node, removed_dims=removed_dims, op_data=bias_op_data, sharding_spec=bias_sharding_spec
            )
            strategy.communication_actions[bias_op_data] = comm_action

        return strategy
