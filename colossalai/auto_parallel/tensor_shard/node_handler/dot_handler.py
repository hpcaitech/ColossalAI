from copy import deepcopy
from typing import Dict, List, Union

import torch
import torch.nn.functional as F

from colossalai.auto_parallel.tensor_shard.utils import (switch_partition_dim, update_partition_dim)
from colossalai.tensor.sharding_spec import ShardingException

from ..sharding_strategy import (OperationData, OperationDataType, ShardingStrategy)
from .node_handler import ModuleHandler, NodeHandler
from .registry import operator_registry
from .strategy import (BatchedMatMulStrategyGenerator, LinearProjectionStrategyGenerator, StrategyGenerator)

__all__ = ['LinearModuleHandler', 'LinearFunctionHandler', 'BMMFunctionHandler']


@operator_registry.register(torch.nn.Linear)
class LinearModuleHandler(ModuleHandler):
    """
    A LinearModuleHandler which deals with the sharding strategies for nn.Linear module.
    """

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(LinearProjectionStrategyGenerator(op_data_mapping, self.device_mesh))
        return generators

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        # use transposed shape for strategies
        # the strategies will be transformed back to its original shape in self.post_process
        input_meta_data = self.node.args[0]._meta_data
        input_logical_shape = input_meta_data.view(-1, input_meta_data.shape[-1]).shape
        physical_input_operand = OperationData(name=str(self.node.args[0]),
                                               type=OperationDataType.ARG,
                                               data=input_meta_data,
                                               logical_shape=input_logical_shape)
        physical_other_operand = OperationData(name="weight",
                                               type=OperationDataType.PARAM,
                                               data=self.named_parameters['weight'],
                                               logical_shape=self.named_parameters['weight'].shape[::-1])
        output_meta_data = self.node._meta_data
        output_logical_shape = output_meta_data.view(-1, output_meta_data.shape[-1]).shape
        physical_output = OperationData(name=str(self.node),
                                        type=OperationDataType.OUTPUT,
                                        data=output_meta_data,
                                        logical_shape=output_logical_shape)

        mapping = {"input": physical_input_operand, "other": physical_other_operand, "output": physical_output}

        if self.named_parameters['bias'] is not None:
            physical_bias_operand = OperationData(name="bias",
                                                  type=OperationDataType.PARAM,
                                                  data=self.named_parameters['bias'])
            mapping['bias'] = physical_bias_operand
        return mapping

    def post_process(self, strategy: ShardingStrategy) -> Union[ShardingStrategy, List[ShardingStrategy]]:
        """
        Convert the sharding spec from the logical shape to the physical shape.
        """
        # switch the dimensions of the transposed weight
        for op_data, sharding_spec in strategy.input_sharding_specs.items():
            if op_data.name == "weight":
                assert op_data.logical_shape != op_data.data.shape
                switch_partition_dim(sharding_spec, 0, -1)

        # create multiple sharding strategies for the inputs
        # as input can be multi-dimensinal and the partition dim is only 2D,
        # we need to map the partition at dim 0 to one of the first few dimensions of the input
        sharding_strategies = []
        input_op_data = strategy.get_op_data_by_name(str(self.node.args[0]))
        output_op_data = strategy.get_op_data_by_name(str(self.node))
        num_input_dims = input_op_data.data.dim()
        input_sharding_spec = strategy.get_sharding_spec_by_name(input_op_data.name)

        if 0 in input_sharding_spec.dim_partition_dict:
            for i in range(num_input_dims - 1):
                new_strategy = strategy.clone()
                input_sharding_spec = new_strategy.get_sharding_spec_by_name(input_op_data.name)
                output_sharding_spec = new_strategy.get_sharding_spec_by_name(output_op_data.name)
                try:
                    update_partition_dim(sharding_spec=input_sharding_spec,
                                         dim_mapping={0: i},
                                         physical_shape=input_op_data.data.shape,
                                         inplace=True)
                    update_partition_dim(sharding_spec=output_sharding_spec,
                                         dim_mapping={0: i},
                                         physical_shape=output_op_data.data.shape,
                                         inplace=True)
                    sharding_strategies.append(new_strategy)
                except ShardingException:
                    pass
        else:
            sharding_strategies.append(strategy)

        return sharding_strategies


@operator_registry.register(F.linear)
class LinearFunctionHandler(NodeHandler):
    """
    A LinearModuleHandler which deals with the sharding strategies for nn.Linear module.
    """

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(LinearProjectionStrategyGenerator(op_data_mapping, self.device_mesh))
        return generators

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        # use transposed shape for strategies
        # the strategies will be transformed back to its original shape in self.post_process
        physical_input_operand = OperationData(name=str(self.node.args[0]),
                                               type=OperationDataType.ARG,
                                               data=self.node.args[0]._meta_data)

        # check if the other operand is a parameter
        if isinstance(self.node.args[1]._meta_data, torch.nn.parameter.Parameter):
            data_type = OperationDataType.PARAM
        else:
            data_type = OperationDataType.ARG

        physical_other_operand = OperationData(name=str(self.node.args[1]),
                                               type=data_type,
                                               data=self.node.args[1]._meta_data,
                                               logical_shape=self.node.args[1]._meta_data.shape[::-1])
        physical_output = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=self.node._meta_data)

        mapping = {"input": physical_input_operand, "other": physical_other_operand, "output": physical_output}

        if self.node.args[2] is not None:
            # check if the other operand is a parameter
            if isinstance(self.node.args[2]._meta_data, torch.nn.parameter.Parameter):
                data_type = OperationDataType.PARAM
            else:
                data_type = OperationDataType.ARG
            physical_bias_operand = OperationData(name=str(self.node.args[2]),
                                                  type=data_type,
                                                  data=self.node.args[2]._meta_data)
            mapping['bias'] = physical_bias_operand
        return mapping

    def post_process(self, strategy: ShardingStrategy):
        """
        Convert the sharding spec of the weight parameter back to its original shape.
        """
        for op_data, sharding_spec in strategy.input_sharding_specs.items():
            if op_data.name == str(self.node.args[1]):
                assert op_data.logical_shape != op_data.data.shape
                switch_partition_dim(sharding_spec, 0, -1)

        # create multiple sharding strategies for the inputs
        # as input can be multi-dimensinal and the partition dim is only 2D,
        # we need to map the partition at dim 0 to one of the first few dimensions of the input
        sharding_strategies = []
        input_op_data = strategy.get_op_data_by_name(str(self.node.args[0]))
        output_op_data = strategy.get_op_data_by_name(str(self.node))
        num_input_dims = input_op_data.data.dim()
        input_sharding_spec = strategy.get_sharding_spec_by_name(input_op_data.name)

        if 0 in input_sharding_spec.dim_partition_dict:
            for i in range(num_input_dims - 1):
                new_strategy = strategy.clone()
                input_sharding_spec = new_strategy.get_sharding_spec_by_name(input_op_data.name)
                output_sharding_spec = new_strategy.get_sharding_spec_by_name(output_op_data.name)
                try:
                    update_partition_dim(sharding_spec=input_sharding_spec,
                                         dim_mapping={0: i},
                                         physical_shape=input_op_data.data.shape,
                                         inplace=True)
                    update_partition_dim(sharding_spec=output_sharding_spec,
                                         dim_mapping={0: i},
                                         physical_shape=output_op_data.data.shape,
                                         inplace=True)
                    sharding_strategies.append(new_strategy)
                except ShardingException:
                    pass
        else:
            sharding_strategies.append(strategy)

        return strategy


@operator_registry.register(torch.bmm)
@operator_registry.register(torch.Tensor.bmm)
class BMMFunctionHandler(NodeHandler):

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        # use transposed shape for strategies
        # the strategies will be transformed back to its original shape in self.post_process
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
