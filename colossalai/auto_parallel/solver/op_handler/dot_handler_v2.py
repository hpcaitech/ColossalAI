import torch
import torch.nn as nn
import torch.nn.functional as F
from .node_handler import ModuleHandler, NodeHandler
from ..sharding_strategy import ShardingStrategy_V2, StrategyGenerator_V2, OperationDataType, OperationData
from typing import List, Dict
from .registry import operator_registry

__all__ = ['LinearModuleHandler']


class DotProductStrategyGenerator(StrategyGenerator_V2):
    """TODO: to be implemented"""
    pass


class MatVecStrategyGenerator(StrategyGenerator_V2):
    """TODO: to be implemented"""
    pass


class LinearProjectionStrategyGenerator(StrategyGenerator_V2):

    def update_compute_cost(self, strategy: ShardingStrategy_V2) -> ShardingStrategy_V2:
        """TODO: to be implemented"""
        pass

    def update_memory_cost(self, strategy: ShardingStrategy_V2) -> ShardingStrategy_V2:
        """TODO: to be implemented"""
        pass

    def generate(self, operand_mapping: Dict[str, OperationData]) -> List[ShardingStrategy_V2]:
        """TODO: to be implemented"""
        pass

    def validate(self, *args, **kwargs) -> bool:
        """TODO: to be implemented"""
        pass


class BatchedMatMulStrategyGenerator(StrategyGenerator_V2):
    """TODO: to be implemented"""
    pass


@operator_registry.register(torch.nn.Linear)
class LinearModuleHandler(ModuleHandler):
    """
    A LinearModuleHandler which deals with the sharding strategies for nn.Linear module.
    """

    def register_strategy_generator(self) -> List[StrategyGenerator_V2]:
        generators = []
        generators.append(LinearProjectionStrategyGenerator(self.device_mesh))
        return generators

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        # use transposed shape for strategies
        # the strategies will be transformed back to its original shape in self.post_process
        physical_input_operand = OperationData(name=str(self.node.args[0]),
                                               type=OperationDataType.ARG,
                                               data=self.node.args[0]._meta_data)
        physical_other_operand = OperationData(name="weight",
                                               type=OperationDataType.PARAM,
                                               data=self.named_parameters['weight'],
                                               logical_shape=self.named_parameters['weight'].shape[::-1])
        physical_output = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=self.node._meta_data)

        mapping = {"input": physical_input_operand, "other": physical_other_operand, "output": physical_output}

        if self.named_parameters['bias'] is not None:
            physical_bias_operand = OperationData(name="bias",
                                                  type=OperationDataType.PARAM,
                                                  data=self.named_parameters['bias'])
            mapping['bias'] = physical_bias_operand
        return mapping

    def post_process(self, strategy: ShardingStrategy_V2):
        """
        Convert the sharding spec of the weight parameter back to its original shape.
        """
        for op_data, sharding_spec in strategy.input_sharding_specs.items():
            if op_data.name == "weight":
                assert op_data.logical_shape != op_data.data.shape
                dim_partition_dict = sharding_spec.dim_partition_dict
                # switch first and last dim of the linear module weight
                dim_partition_dict[0], dim_partition_dict[-1] = dim_partition_dict[-1], dim_partition_dict[0]

                # re-init the sharding spec
                sharding_spec.__init__(sharding_spec.device_mesh, sharding_spec.entire_shape, dim_partition_dict)
        return strategy


@operator_registry.register(F.linear)
class LinearFunctionHandler(NodeHandler):
    """
    A LinearModuleHandler which deals with the sharding strategies for nn.Linear module.
    """

    def register_strategy_generator(self) -> List[StrategyGenerator_V2]:
        generators = []
        generators.append(LinearProjectionStrategyGenerator(self.device_mesh))
        return generators

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        # use transposed shape for strategies
        # the strategies will be transformed back to its original shape in self.post_process
        physical_input_operand = OperationData(name=str(self.node.args[0]),
                                               type=OperationDataType.ARG,
                                               data=self.node.args[0]._meta_data)
        physical_other_operand = OperationData(name=str(self.node.args[1]),
                                               type=OperationDataType.ARG,
                                               data=self.node.args[1]._meta_data,
                                               logical_shape=self.node.args[1]._meta_data.shape[::-1])
        physical_output = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=self.node._meta_data)

        mapping = {"input": physical_input_operand, "other": physical_other_operand, "output": physical_output}

        if self.node.args[2] is not None:
            physical_bias_operand = OperationData(name=str(self.node.args[2]),
                                                  type=OperationDataType.ARG,
                                                  data=self.node.args[2]._meta_data)
            mapping['bias'] = physical_bias_operand
        return mapping

    def post_process(self, strategy: ShardingStrategy_V2):
        """
        Convert the sharding spec of the weight parameter back to its original shape.
        """
        for op_data, sharding_spec in strategy.input_sharding_specs.items():
            if op_data.name == str(self.node.args[1]):
                assert op_data.logical_shape != op_data.data.shape
                dim_partition_dict = sharding_spec.dim_partition_dict
                # switch first and last dim of the linear module weight
                dim_partition_dict[0], dim_partition_dict[-1] = dim_partition_dict[-1], dim_partition_dict[0]

                # re-init the sharding spec
                sharding_spec.__init__(sharding_spec.device_mesh, sharding_spec.entire_shape, dim_partition_dict)
        return strategy
