import torch
import torch.nn.functional as F
from .node_handler import ModuleHandler, NodeHandler
from ..sharding_strategy import ShardingStrategy_V2, OperationDataType, OperationData
from ..strategy import ConvStrategyGenerator, StrategyGenerator_V2
from typing import List, Dict
from .registry import operator_registry

__all__ = ['LinearModuleHandler', 'LinearFunctionHandler']


@operator_registry.register(torch.nn.Conv1d)
@operator_registry.register(torch.nn.Conv2d)
@operator_registry.register(torch.nn.Conv3d)
class ConvModuleHandler(ModuleHandler):
    """
    A ConvModuleHandler which deals with the sharding strategies for nn.Convxd module.
    """

    def get_strategy_generator(self) -> List[StrategyGenerator_V2]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(ConvStrategyGenerator(op_data_mapping, self.device_mesh))
        return generators

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        # use transposed shape for strategies
        # the strategies will be transformed back to its original shape in self.post_process
        physical_input_operand = OperationData(name=str(self.node.args[0]),
                                               type=OperationDataType.ARG,
                                               data=self.node.args[0]._meta_data)
        logical_shape_for_weight = list(self.named_parameters["weight"].shape)
        logical_shape_for_weight[0], logical_shape_for_weight[1] = logical_shape_for_weight[
            1], logical_shape_for_weight[0]
        physical_other_operand = OperationData(name="weight",
                                               type=OperationDataType.PARAM,
                                               data=self.named_parameters['weight'],
                                               logical_shape=torch.Size(logical_shape_for_weight))
        physical_output = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=self.node._meta_data)

        mapping = {"input": physical_input_operand, "other": physical_other_operand, "output": physical_output}

        if "bias" in self.named_parameters:
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
                dim_partition_dict = sharding_spec.dim_partition_dict

                # switch first and second dim of the conv module weight
                first_dim_partition = dim_partition_dict.pop(1, None)
                second_dim_partition = dim_partition_dict.pop(0, None)

                if first_dim_partition:
                    dim_partition_dict[0] = first_dim_partition

                if second_dim_partition:
                    dim_partition_dict[1] = second_dim_partition

                # re-init the sharding spec
                sharding_spec.__init__(sharding_spec.device_mesh, sharding_spec.entire_shape, dim_partition_dict)
        return strategy


@operator_registry.register(F.conv1d)
@operator_registry.register(F.conv2d)
@operator_registry.register(F.conv3d)
class ConvFunctionHandler(NodeHandler):
    """
    A ConvFunctionHandler which deals with the sharding strategies for nn.functional.ConvXd functions.
    """

    def get_strategy_generator(self) -> List[StrategyGenerator_V2]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(ConvStrategyGenerator(op_data_mapping, self.device_mesh))
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

        logical_shape_for_weight = list(self.node.args[1]._meta_data.shape)
        logical_shape_for_weight[0], logical_shape_for_weight[1] = logical_shape_for_weight[
            1], logical_shape_for_weight[0]
        physical_other_operand = OperationData(name=str(self.node.args[1]),
                                               type=data_type,
                                               data=self.node.args[1]._meta_data,
                                               logical_shape=torch.Size(logical_shape_for_weight))
        physical_output = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=self.node._meta_data)

        mapping = {"input": physical_input_operand, "other": physical_other_operand, "output": physical_output}

        if "bias" in self.node.kwargs:
            # check if the other operand is a parameter
            if isinstance(self.node.kwargs["bias"]._meta_data, torch.nn.parameter.Parameter):
                data_type = OperationDataType.PARAM
            else:
                data_type = OperationDataType.ARG
            physical_bias_operand = OperationData(name=str(self.node.kwargs["bias"]),
                                                  type=data_type,
                                                  data=self.node.kwargs["bias"]._meta_data)
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

                # switch first and second dim of the conv function weight
                first_dim_partition = dim_partition_dict.pop(1, None)
                second_dim_partition = dim_partition_dict.pop(0, None)

                if first_dim_partition:
                    dim_partition_dict[0] = first_dim_partition

                if second_dim_partition:
                    dim_partition_dict[1] = second_dim_partition

                # re-init the sharding spec
                sharding_spec.__init__(sharding_spec.device_mesh, sharding_spec.entire_shape, dim_partition_dict)
        return strategy
