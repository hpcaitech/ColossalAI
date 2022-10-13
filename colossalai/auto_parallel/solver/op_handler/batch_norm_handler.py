import torch
import torch.nn.functional as F
from .node_handler import ModuleHandler, NodeHandler
from ..sharding_strategy import ShardingStrategy, OperationDataType, OperationData
from ..strategy import BatchNormStrategyGenerator, StrategyGenerator
from typing import List, Dict
from .registry import operator_registry

__all__ = ['BatchNormModuleHandler']


@operator_registry.register(torch.nn.BatchNorm1d)
@operator_registry.register(torch.nn.BatchNorm2d)
@operator_registry.register(torch.nn.BatchNorm3d)
class BatchNormModuleHandler(ModuleHandler):
    """
    A BatchNormModuleHandler which deals with the sharding strategies for nn.BatchNormXd module.
    """

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(BatchNormStrategyGenerator(op_data_mapping, self.device_mesh))
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
                                               logical_shape=self.named_parameters['weight'].shape)
        physical_output = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=self.node._meta_data)

        mapping = {"input": physical_input_operand, "other": physical_other_operand, "output": physical_output}

        if self.named_parameters['bias'] is not None:
            physical_bias_operand = OperationData(name="bias",
                                                  type=OperationDataType.PARAM,
                                                  data=self.named_parameters['bias'])
            mapping['bias'] = physical_bias_operand
        return mapping
