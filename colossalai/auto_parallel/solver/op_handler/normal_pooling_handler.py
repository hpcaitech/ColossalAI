import torch
import torch.nn.functional as F
from .node_handler import ModuleHandler, NodeHandler
from ..sharding_strategy import ShardingStrategy_V2, OperationDataType, OperationData
from ..strategy import NormalPoolStrategyGenerator, StrategyGenerator_V2
from typing import List, Dict
from .registry import operator_registry

__all__ = ['LinearModuleHandler', 'LinearFunctionHandler']


@operator_registry.register(torch.nn.MaxPool1d)
@operator_registry.register(torch.nn.MaxPool2d)
@operator_registry.register(torch.nn.MaxPool1d)
@operator_registry.register(torch.nn.AvgPool1d)
@operator_registry.register(torch.nn.AvgPool2d)
@operator_registry.register(torch.nn.AvgPool3d)
class NormPoolingHandler(ModuleHandler):
    """
    A NormPoolingHandler which deals with the sharding strategies for nn.MaxPoolxd module.
    """

    def get_strategy_generator(self) -> List[StrategyGenerator_V2]:
        op_data_mapping = self.get_operation_data_mapping()
        generators = []
        generators.append(NormalPoolStrategyGenerator(op_data_mapping, self.device_mesh))
        return generators

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        # use transposed shape for strategies
        # the strategies will be transformed back to its original shape in self.post_process
        physical_input_operand = OperationData(name=str(self.node.args[0]),
                                               type=OperationDataType.ARG,
                                               data=self.node.args[0]._meta_data)
        physical_weight_operand = OperationData(name="kernel", type=OperationDataType.ARG, data=self.module.kernel_size)
        physical_output = OperationData(name=str(self.node), type=OperationDataType.OUTPUT, data=self.node._meta_data)

        mapping = {"input": physical_input_operand, "other": physical_weight_operand, "output": physical_output}

        return mapping
