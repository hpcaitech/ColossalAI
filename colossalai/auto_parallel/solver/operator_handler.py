from abc import ABC, abstractmethod
from torch.fx.node import Node
import torch.nn as nn
from colossalai.device.device_mesh import DeviceMesh
from .sharding_strategy import StrategiesVector
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.tensor.sharding_spec import ShardingSpec


class OperatorHanlder(ABC):
    '''
    The OperatorHanlder is an abstract class used to generate every possible strategies for a operator node.

    Argument:
        input_node(Node): the input node in node argument list.
        input_index(int): the index of input node in the node argument list.
        weight(torch.Tensor): Weight of the node.
        output_node(Node): Output_node is the output of the node.
        device_mesh(DeviceMesh): A logical view of a physical mesh.
        strategies_vector(StrategiesVector): all the strategies generated in this handler will be recorded into the strategies_vector.
        shape_consistency_manager(ShapeConsistencyManager): ShapeConsistencyManager will give the resharding costs of the different sharding specs. 
    '''

    def __init__(self, input_node: Node, input_index: int, weight: nn.Parameter, output_node: Node,
                 device_mesh: DeviceMesh, strategies_vector: StrategiesVector,
                 shape_consistency_manager: ShapeConsistencyManager):
        self.input_node = input_node
        self.input_data = self.input_node._meta_data
        self.weight = weight
        self.input_index = input_index
        self.output_node = output_node
        self.output = self.output_node._meta_data
        self.device_mesh = device_mesh
        self.strategies_vector = strategies_vector
        self.shape_consistency_manager = shape_consistency_manager

    @abstractmethod
    def register_strategy_into_strategies_vector(self):
        pass

    def _generate_sharding_spec(self, tensor, dim_partition_dict):
        sharding_spec = ShardingSpec(device_mesh=self.device_mesh,
                                     entire_shape=tensor.shape,
                                     dim_partition_dict=dim_partition_dict)
        return sharding_spec
