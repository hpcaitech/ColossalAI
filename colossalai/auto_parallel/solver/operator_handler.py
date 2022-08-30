from webbrowser import Opera
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch.fx.node import Node
from typing import Dict, List
from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.tensor.sharding_spec import ShardingSpec

from .sharding_strategy import StrategiesVector

__all__ = ['OperatorHandler']


class OperatorHandler(ABC):
    '''
    The OperatorHandler is an abstract class used to generate every possible strategies for a operator node.

    Argument:
        input_node(Node): the input node in node argument list.
        input_index(int): the index of input node in the node argument list.
        weight(torch.Tensor): Weight of the node.
        output_node(Node): Output_node is the output of the node.
        device_mesh(DeviceMesh): A logical view of a physical mesh.
        strategies_vector(StrategiesVector): all the strategies generated in this handler will be recorded into the strategies_vector.
        shape_consistency_manager(ShapeConsistencyManager): ShapeConsistencyManager will give the resharding costs of the different sharding specs. 
    '''

    def __init__(self, node: Node, device_mesh: DeviceMesh, strategies_vector: StrategiesVector,
                 shape_consistency_manager: ShapeConsistencyManager):
        self.node = node
        self.predecessor_node = list(node._input_nodes.keys())
        self.successor_node = list(node.users.keys())
        self.device_mesh = device_mesh
        self.strategies_vector = strategies_vector
        self.shape_consistency_manager = shape_consistency_manager

        # find the module and its parameters associated with this node
        # this can be used to compute the compute/communication/sharding cost
        if self.node.op == 'call_module':
            module = node.graph.owning_module.get_submodule(node.target)
            named_parameters = list(module.named_parameters(recurse=False))
            # convert named parameters from list to dict
            named_parameters = {k: v for k, v in named_parameters}
        else:
            module = None
            named_parameters = None
        self.module = module
        self.module_named_parameters = named_parameters

    @abstractmethod
    def register_strategy(self) -> StrategiesVector:
        """
        Register 
        """
        pass

    def _generate_sharding_spec(self, tensor: torch.Tensor, dim_partition_dict: Dict[int, List[int]]) -> ShardingSpec:
        """
        Generate the sharding spec of the tensor based on the given dim_partition_dict 
        where the key is the tensor dimension and the value is the mesh dimension for sharding.
        """
        sharding_spec = ShardingSpec(device_mesh=self.device_mesh,
                                     entire_shape=tensor.shape,
                                     dim_partition_dict=dim_partition_dict)
        return sharding_spec

    def _generate_resharding_costs(self, sharding_spec_for_input):
        '''
        Compute the resharding costs with this specific strategy.

        Note: The resharding_cost of weight is NOT counted.

        Argument:
            resharding_costs(Dict[int, List[float]]): The resharding cost generated in this method will be appended into this dictionary. 
                                                      Resharding_cost[i][j] means the cost of i-th argument in the output node argument list
                                                      with j-th strategy in its strategies_vector transforms to sharding spec wanted in this
                                                      strategy.
            sharding_spec_for_input(ShardingSpec): ShardingSpec of the input node.
        '''
        # The resharding_cost of weight is counted due to sharing weight cases.
        resharding_costs = {}
        for input_node, input_spec in zip(self.predecessor_node, sharding_spec_for_input):
            resharding_costs[input_node] = []
            for strategy in input_node.strategies_vector:
                input_sharding_spec = strategy.output_sharding_spec
                assert isinstance(input_sharding_spec, ShardingSpec), f'The input node should NOT be a tuple of tensor.'
                _, _, resharding_cost = self.shape_consistency_manager.shape_consistency(
                    input_sharding_spec, input_spec)
                resharding_costs[input_node].append(resharding_cost)
        return resharding_costs
