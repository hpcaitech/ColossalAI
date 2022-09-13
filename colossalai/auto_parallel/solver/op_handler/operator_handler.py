from webbrowser import Opera
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch.fx.node import Node
from typing import Dict, List
from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.tensor.sharding_spec import ShardingSpec
from .._utils import generate_resharding_costs, generate_sharding_spec

from ..sharding_strategy import StrategiesVector

__all__ = ['OperatorHandler']


class OperatorHandler(ABC):
    '''
    The OperatorHandler is an abstract class used to generate every possible strategies for an operator node.

    Args:
        node (Node): the input node in node argument list.
        device_mesh (DeviceMesh): A logical view of a physical mesh.
        strategies_vector (StrategiesVector): all the strategies generated in this handler will be recorded into the strategies_vector.
        handle_backward (Optional[bool]): whether to consider the backward pass. The default value is True. False can be used for inference.
    '''

    def __init__(self,
                 node: Node,
                 device_mesh: DeviceMesh,
                 strategies_vector: StrategiesVector,
                 handle_backward: bool = True):
        self.node = node
        self.predecessor_node = list(node._input_nodes.keys())
        self.successor_node = list(node.users.keys())
        self.device_mesh = device_mesh
        self.strategies_vector = strategies_vector
        self.handle_backward = handle_backward

        # find the module and its parameters associated with this node
        # this can be used to compute the compute/communication/sharding cost
        if self.node.op == 'call_module':
            module = node.graph.owning_module.get_submodule(node.target)
            named_parameters = list(module.named_parameters(recurse=False))
            # convert named parameters from list to dict
            named_parameters = {k: v for k, v in named_parameters}
        elif self.node.op == 'call_function':
            module = None
            parameters = list(self.node.args)[1]
            named_parameters = {'weight': parameters._meta_data}
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

    def _generate_memory_cost(self, dim_partition_dict_for_output, dim_partition_dict_for_weight):
        '''
        Compute the memory cost per device with this specific strategy.

        Argument:
            dim_partition_dict_for_output(List[int]): The key is the dimension of output to be sharded,
                and the value of the key decribe which logical axis will be sharded in that dimension.
            dim_partition_dict_for_weight(List[int]): The key is the dimension of weight to be sharded,
                and the value of the key decribe which logical axis will be sharded in that dimension.
        Return:
            total_memory_cost(float): total memory cost per device with this specific strategy
            activation_cost(float): the memory cost of activation per device with this specific strategy
            weight_memory_cost(float): the memory cost of weight per device with this specific strategy
        '''
        # compute the size of one element with specific dtype
        dtype = self.input_data.dtype
        size_per_elem_bytes = torch.tensor([], dtype=dtype).element_size()

        # compute the memory cost of activation
        activation_numel = self.output_data.numel()
        output_mesh_dims = []
        for sharding_dim, mesh_dims in dim_partition_dict_for_output.items():
            output_mesh_dims.extend(mesh_dims)
        activation_sharding_size = 1
        for mesh_dim in output_mesh_dims:
            activation_sharding_size *= self.device_mesh.shape[mesh_dim]
        activation_memory_cost = activation_numel / activation_sharding_size * size_per_elem_bytes

        # compute the memory cost of weight
        weight_numel = self.weight.numel()
        weight_sharding_size = 1
        weight_mesh_dims = []
        for sharding_dim, mesh_dims in dim_partition_dict_for_weight.items():
            weight_mesh_dims.extend(mesh_dims)
        for mesh_dim in weight_mesh_dims:
            weight_sharding_size *= self.device_mesh.shape[mesh_dim]
        weight_memory_cost = weight_numel / weight_sharding_size * size_per_elem_bytes

        total_memory_cost = activation_memory_cost + weight_memory_cost

        return total_memory_cost, activation_memory_cost, weight_memory_cost

    def _generate_resharding_costs(self, sharding_specs):
        # The resharding_cost of weight is counted due to sharing weight cases.
        dtype = self.node._meta_data.dtype
        nodes = self.predecessor_node
        return generate_resharding_costs(nodes=nodes,
                                         sharding_specs=sharding_specs,
                                         count_backward=self.handle_backward,
                                         dtype=dtype)

    def _generate_sharding_spec(self, input_: torch.Tensor, dim_partition_dict: Dict[int, List[int]]) -> ShardingSpec:
        return generate_sharding_spec(input_=input_,
                                      device_mesh=self.device_mesh,
                                      dim_partition_dict=dim_partition_dict)

    @abstractmethod
    def _generate_compute_cost(self, *args, **kwargs):
        """
        Compute the flops involved in the node.
        """
        pass
