import operator
import torch
from colossalai.tensor.sharding_spec import ShardingSpec
from functools import reduce
from abc import ABC, abstractmethod
from colossalai.tensor.shape_consistency import CollectiveCommPattern, CommSpec
from colossalai.tensor.sharding_spec import ShardingSpec
from colossalai.device.device_mesh import DeviceMesh
from typing import Dict, List, Union, Any
from ..sharding_strategy import OperationData, ShardingStrategy_V2, TrainCycleItem, OperationDataType
from torch.fx import Node
import copy


class StrategyGenerator_V2(ABC):
    """
    StrategyGenerator is used to generate the same group of sharding strategies. 

    TODO: remove the original strategy_generator.py after refactoring
    """

    def __init__(self, operation_data_mapping: Dict[str, OperationData], device_mesh: DeviceMesh):
        self.op_data = operation_data_mapping
        self.device_mesh = device_mesh

    def is_param(self, op_data_name):
        other_data = self.op_data[op_data_name]
        return other_data.type == OperationDataType.PARAM

    def get_sharding_strategy(self, name: str, sharding_spec_mapping: Dict[str, ShardingSpec],
                              communication_action_mapping: Dict[str, CommSpec]):
        """
        A factory method to produce a ShardingStrategy object.

        Args:
            sharding_spec_mapping (Dict[str, ShardingSpec]): the mapping between the operation data name and the ShardingSpec object.
            communication_action_mapping (Dict[str, CommSpec]): the mapping between the operation data name and the CommSpec object.
        """
        sharding_specs = self.replace_op_name_with_op_data(sharding_spec_mapping)
        communication_actions = self.replace_op_name_with_op_data(communication_action_mapping)
        return ShardingStrategy_V2(name=name,
                                   sharding_specs=sharding_specs,
                                   communication_actions=communication_actions)

    def to_sharding_spec_mapping(self, mapping: Dict[str, Dict[int, List[int]]]):
        """
        A utility method to convert the the dim partition dict to a ShardingSpec object.

        Args:
            mapping (Dict[str, Dict[int, List[int]]]): the key of the mapping is the operation data name and the value is a dim partition dictionary.
        """
        results = {}
        for op_data_name, dim_partition_dict in mapping.items():
            if op_data_name in self.op_data:
                op_data = self.op_data[op_data_name]
                if isinstance(op_data.data, tuple) and isinstance(op_data.data[0], torch.Tensor):
                    sharding_spec = []
                    for output, dim_partition_dict_element in zip(op_data.data, dim_partition_dict):
                        sharding_spec = ShardingSpec(device_mesh=self.device_mesh,
                                                     entire_shape=output.shape,
                                                     dim_partition_dict=dim_partition_dict_element)
                else:
                    sharding_spec = ShardingSpec(device_mesh=self.device_mesh,
                                                 entire_shape=op_data.logical_shape,
                                                 dim_partition_dict=dim_partition_dict)
                results[op_data_name] = sharding_spec
        return results

    def replace_op_name_with_op_data(self, mapping: Dict[str, Any]):
        """
        Convert the key of the dictionary from the operation data name to an OperationData object.
        """
        results = {}
        for k, v in mapping.items():
            op_data = self.op_data[k]
            results[op_data] = v
        return results

    def get_communication_spec(self, sharding_spec: ShardingSpec, communication_pattern: CollectiveCommPattern,
                               logical_process_axis: Union[int, List[int]]):
        """
        A factory method to produce a CommSpec object.
        """
        return CommSpec(comm_pattern=communication_pattern,
                        sharding_spec=sharding_spec,
                        logical_process_axis=logical_process_axis)

    def update_communication_cost(self, strategy: ShardingStrategy_V2) -> ShardingStrategy_V2:
        """
        Compute the communication cost involved in the forward and backward iteration.
        """

        comm_cost = TrainCycleItem(fwd=0, bwd=0, total=0)

        def _compute_and_add(data: OperationData, comm_spec: CommSpec):
            num_ele_in_comm = comm_spec.get_comm_cost()
            dtype = operand.data.dtype
            size_per_elem_bytes = torch.tensor([], dtype=dtype).element_size()
            for phase, cost in num_ele_in_comm.items():
                num_ele_in_comm[phase] = num_ele_in_comm[phase] * size_per_elem_bytes
            comm_cost.fwd += num_ele_in_comm['forward']
            comm_cost.bwd += num_ele_in_comm['backward']
            comm_cost.total += num_ele_in_comm['total']

        # check if communication action exists
        # if so, loop over each action and compute the cost of each action
        if strategy.communication_actions is not None:
            for operand, comm_spec in strategy.communication_actions.items():
                _compute_and_add(operand, comm_spec)

        # update the communication cost attribute in-place
        strategy.communication_cost = comm_cost
        return strategy

    @abstractmethod
    def update_compute_cost(self, strategy: ShardingStrategy_V2) -> ShardingStrategy_V2:
        """
        Customize this method to compute the computation flops.
        """
        pass

    @abstractmethod
    def update_memory_cost(self, strategy: ShardingStrategy_V2) -> ShardingStrategy_V2:
        """
        Customize this method to compute the memory cost in bytes.
        """
        pass

    def _compute_size_in_bytes(self, strategy: ShardingStrategy_V2, key: str):
        """
        Compute the size of a tensor in bytes.
        
        Args:
            strategy (ShardingStrategy): the ShardingStrategy generated.
            key (str): the name of the operation data defined by the generator.

        """
        op_data = self.op_data[key]
        sharded_shape = strategy.sharding_specs[op_data].get_sharded_shape_per_device()
        dtype = self.op_data[key].data.dtype
        size_per_elem_bytes = torch.tensor([], dtype=dtype).element_size()
        return reduce(operator.mul, sharded_shape) * size_per_elem_bytes

    @abstractmethod
    def generate(self) -> List[ShardingStrategy_V2]:
        """
        Generate all possible sharding strategies for this operation.
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate if the operands are of desired shape. 
        If True, means this generator can be used for the current operation.
        """
        pass


class FollowingStrategyGenerator(StrategyGenerator_V2):
    """
    FollowingStrategyGenerator is used to generate the sharding strategies which depends on its predecessor node. 

    TODO: remove the original strategy_generator.py after refactoring
    """

    def __init__(self, operation_data_mapping: Dict[str, OperationData], device_mesh: DeviceMesh,
                 predecessor_node: Node):
        self.op_data = operation_data_mapping
        self.device_mesh = device_mesh
        self.predecessor_node = predecessor_node


class OutputStrategyGenerator(StrategyGenerator_V2):
    """
    OutputStrategyGenerator is used to generate the sharding strategies for Output Node.
    """

    def __init__(self, operation_data_mapping: Dict[str, OperationData], device_mesh: DeviceMesh,
                 predecessor_nodes: List[Node]):
        self.op_data = operation_data_mapping
        self.device_mesh = device_mesh
        self.predecessor_nodes = predecessor_nodes
