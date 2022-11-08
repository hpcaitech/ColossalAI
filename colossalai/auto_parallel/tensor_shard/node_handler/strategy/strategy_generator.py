import operator
from abc import ABC, abstractmethod
from functools import reduce
from typing import Any, Dict, List, Union

import torch
from torch.fx import Node

from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    CommAction,
    CommType,
    OperationData,
    OperationDataType,
    ShardingStrategy,
    TrainCycleItem,
)
from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.shape_consistency import CollectiveCommPattern, CommSpec, ShapeConsistencyManager
from colossalai.tensor.sharding_spec import ShardingSpec
from colossalai.tensor.utils import convert_dim_partition_dict


class StrategyGenerator(ABC):
    """
    StrategyGenerator is used to generate the same group of sharding strategies.

    TODO: remove the original strategy_generator.py after refactoring
    """

    def __init__(self, operation_data_mapping: Dict[str, OperationData], device_mesh: DeviceMesh):
        self.op_data = operation_data_mapping
        self.device_mesh = device_mesh

        # validate the whether operation data is of desired value
        self.validate()

    @property
    def has_bias(self):
        """
        A utility method to check for the existence of bias operand for convenience.
        """
        return 'bias' in self.op_data

    def is_param(self, op_data_name):
        other_data = self.op_data[op_data_name]
        return other_data.type == OperationDataType.PARAM

    def is_buffer(self, op_data_name):
        other_data = self.op_data[op_data_name]
        return other_data.type == OperationDataType.BUFFER

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
        return ShardingStrategy(name=name, sharding_specs=sharding_specs, communication_actions=communication_actions)

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
                    for logical_shape, dim_partition_dict_element in zip(op_data.logical_shape, dim_partition_dict):
                        dim_size = len(logical_shape)
                        dim_partition_dict_element = convert_dim_partition_dict(dim_size, dim_partition_dict_element)
                        sharding_spec = ShardingSpec(device_mesh=self.device_mesh,
                                                     entire_shape=logical_shape,
                                                     dim_partition_dict=dim_partition_dict_element)
                else:
                    dim_size = len(op_data.logical_shape)
                    dim_partition_dict = convert_dim_partition_dict(dim_size, dim_partition_dict)
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

    def get_communication_action(self,
                                 sharding_spec: ShardingSpec,
                                 communication_pattern: CollectiveCommPattern,
                                 logical_process_axis: Union[int, List[int]],
                                 comm_type: CommType,
                                 arg_index: int = -1,
                                 key_for_kwarg: any = None) -> CommAction:
        """
        A factory method to produce a CommAction object.
        """
        return CommAction(comm_spec=self.get_communication_spec(sharding_spec=sharding_spec,
                                                                communication_pattern=communication_pattern,
                                                                logical_process_axis=logical_process_axis),
                          comm_type=comm_type,
                          arg_index=arg_index,
                          key_for_kwarg=key_for_kwarg)

    def update_communication_cost(self, strategy: ShardingStrategy) -> ShardingStrategy:
        """
        Compute the communication cost involved in the forward and backward iteration.
        """

        comm_cost = TrainCycleItem(fwd=0, bwd=0, total=0)

        def _compute_and_add(op_data: OperationData, comm_spec: CommSpec):
            num_ele_in_comm = comm_spec.get_comm_cost()
            dtype = op_data.data.dtype
            size_per_elem_bytes = torch.tensor([], dtype=dtype).element_size()
            for phase, cost in num_ele_in_comm.items():
                num_ele_in_comm[phase] = num_ele_in_comm[phase] * size_per_elem_bytes
            comm_cost.fwd += num_ele_in_comm['forward']
            comm_cost.bwd += num_ele_in_comm['backward']
            comm_cost.total += num_ele_in_comm['total']

        # check if communication action exists
        # if so, loop over each action and compute the cost of each action
        if strategy.communication_actions is not None:
            for operand, comm_action in strategy.communication_actions.items():
                if isinstance(comm_action, CommAction):
                    comm_spec = comm_action.comm_spec
                else:
                    # this condition branch will be removed after all the handler updated.
                    comm_spec = comm_action
                if isinstance(comm_spec, dict):
                    src_spec = comm_spec['src_spec']
                    tgt_spec = comm_spec['tgt_spec']
                    shape_consistency_manager = ShapeConsistencyManager()
                    _, comm_action_sequence, _ = shape_consistency_manager.shape_consistency(src_spec, tgt_spec)
                    for comm_spec_ in comm_action_sequence:
                        _compute_and_add(operand, comm_spec_)
                else:
                    _compute_and_add(operand, comm_spec)

        # update the communication cost attribute in-place
        strategy.communication_cost = comm_cost
        return strategy

    @abstractmethod
    def update_compute_cost(self, strategy: ShardingStrategy) -> ShardingStrategy:
        """
        Customize this method to compute the computation flops.
        """
        pass

    @abstractmethod
    def update_memory_cost(self, strategy: ShardingStrategy) -> ShardingStrategy:
        """
        Customize this method to compute the memory cost in bytes.
        """
        pass

    def _compute_size_in_bytes(self, strategy: ShardingStrategy, key: str):
        """
        Compute the size of a tensor in bytes.

        Args:
            strategy (ShardingStrategy): the ShardingStrategy generated.
            key (str): the name of the operation data defined by the generator.

        """
        op_data = self.op_data[key]
        sharded_shape = strategy.sharding_specs[op_data].get_sharded_shape_per_device()

        if len(sharded_shape) == 0:
            num_elements = 1
        else:
            num_elements = reduce(operator.mul, sharded_shape)
        dtype = self.op_data[key].data.dtype
        size_per_elem_bytes = torch.tensor([], dtype=dtype).element_size()
        return num_elements * size_per_elem_bytes

    def generate(self) -> List[ShardingStrategy]:
        """
        Generate all possible sharding strategies for this operation.
        """
        strategies = self.collate_strategies()

        # some strategies may be None as ignore_sharding_exception may return None
        # when ShardingSpecException occurs.
        # thus, remove those None values
        strategies = [strategy for strategy in strategies if strategy]

        # update the costs
        # update mete info on cost
        # these update methods are all in-place, the default method will do nothing
        # the cost info will only be added if the child class overrides these methods
        for strategy in strategies:
            self.update_communication_cost(strategy)
            self.update_compute_cost(strategy)
            self.update_memory_cost(strategy)

        return strategies

    @abstractmethod
    def collate_strategies(self) -> List[ShardingStrategy]:
        pass

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate if the operands are of desired shape.
        If True, means this generator can be used for the current operation.
        """
        pass


class FollowingStrategyGenerator(StrategyGenerator):
    """
    FollowingStrategyGenerator is used to generate the sharding strategies which depends on its predecessor node.

    TODO: remove the original strategy_generator.py after refactoring
    """

    def __init__(self, operation_data_mapping: Dict[str, OperationData], device_mesh: DeviceMesh,
                 predecessor_node: Node):
        self.op_data = operation_data_mapping
        self.device_mesh = device_mesh
        self.predecessor_node = predecessor_node


class OutputStrategyGenerator(StrategyGenerator):
    """
    OutputStrategyGenerator is used to generate the sharding strategies for Output Node.
    """

    def __init__(self, operation_data_mapping: Dict[str, OperationData], device_mesh: DeviceMesh,
                 predecessor_nodes: List[Node]):
        self.op_data = operation_data_mapping
        self.device_mesh = device_mesh
        self.predecessor_nodes = predecessor_nodes
