from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import operator
import torch
from functools import reduce

from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.sharding_spec import ShardingSpec
from typing import Dict, List, Union, Tuple, Any
from torch.fx.node import Node
from .constants import *

__all__ = ['ShardingStrategy', 'StrategiesVector']


@dataclass
class ShardingStrategy:
    '''
    ShardingStrategy is a structure containing sharding strategies of inputs and output of this node
    and costs information using in solver.

    Argument:
        name(str): express the sharding strategies in string, such as 'S0S1 = S0R x RS1'.
        output_sharding_spec(ShardingSpec): ShardingSpec of the output node.
        compute_cost(float): Computation cost to complete this strategy.(default to 0)
        communication_cost(float): Communication cost to complete this strategy.(default to 0)
        memory_cost(float): Memory cost of the output node using this strategy.(default to 0)
        resharding_costs(Dict[int, List[float]]): resharding_cost[i][j] means the cost of i-th argument in the output node argument list
                                                  with j-th strategy in its strategies_vector transforms to sharding spec wanted in this
                                                  strategy.(default to None)
        input_shardings(List(ShardingSpec)): The ShardingSpecs of the input nodes.
    '''

    name: str
    # TODO: output of fx node,such as torch.var_mean, could be a tuple, so we cannot simply suppose it is a tensor.
    output_sharding_spec: Union[ShardingSpec, Tuple[ShardingSpec]]
    compute_cost: float = 0.
    communication_cost: float = 0.
    memory_cost: float = 0.
    resharding_costs: Dict[Node, List[float]] = None
    # sometimes the input node could be a tuple of nodes, but most of op won't accept tuple of node as input.
    # Therefore, we could process them at the specific op(operator.getitem)
    input_shardings: List[ShardingSpec] = None


class OperationDataType(Enum):
    """
    An operation can come from the argument list of an operator or the parameter list of a module.
    """
    ARG = 0
    PARAM = 1
    OUTPUT = 2


@dataclass
class OperationData:
    """
    OperationData is the data related to an operator, the data can be the operand or the output.

    Args:
        name (str): the name of the operation-related data
        type (OperationDataType): the type of the operation data
        data (torch.Tensor): the value for this data, usually it is a meta tensor.
        logical_shape (Tuple[int]): the logical shape of the data, it can be different from the its actual shape in memory.
    """
    name: str
    type: OperationDataType
    data: torch.Tensor
    logical_shape: Tuple[int] = None

    def __post_init__(self):
        # if no logical shape is specified, use the data shape as the logical shape
        if self.logical_shape is None:
            self.logical_shape = self.data.shape


@dataclass
class TrainCycleItem:
    """
    TrainCycleItem is a dataclass to store the items which have different values for the forward and backward pass
    in a training iteration.

    Args:
        fwd (float): the item for the forward pass
        bwd (float): the item for the backward pass
    """
    fwd: Any
    bwd: Any
    total: Any


class CommunicationType(Enum):
    FWD_ALL_REDUCE = 0
    BWD_ALL_REDUCE = 1


@dataclass
class CommunicationAction:
    """
    The actions 
    """
    type: CommunicationType
    mesh_dim: int


@dataclass
class ShardingStrategy_V2:
    """
    ShardingStrategy is a dataclass to store the meta information on tensor sharding for a node.

    Args:
        name (str): express the sharding strategies in string, such as 'S0S1 = S0R x RS1'.
        output_sharding_spec (ShardingSpec): ShardingSpec of the output node.
        compute_cost (TrainCycleItem): Computation cost to complete this strategy. (default to None)
        communication_cost (TrainCycleItem): Communication cost to complete this strategy. (default to None)
        memory_cost (TrainCycleItem): Memory cost of the output node using this strategy. (default to None)
        input_sharding_specs (List(ShardingSpec)): The ShardingSpecs of the input nodes.
        input_resharding_costs (Dict[int, List[float]]): resharding_cost[i][j] means the cost of i-th argument in the output node argument list
                                                  with j-th strategy in its strategies_vector transforms to sharding spec wanted in this
                                                  strategy.(default to None)
    """
    name: str
    sharding_specs: Dict[OperationData, ShardingSpec] = None
    compute_cost: TrainCycleItem = None
    communication_cost: TrainCycleItem = None
    memory_cost: TrainCycleItem = None
    input_resharding_costs: Dict[OperationData, List[float]] = None
    communication_actions: Dict[OperationData, List[CommunicationAction]] = None

    @property
    def input_sharding_specs(self) -> Dict[OperationData, ShardingSpec]:
        specs = {}
        specs.update(self._get_sharding_spec(OperationDataType.ARG))
        specs.update(self._get_sharding_spec(OperationDataType.PARAM))
        return specs

    @property
    def argument_sharding_specs(self) -> Dict[OperationData, ShardingSpec]:
        return self._get_sharding_spec(OperationDataType.ARG)

    @property
    def param_sharding_specs(self) -> Dict[OperationData, ShardingSpec]:
        return self._get_sharding_spec(OperationDataType.PARAM)

    @property
    def output_sharding_specs(self) -> Dict[OperationData, ShardingSpec]:
        return self._get_sharding_spec(OperationDataType.OUTPUT)

    def _get_sharding_spec(self, operation_data_type: OperationDataType):
        specs = {k: v for k, v in self.sharding_specs.items() if k.type == operation_data_type}
        return specs


class StrategyGenerator_V2(ABC):
    """
    StrategyGenerator is used to generate the same group of sharding strategies. 

    TODO: remove the original strategy_generator.py after refactoring
    """

    def __init__(self, device_mesh: DeviceMesh):
        self.device_mesh = device_mesh

    def update_communication_cost(self, strategy: ShardingStrategy_V2) -> ShardingStrategy_V2:
        """
        Compute the communication cost involved in the forward and backward iteration.
        """

        comm_cost = TrainCycleItem(fwd=0, bwd=0)

        def _compute_and_add(data: OperationData, action: CommunicationAction):
            sharded_shape = strategy.sharding_specs[data].get_sharded_shape_per_device()
            dtype = operand.data.dtype
            size_per_elem_bytes = torch.tensor([], dtype=dtype).element_size()
            num_bytes = size_per_elem_bytes * reduce(operator.mul, sharded_shape)
            cost = self.device_mesh.all_reduce_cost(num_bytes=num_bytes, mesh_dim=action.mesh_dim)

            # compute the fwd
            if action.type == CommunicationType.FWD_ALL_REDUCE:
                comm_cost.fwd += cost
            elif action.type == CommunicationType.BWD_ALL_REDUCE:
                comm_cost.fwd += cost
            else:
                raise ValueError(f"Found unknown CommunicationType {action.type}")

        # check if communication action exists
        # if so, loop over each action and compute the cost of each action
        if strategy.communication_actions is not None:
            for operand, actions in strategy.communication_actions:
                for action in actions:
                    _compute_and_add(operand, action)

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

    @abstractmethod
    def generate(self, operand_mapping: Dict[str, OperationData]) -> List[ShardingStrategy_V2]:
        """
        Generate all possible sharding strategies for this operation.
        """
        pass

    @abstractmethod
    def validate(self, *args, **kwargs) -> bool:
        """
        Validate if the operands are of desired shape. 
        If True, means this generator can be used for the current operation.
        """
        pass


class StrategiesVector(list):
    '''
    Each node in fx graph will have a corresponding StrategiesVector, to store all the possible
    strategies of the node.

    Argument:
        node (Node): node for which the list of sharding strategies are generated.
    '''

    def __init__(self, node: Node):
        super().__init__()
        self.node = node
        # fetch its input and output nodes
        # TODO: placeholder input nodes
        self.predecessor_nodes = list(node._input_nodes.keys())
        self.successor_nodes = list(node.users.keys())

    def check_merge(self):
        merge_label = False
        if self.node.op == 'call_module':
            target = self.node.target
            root_module = self.node.graph.owning_module
            submod = root_module.get_submodule(target)
            submod_type = type(submod)
            # merge elementwise module node into source nodes
            # we could merge element-wise op, because the output sharding spec is always same as the input sharding spec.
            if submod_type in ELEMENTWISE_MODULE_OP:
                merge_label = True

        if self.node.op == 'call_function':
            # we could merge element-wise op, because the output sharding spec is always same as the input sharding spec.
            if self.node.target in ELEMENTWISE_FUNC_OP:
                merge_label = True
            # we could merge bcast op if the rhs is a scalar, because it will fall back to the element-wise case.
            if self.node.target in BCAST_FUNC_OP and len(self.predecessor_nodes) == 1:
                merge_label = True
            # we could merge reshape op, because the output sharding spec of reshape op is always fully replicated.
            if self.node.target in RESHAPE_FUNC_OP:
                merge_label = True

        return merge_label
