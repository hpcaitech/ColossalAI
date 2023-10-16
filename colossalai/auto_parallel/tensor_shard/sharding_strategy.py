from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import torch
from torch.fx.node import Node

from colossalai.tensor.comm_spec import CommSpec
from colossalai.tensor.sharding_spec import ShardingSpec

from .constants import (
    ELEMENTWISE_FUNC_OP,
    ELEMENTWISE_METHOD_OP,
    ELEMENTWISE_MODULE_OP,
    RESHAPE_FUNC_OP,
    RESHAPE_METHOD_OP,
)

__all__ = ["OperationDataType", "OperationData", "TrainCycleItem", "MemoryCost", "ShardingStrategy", "StrategiesVector"]


class OperationDataType(Enum):
    """
    An operation can come from the argument list of an operator or the parameter list of a module.
    """

    INPUT = 0
    ARG = 1
    PARAM = 2
    BUFFER = 3
    OUTPUT = 4


@dataclass
class OperationData:
    """
    OperationData is the data related to an operator, the data can be the operand or the output.

    Args:
        name (str): the name of the operation-related data
        type (OperationDataType): the type of the operation data
        data (Any): the value for this data, usually it is a meta tensor.
        logical_shape (Tuple[int]): the logical shape of the data, it can be different from the its actual shape in memory.
    """

    name: str
    type: OperationDataType
    data: Any
    logical_shape: Tuple[int] = None

    def __post_init__(self):
        # if no logical shape is specified, use the data shape as the logical shape
        if self.logical_shape is None:

            def _infer_logical_shape(data: any):
                """
                This function is used to infer the logical shape of the data.
                """
                if isinstance(data, torch.Tensor):
                    return data.shape
                elif isinstance(data, torch.Size):
                    return None
                elif isinstance(data, (tuple, list)):
                    data_type = type(data)
                    return data_type([_infer_logical_shape(d) for d in data])
                else:
                    return None

            self.logical_shape = _infer_logical_shape(self.data)

    def __repr__(self) -> str:
        return f"OperationData(name={self.name}, type={self.type})"

    def __eq__(self, other) -> bool:
        return other.name == self.name

    def __hash__(self) -> int:
        return hash(f"{self.name}")


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


@dataclass
class MemoryCost:
    """
    MemoryCost is a dataclass which stores the memory usage in the program.

    Args:
        activation (int): the memory cost incurred by the activations in bytes.
        parameter (int): the memory cost incurred by the module parameter in bytes.
        temp (int): the memory cost incurred by the temporary tensors in bytes.
        buffer (int): the memory cost incurred by the module buffer in bytes.
    """

    activation: int = 0
    parameter: int = 0
    temp: int = 0
    buffer: int = 0


class CommType(Enum):
    """
    CommType describes the sequential order of a communication action and a computation action.

    Meaning:
        BEFORE: the communication action happens just before the computation operation.
        AFTER: the communication action happens after the computation operation.
        HOOK: the communication action is used to do the grad all reduce.
        IMPLICIT: the communication action happens during the kernel execution, such as SyncBatchNorm
    """

    BEFORE = 0
    AFTER = 1
    HOOK = 2
    IMPLICIT = 3


@dataclass
class CommAction:
    """
    CommAction is used to record the communication action.

    Args:
        comm_spec: express the communication pattern and the process groups to execute the communication action.
        comm_type: describes the sequential order of a communication action and a computation action.
        arg_index: record the location of tensor which join the communication, we cannot use name of node or op_data at runtime,
                   because the args of node may be changed by graph transform passes.
    """

    comm_spec: CommSpec = None
    comm_type: CommType = None
    arg_index: int = -1
    key_for_kwarg: any = None


@dataclass
class ShardingStrategy:
    """
    ShardingStrategy is a dataclass to store the meta information on tensor sharding for a node.

    Args:
        name (str): express the sharding strategies in string, such as 'S0S1 = S0R x RS1'.
        output_sharding_spec (ShardingSpec): ShardingSpec of the output node.
        compute_cost (TrainCycleItem): Computation cost to complete this strategy. (default to None)
        communication_cost (TrainCycleItem): Communication cost to complete this strategy. (default to None)
        memory_cost (TrainCycleItem): Memory cost of the output node using this strategy. (default to None)
        input_sharding_specs (List(ShardingSpec)): The ShardingSpecs of the input nodes.
    """

    name: str
    sharding_specs: Dict[OperationData, Union[ShardingSpec, Tuple[ShardingSpec]]] = None
    compute_cost: TrainCycleItem = None
    communication_cost: TrainCycleItem = None
    memory_cost: TrainCycleItem = None
    communication_actions: Dict[OperationData, CommAction] = None
    resharding_costs: Dict[Node, List[TrainCycleItem]] = None

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

    def get_op_data_by_name(self, name: str):
        for op_data in self.sharding_specs.keys():
            if op_data.name == name:
                return op_data
        raise KeyError(f"Could not find the OperationData with name {name}")

    def get_sharding_spec_by_name(self, name: str):
        for op_data, sharding_spec in self.sharding_specs.items():
            if op_data.name == name:
                return sharding_spec
        raise KeyError(f"Could not find the ShardingSpec for OperationData with name {name}")

    def clone(self):
        def _deepcopy_dict_vals(data: Dict):
            return {k: deepcopy(v) for k, v in data.items()}

        sharding_specs = _deepcopy_dict_vals(self.sharding_specs) if self.sharding_specs is not None else None
        # We need to deepcopy it when self.communication_actions is not None, instead of checking its __bool__ value.
        # Consider the examples below:
        # If self.communication_actions is an empty dictionary {}, then self.communication_actions is not None, but its __bool__ value is False.
        # In this case, if we set None to the new object, program will crash when we try to access the communication_actions.items.
        communication_actions = (
            _deepcopy_dict_vals(self.communication_actions) if self.communication_actions is not None else None
        )
        # same reason as communication_actions
        resharding_costs = _deepcopy_dict_vals(self.resharding_costs) if self.resharding_costs is not None else None
        compute_cost = deepcopy(self.compute_cost)
        communication_cost = deepcopy(self.communication_cost)
        memory_cost = deepcopy(self.memory_cost)

        return ShardingStrategy(
            name=self.name,
            sharding_specs=sharding_specs,
            compute_cost=compute_cost,
            communication_cost=communication_cost,
            memory_cost=memory_cost,
            communication_actions=communication_actions,
            resharding_costs=resharding_costs,
        )


class StrategiesVector(list):
    """
    Each node in fx graph will have a corresponding StrategiesVector, to store all the possible
    strategies of the node.

    Argument:
        node (Node): node for which the list of sharding strategies are generated.
    """

    def __init__(self, node: Node):
        super().__init__()
        self.node = node
        # fetch its input and output nodes
        # TODO: placeholder input nodes
        self.predecessor_nodes = list(node._input_nodes.keys())
        self.successor_nodes = list(node.users.keys())

    def check_merge(self):
        merge_label = False
        if self.node.op == "call_module":
            target = self.node.target
            root_module = self.node.graph.owning_module
            submod = root_module.get_submodule(target)
            submod_type = type(submod)
            # merge elementwise module node into source nodes
            # we could merge element-wise op, because the output sharding spec is always same as the input sharding spec.
            if submod_type in ELEMENTWISE_MODULE_OP:
                merge_label = True

        if self.node.op == "call_function":
            # we could merge element-wise op, because the output sharding spec is always same as the input sharding spec.
            if self.node.target in ELEMENTWISE_FUNC_OP:
                merge_label = True
            # we could merge bcast op if the rhs is a scalar, because it will fall back to the element-wise case.
            # TODO: remove this after we support the fall back logic.
            # if self.node.target in BCAST_FUNC_OP and len(self.predecessor_nodes) == 1:
            #     merge_label = True
            # we could merge reshape op, because their computation costs are negligible.
            if self.node.target in RESHAPE_FUNC_OP:
                merge_label = True

        if self.node.op == "call_method":
            # we could merge reshape op, because their computation costs are negligible.
            method = getattr(self.node.args[0]._meta_data.__class__, self.node.target)
            if method in RESHAPE_METHOD_OP:
                merge_label = True
            if method in ELEMENTWISE_METHOD_OP:
                merge_label = True
        return merge_label
