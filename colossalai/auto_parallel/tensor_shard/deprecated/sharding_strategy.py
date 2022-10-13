from copy import deepcopy
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import operator
import torch
from functools import reduce

from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.sharding_spec import ShardingSpec
from colossalai.tensor.shape_consistency import CollectiveCommPattern, CommSpec
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
        if self.node.op == 'output':
            self.predecessor_nodes = list(node._input_nodes.keys())[:1]
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
