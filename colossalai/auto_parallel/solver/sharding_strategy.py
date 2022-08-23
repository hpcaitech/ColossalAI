from dataclasses import dataclass
from colossalai.tensor.sharding_spec import ShardingSpec
from typing import Dict, List
from torch.fx.node import Node

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
    output_sharding_spec: ShardingSpec
    compute_cost: float = 0.
    communication_cost: float = 0.
    memory_cost: float = 0.
    resharding_costs: Dict[int, List[float]] = None
    input_shardings: ShardingSpec = None


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
        self.predecessor_nodes = list(node._input_nodes.keys())
        self.successor_ndoes = list(node.users.keys())

    def check_merge(self):
        pass
