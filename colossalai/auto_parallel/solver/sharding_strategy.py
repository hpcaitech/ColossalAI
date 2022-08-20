from dataclasses import dataclass
from colossalai.tensor.sharding_spec import ShardingSpec
from typing import Dict, List


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


class StrategiesVector:
    '''
    Each node in fx graph will have a corresponding StrategiesVector, to store all the possible
    strategies of the node.

    Argument:
        node(Node): node to build corresponding strategies_vector.
        in_nodes(List[Node]): input nodes in the argument list of the node.
        following_nodes(List[Node]): the nodes take the target node as their argument.
        strategies(List[ShardingStrategy]): enumerate all the possible sharding strategies of the node.
    '''

    def __init__(self, node, in_nodes, following_nodes=None, strategies=[]):
        self.node = node
        self.in_nodes = in_nodes
        self.following_nodes = following_nodes
        self.strategies = strategies

    def check_merge(self):
        pass
