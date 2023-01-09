from dataclasses import dataclass
from torch.fx.node import Node
from colossalai.auto_parallel.tensor_shard.sharding_strategy import StrategiesVector


class OffloadStrategiesVector(StrategiesVector):
    '''
    Each node in fx graph will have a corresponding OffloadStrategiesVector, to store all the possible
    strategies of the node.

    Argument:
        node (Node): node for which the list of offload strategies are generated.
    '''

    def __init__(self, node: Node):
        super().__init__(node)


@dataclass
class OffloadStrategy:
    comm_cost: float = 0
    reuse_interval: float = 0
    # TODO Information required for asynchronous offload


class SystemConfig:
    BANDWIDTH = 1.2e9
    COMPUTE_POWER = 1.9e12
