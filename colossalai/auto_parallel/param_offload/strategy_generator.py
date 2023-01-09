from typing import List
from torch.fx import Graph, Node

from colossalai.auto_parallel.param_offload.offload_strategy import OffloadStrategy, SystemConfig

class StrategyGenerator:
    """
    StrategyGenerator is used to generate the offload strategies.
    """

    def __init__(self, node: Node, graph: Graph):
        self.node = node
        self.graph = graph
        self.nodes = list(self.graph.nodes)
        self.node_idx = self.nodes.index(node)

    def collate_strategies(self) -> List[OffloadStrategy]:

        # currently have only one strategy

        strategies = []

        comm_cost = self.node.node_info.param_size / SystemConfig.BANDWIDTH
        strategies.append(OffloadStrategy(comm_cost=comm_cost))

        return strategies

    def update_reuse_interval(self, strategy: OffloadStrategy):
        reuse_interval = 0
        for following_node in self.nodes[self.node_idx:]:
            reuse_interval += following_node.meta.get('fwd_flop', 0) / SystemConfig.COMPUTE_POWER
            reuse_interval += following_node.meta.get('bwd_flop', 0) / SystemConfig.COMPUTE_POWER
            if hasattr(following_node, 'node_info') and following_node.node_info.offload_param_flag:
                reuse_interval += following_node.node_info.param_size / SystemConfig.BANDWIDTH

        strategy.reuse_interval = reuse_interval

    def update_strategies(self, strategies: List[OffloadStrategy]):
        for strategy in strategies:
            self.update_reuse_interval(strategy)

    def generate(self) -> List[OffloadStrategy]:
        """
        Generate all possible sharding strategies for this operation.
        """
        strategies = self.collate_strategies()

        # update the costs
        self.update_strategies(strategies)
        return strategies

