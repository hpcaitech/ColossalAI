from colossalai.auto_parallel.solver.sharding_strategy import ShardingStrategy, StrategiesVector
from typing import List
from torch.fx.node import Node


class CostGraph:
    '''
    A graph data structure to simplify the edge cost graph. It has two main functions:
    1. To feed the quadratic resharding costs into solver, we need to linearize it. We build edge_cost in
    CostGraph, and it stored every combinations of strategies for a src-dst node pair in an 1D list.
    2. To reduce the searching space, we merge computationally-trivial operators, such as 
    element-wise operators, transpose, and reduction, into their following nodes. The merging infomation will
    be given by the StrategiesVector depending on the type of target node and following nodes.

    Argument:
        leaf_strategies(List[StrategiesVector]): It stores StrategiesVector of every nodes on the graph.
        simplify(bool, optional): The generated cost graph will be simplified if it is true. (default to True)
    '''

    def __init__(self, leaf_strategies, simplify=True):
        self.leaf_strategies = leaf_strategies
        # stores number of strategies in each node
        self.node_lens = {strategies_vector.node: len(strategies_vector) for strategies_vector in self.leaf_strategies}
        # extra_node_costs will store the extra costs introduced by merging nodes
        self.extra_node_costs = {}
        self.simplify = simplify
        self._build_cost_graph()

    def _build_cost_graph(self):
        '''
        This method will generate edge_cost for adjacent node pair. Additionally, 'parents' and 'children' attribute will be
        set to node.
        '''
        self.edge_costs = {}
        if self.simplify:
            self.merge_pair = []
        for strategies_vector in self.leaf_strategies:
            # build edge_cost
            dst_node = strategies_vector.node
            for src_node in strategies_vector.predecessor_nodes:
                node_pair = (src_node, dst_node)
                # src_index = strategies_vector.predecessor_nodes.index(src_node)
                edge_cost = {}
                for i in range(len(strategies_vector)):
                    for j in range(len(src_node.strategies_vector)):
                        edge_cost[(j, i)] = strategies_vector[i].resharding_costs[src_node][j]
                self.edge_costs[node_pair] = edge_cost
            # add parents and children attribute to node
            setattr(dst_node, 'parents', strategies_vector.predecessor_nodes)
            setattr(dst_node, 'children', strategies_vector.successor_nodes)

            if self.simplify and strategies_vector.check_merge():
                for following_node in strategies_vector.successor_nodes:
                    self.merge_pair.append((dst_node, following_node))

    def get_edge_cost(self, src_node, dst_node):
        return self.edge_costs[(src_node, dst_node)]

    def merge_node(self, src_node, dst_node):
        '''
        To merge src_node into dst_node, we need to do it in following steps:
        
        1. For each strategy in dst_node, we need to pick an appropriate strategy
        of src_node to merge, it is important because the logical resharding costs 
        between the parents node of src_node and merged node depend on the src_node 
        strategies dispatching. For example, for the graph 0->1->2, after merging node 1
        into node 2, edge_costs[(node 0, node 2)][(0, 0)] = edge_costs[(node 0, node 1)][(0, x)]
        x represents the picking strategy of node 1 merged into node 2 strategy 0.
        
        2. We need to accumulate the extra costs introduced by merging nodes, the extra costs
        contains two parts, one is resharding costs between src_node strategy and dst_node strategy,
        another is the origin extra costs in src_node strategy.

        3. Build connections between new node pairs, and remove the src_node after all consumer nodes
        detached from it.

        Argument:
            src_node(Node): The node will be merged into dst_node.
            dst_node(Node): The node to integrate src_node.
        '''
        src_node_index = dst_node.parents.index(src_node)
        # build merge_map
        merge_map = {}
        for dst_strate_index, strategy in enumerate(dst_node.strategies_vector):
            resharding_costs = strategy.resharding_costs
            resharding_cost_for_src = resharding_costs[src_node]
            lowest_cost_index = resharding_cost_for_src.index(min(resharding_cost_for_src))
            merge_map[dst_strate_index] = lowest_cost_index

        # extra_node_cost for dst node
        self.extra_node_costs[dst_node] = [0.0 for _ in range(self.node_lens[dst_node])]
        for dst_strate_index, strategy in enumerate(dst_node.strategies_vector):
            target_strate_index = merge_map[dst_strate_index]
            self.extra_node_costs[dst_node][dst_strate_index] += strategy.resharding_costs[src_node][
                target_strate_index]
            if src_node in self.extra_node_costs:
                self.extra_node_costs[dst_node][dst_strate_index] += self.extra_node_costs[src_node][
                    target_strate_index]

        # add new node pair to cost graph
        for parent_node in src_node.parents:
            new_node_pair = (parent_node, dst_node)
            old_node_pair = (parent_node, src_node)
            if new_node_pair in self.edge_costs:
                continue
            edge_cost = {}
            for i in range(self.node_lens[dst_node]):
                for j in range(self.node_lens[parent_node]):
                    src_strate_index = merge_map[i]
                    edge_cost[(j, i)] = self.edge_costs[old_node_pair][(j, src_strate_index)]
            self.edge_costs[new_node_pair] = edge_cost

        # connect dst node and parents of src node
        dst_node.parents.remove(src_node)
        src_node.children.remove(dst_node)
        self.edge_costs.pop((src_node, dst_node))
        for parent_node in src_node.parents:
            if parent_node not in dst_node.parents:
                dst_node.parents.append(parent_node)
            if dst_node not in parent_node.children:
                parent_node.children.append(dst_node)
            # remove src node from cost graph when src node has no consumer.
            if len(src_node.children) == 0:
                parent_node.children.remove(src_node)
                node_pair = (parent_node, src_node)
                self.edge_costs.pop(node_pair)

    def simplify_graph(self):
        if not self.simplify:
            return
        for (src_node, dst_node) in self.merge_pair:
            self.merge_node(src_node, dst_node)
