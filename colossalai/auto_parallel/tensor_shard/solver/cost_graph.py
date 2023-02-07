import torch

from colossalai.auto_parallel.tensor_shard.constants import INFINITY_COST


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

    def __init__(self, leaf_strategies, simplify=True, forward_only=False):
        self.leaf_strategies = leaf_strategies
        self.nodes = [strategies_vector.node for strategies_vector in self.leaf_strategies]
        # stores number of strategies in each node
        self.node_lens = {strategies_vector.node: len(strategies_vector) for strategies_vector in self.leaf_strategies}
        # extra_node_costs will store the extra costs introduced by merging nodes
        self.extra_node_costs = {}
        self.following_dict = {}
        self.simplify = simplify
        self.forward_only = forward_only
        self._build_cost_graph()

    def _remove_invalid_node(self, node, attr_name):
        remove_list = []
        target_node_list = getattr(node, attr_name, [])
        for target_node in target_node_list:
            if target_node not in self.nodes:
                remove_list.append(target_node)
        for element in remove_list:
            target_node_list.remove(element)

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
                if src_node not in self.nodes:
                    continue
                node_pair = (src_node, dst_node)
                edge_cost = {}
                for i in range(len(strategies_vector)):
                    for j in range(len(src_node.strategies_vector)):
                        resharding_cost_item = strategies_vector[i].resharding_costs[src_node][j]
                        if self.forward_only:
                            edge_cost[(j, i)] = resharding_cost_item.fwd
                        else:
                            edge_cost[(j, i)] = resharding_cost_item.total
                self.edge_costs[node_pair] = edge_cost
            parent_nodes = []
            children_nodes = []

            def _check_tensor_in_node(data):
                """
                This method is used to check whether the data has a tensor inside or not.
                """
                has_tensor_flag = False
                if isinstance(data, torch.Tensor):
                    return True
                elif isinstance(data, (tuple, list)):
                    for d in data:
                        has_tensor_flag = has_tensor_flag or _check_tensor_in_node(d)
                return has_tensor_flag

            for node in strategies_vector.predecessor_nodes:
                if _check_tensor_in_node(node._meta_data):
                    parent_nodes.append(node)
            for node in strategies_vector.successor_nodes:
                if _check_tensor_in_node(node._meta_data):
                    children_nodes.append(node)

            setattr(dst_node, 'parents', parent_nodes)
            setattr(dst_node, 'children', children_nodes)

            if self.simplify and strategies_vector.check_merge():
                for followed_node in strategies_vector.predecessor_nodes:
                    # we only merge node pairs which src node has a tensor element inside.
                    # This is necessay because the node without a tensor element inside will not
                    # be assigned any strategy.
                    if _check_tensor_in_node(followed_node._meta_data):
                        self.merge_pair.append((followed_node, dst_node))

    def get_edge_cost(self, src_node, dst_node):
        return self.edge_costs[(src_node, dst_node)]

    def merge_node(self, src_node, dst_node):
        '''
        To merge dst_node into src_node, we need to do it in following steps:

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
        # build merge_map
        merge_map = {}
        for src_index, _ in enumerate(src_node.strategies_vector):
            min_cost = INFINITY_COST
            lowest_cost_index = -1
            for dst_index, dst_strategy in enumerate(dst_node.strategies_vector):
                resharding_cost_item = dst_strategy.resharding_costs[src_node][src_index]
                if self.forward_only:
                    resharding_cost = resharding_cost_item.fwd
                else:
                    resharding_cost = resharding_cost_item.total
                if resharding_cost <= min_cost:
                    min_cost = resharding_cost
                    lowest_cost_index = dst_index
            merge_map[src_index] = lowest_cost_index

        # extra_node_cost for src node
        self.extra_node_costs[src_node] = [0.0] * self.node_lens[src_node]
        for src_index, strategy in enumerate(src_node.strategies_vector):
            target_strate_index = merge_map[src_index]
            target_strategy = dst_node.strategies_vector[target_strate_index]
            resharding_cost_item = target_strategy.resharding_costs[src_node][src_index]
            if self.forward_only:
                resharding_cost_to_add = resharding_cost_item.fwd
            else:
                resharding_cost_to_add = resharding_cost_item.total
            self.extra_node_costs[src_node][src_index] += resharding_cost_to_add
            if dst_node in self.extra_node_costs:
                self.extra_node_costs[src_node][src_index] += self.extra_node_costs[dst_node][target_strate_index]

        # add new node pair to cost graph
        for child_node in dst_node.children:
            new_node_pair = (src_node, child_node)
            old_node_pair = (dst_node, child_node)
            if new_node_pair in self.edge_costs:
                continue
            edge_cost = {}
            for i in range(self.node_lens[src_node]):
                for j in range(self.node_lens[child_node]):
                    dst_strate_index = merge_map[i]
                    edge_cost[(i, j)] = self.edge_costs[old_node_pair][(dst_strate_index, j)]
            if new_node_pair not in self.edge_costs:
                self.edge_costs[new_node_pair] = edge_cost
            else:
                # we should accumulate the resharding costs if args of child node contain
                # both src node and dst node.
                for index_pair, resharding_cost in self.edge_costs[new_node_pair]:
                    self.edge_costs[new_node_pair][index_pair] += edge_cost[index_pair]

        # connect src node and children of dst node
        dst_node.parents.remove(src_node)
        src_node.children.remove(dst_node)
        self.edge_costs.pop((src_node, dst_node))
        for child_node in dst_node.children:
            if child_node not in src_node.children:
                src_node.children.append(child_node)
            if src_node not in child_node.parents:
                child_node.parents.append(src_node)
            # remove dst node from cost graph when dst node has no producer.
            if len(dst_node.parents) == 0:
                child_node.parents.remove(dst_node)
                node_pair = (dst_node, child_node)
                self.edge_costs.pop(node_pair)
        if len(dst_node.parents) == 0:
            self.following_dict[dst_node] = src_node
            dst_node.children = []

    def _reindexing_src(self, src):
        if src not in self.following_dict:
            return src
        return self._reindexing_src(self.following_dict[src])

    def simplify_graph(self):
        if not self.simplify:
            return
        self.merge_pair.reverse()
        for (src_node, dst_node) in self.merge_pair:
            self.merge_node(src_node, dst_node)
        self.merge_pair.reverse()
        reindexing_following_dict = {}
        for dst, src in self.following_dict.items():
            reindexing_following_dict[dst] = self._reindexing_src(src)
        self.following_dict = reindexing_following_dict
