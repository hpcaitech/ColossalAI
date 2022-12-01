import multiprocessing
import time
import warnings
from typing import Dict

import numpy as np
from torch.fx.graph import Graph
from torch.fx.node import Node

from colossalai.auto_parallel.tensor_shard.constants import INFINITY_COST

from .cost_graph import CostGraph
from .graph_analysis import GraphAnalyser
from .strategies_constructor import StrategiesConstructor

try:
    import pulp
    from pulp import LpMinimize, LpProblem, LpStatus, LpVariable, lpDot, lpSum
except:
    warnings.warn(f'please install the pulp')

__all___ = ['Solver']


class Solver:

    def __init__(self,
                 graph: Graph,
                 strategies_constructor: StrategiesConstructor,
                 cost_graph: CostGraph,
                 graph_analyser: GraphAnalyser,
                 memory_budget: float = -1.0,
                 solution_numbers: int = 1,
                 forward_only: bool = False,
                 memory_increasing_coefficient: float = 1.3,
                 verbose=True):
        '''
        Solver class will integrate information provided by the components and use ILP solver to find a possible optimal strategies combination for target computing graph.
        Argument:
            graph: The computing graph to be optimized.
            strategies_constructor: It will provide all the possible strategies for each node in the computing graph.
            cost_graph: A graph data structure to simplify the edge cost graph.
            graph_analyser: graph_analyser will analyse the graph to obtain the variable liveness information, which will be used to generate memory constraints.
            memory_budget: Memory constraint for the solution.
            solution_numbers: If solution_numbers is larger than one, solver will us a serious of solutions based on different memory budget.
            memory_increasing_coefficient: If solution_numbers is larger than one, we will use this coefficient to generate new memory budget.
        '''
        self.graph = graph
        self.strategies_constructor = strategies_constructor
        self.cost_graph = cost_graph
        self.graph_analyser = graph_analyser
        self.leaf_strategies = self.strategies_constructor.leaf_strategies
        self.nodes = [strategies_vector.node for strategies_vector in self.leaf_strategies]
        self.strategy_map = self.strategies_constructor.strategy_map
        self.memory_budget = memory_budget
        self.solution_numbers = solution_numbers
        self.forward_only = forward_only
        if self.solution_numbers > 1:
            self.memory_increasing_coefficient = memory_increasing_coefficient
        else:
            self.memory_increasing_coefficient = 1
        self.liveness_list = self.graph_analyser.liveness_analysis()
        self.node_index_dict = self._generate_node_index_dict()
        # The last solution vector of auto sharding.
        self.last_s_val = None
        # The last objective value of the best ILP solution.
        self.last_objective = None
        self.verbose = verbose

    def _recover_merged_node_strategy(self):
        '''
        During cost graph constructing, some nodes, such as unary element-wise node or ReshapeOp, were merged into the previous node.
        Therefore, the index of those strategies are copied from the previous node. This method is used to recover the strategy index of those merged
        node.
        '''
        for node_index, node in enumerate(self.nodes):
            if node.strategies_vector.check_merge():
                # the merged node has only one input, and its strategies follow the input sharding strategy
                input_strategies_vector = node.args[0].strategies_vector
                input_best_strategy_index = self.last_s_val[node_index - 1]
                input_sharding_spec = input_strategies_vector[input_best_strategy_index].output_sharding_spec
                for strategy_index, strategy in enumerate(node.strategies_vector):
                    if strategy.input_shardings[0].sharding_sequence == input_sharding_spec.sharding_sequence:
                        self.last_s_val[node_index] = strategy_index
                        break

    def _generate_node_index_dict(self) -> Dict[Node, int]:
        node_index_dict = {}
        for index, strategies_vector in enumerate(self.leaf_strategies):
            node_index_dict[strategies_vector.node] = index
        return node_index_dict

    def _prepare_data_for_solver(self):
        '''
        Extract information from components for solver.
        '''
        node_nums = len(self.leaf_strategies)
        memory_budget = self.memory_budget

        # prepare strategies_len
        strategies_len = []
        for node in self.nodes:
            strategies_len.append(self.cost_graph.node_lens[node])
        strategies_len = np.array(strategies_len)

        # prepare following_nodes
        following_nodes = self.cost_graph.following_dict
        index_following_nodes = {}
        for src, target in following_nodes.items():
            src_index = self.node_index_dict[src]
            target_index = self.node_index_dict[target]
            index_following_nodes[src_index] = target_index
        following_nodes = index_following_nodes
        for index in range(node_nums):
            if index not in following_nodes:
                following_nodes[index] = -1

        # prepare edge_pairs and resharding costs
        edge_pairs = []
        resharding_costs = []
        for pairs, edge_cost in self.cost_graph.edge_costs.items():
            src_node = pairs[0]
            dst_node = pairs[1]
            src_node_index = self.node_index_dict[src_node]
            dst_node_index = self.node_index_dict[dst_node]
            edge_pairs.append(src_node_index)
            edge_pairs.append(dst_node_index)

            for i in range(strategies_len[src_node_index]):
                for j in range(strategies_len[dst_node_index]):
                    resharding_costs.append(edge_cost[(i, j)])
        edge_pairs = np.array(edge_pairs)
        resharding_costs = np.array(resharding_costs)

        # prepare liveness_set
        liveness_set = self.liveness_list

        # omit alias_set now
        alias_set = None
        alias_convert_costs = None

        # prepare compute_costs, communication_costs and memory_costs
        compute_costs = []
        communication_costs = []
        memory_costs = []
        extra_node_costs = self.cost_graph.extra_node_costs
        for strategies_vector in self.leaf_strategies:
            node = strategies_vector.node
            for index, strategy in enumerate(strategies_vector):
                compute_cost_item = strategy.compute_cost
                communication_cost_item = strategy.communication_cost
                memory_cost_item = strategy.memory_cost

                if self.forward_only:
                    origin_communication_cost = communication_cost_item.fwd
                    compute_cost = compute_cost_item.fwd
                    # extract MemoryCost item from the memory TrainCycleItem
                    memory_cost = memory_cost_item.fwd
                else:
                    origin_communication_cost = communication_cost_item.total
                    compute_cost = compute_cost_item.total
                    # extract MemoryCost item from the memory TrainCycleItem
                    memory_cost = memory_cost_item.total

                # extract the memory cost in float from MemoryCost item and sum them up
                memory_cost = memory_cost.parameter + memory_cost.activation + memory_cost.buffer
                compute_costs.append(compute_cost)
                # node in extra_node_costs means it has some extra communication
                # cost from node merging, so we need to add those extra communication
                # cost into
                if node in extra_node_costs:
                    extra_node_cost = extra_node_costs[node][index]
                    communication_cost = origin_communication_cost + extra_node_cost
                    communication_costs.append(communication_cost)
                else:
                    communication_costs.append(origin_communication_cost)
                memory_costs.append(memory_cost)

        compute_costs = np.array(compute_costs)
        communication_costs = np.array(communication_costs)
        memory_costs = np.array(memory_costs)

        # omit initial value for nodes
        s_init_np = None

        return node_nums, memory_budget, strategies_len, following_nodes, edge_pairs, alias_set, liveness_set, compute_costs, communication_costs, memory_costs, resharding_costs, alias_convert_costs, s_init_np, self.verbose

    def _call_solver_serialized_args(self,
                                     node_nums,
                                     memory_budget,
                                     strategies_len,
                                     following_nodes,
                                     edge_pairs,
                                     alias_set,
                                     liveness_set,
                                     compute_costs,
                                     communication_costs,
                                     memory_costs,
                                     resharding_costs,
                                     alias_convert_costs,
                                     s_init_np=None,
                                     verbose=True):
        """
        Call the solver with serialized arguments.
        """

        tic = time.time()

        for x in [strategies_len, edge_pairs, compute_costs, communication_costs, memory_costs, resharding_costs]:
            assert isinstance(x, np.ndarray)
        assert len(strategies_len) == node_nums, "strategies_len"

        def get_non_zero_index(binary_vector):
            """
            Get the index of non-zero item in a vector.
            """
            ct = 0
            ret = None
            for i, elem in enumerate(binary_vector):
                if pulp.value(elem):
                    ret = i
                    ct += 1

            assert ct == 1
            return ret

        # 0. Unpack flatten numpy arrays
        s_follow = following_nodes

        E = edge_pairs.reshape((-1, 2))    # noqa
        r = []
        pt = 0
        edge_set = set()
        for (i, j) in E:
            prod_length = strategies_len[i] * strategies_len[j]

            if (i, j) in edge_set:
                raise ValueError(f"Duplicated edges: {(i, j)}")

            edge_set.add((i, j))
            r.append(resharding_costs[pt:pt + prod_length])
            pt += prod_length
        assert pt == len(resharding_costs)

        ######################
        # omit alias set now #
        ######################

        # A = alias_set.reshape((-1, 2))  # noqa
        # for (i, j) in A:
        #     prod_length = strategies_len[i] * strategies_len[j]
        #     v.append(alias_convert_costs[pt:pt + prod_length])
        #     pt += prod_length
        # assert pt == len(alias_convert_costs)

        # L = []  # noqa
        # pt = node_nums
        # for i in range(node_nums):
        #     length = liveness_set[i]
        #     L.append(liveness_set[pt:pt + length])
        #     pt += length
        # assert pt == len(liveness_set)
        v = []
        pt = 0

        c = []
        d = []
        m = []
        pt = 0
        for i in range(node_nums):
            length = strategies_len[i]
            c.append(compute_costs[pt:pt + length])
            d.append(communication_costs[pt:pt + length])
            m.append(memory_costs[pt:pt + length])
            pt += length
        assert pt == len(compute_costs), f"{pt} == {len(compute_costs)}"
        assert pt == len(communication_costs), f"{pt} == {len(communication_costs)}"
        assert pt == len(memory_costs), f"{pt} == {len(memory_costs)}"

        # 1. Create variables

        #############################
        # create variables for node #
        #############################
        s = []
        num_nodes = 0
        reverse_follow_backpatch = []
        for i in range(node_nums):
            if s_follow[i] < 0:
                if strategies_len[i] == 1:
                    s.append([1])
                else:
                    num_nodes += 1
                    s.append(LpVariable.matrix(f"s[{i}]", (range(strategies_len[i]),), cat="Binary"))
            else:
                if s_follow[i] < len(s):
                    s.append(s[s_follow[i]])
                else:
                    s.append(None)
                    reverse_follow_backpatch.append(i)

        for i in reverse_follow_backpatch:
            s[i] = s[s_follow[i]]

        #############################
        # create variables for edge #
        #############################
        e = []
        num_edges = 0
        for (idx, (i, j)) in enumerate(E):
            if len(s[i]) == 1:
                e.append(s[j])
            elif len(s[j]) == 1:
                e.append(s[i])
            else:
                num_edges += 1
                e.append(LpVariable.matrix(f"e[{i},{j}]", (range(len(s[i]) * len(s[j])),), cat="Binary"))
            assert len(e[idx]) == len(r[idx])
        for element in s:
            assert len(element) > 0
        # 2. Set initial value
        ######################################
        # set a initial value for warm start #
        ######################################
        if s_init_np is not None:
            s_init = s_init_np.reshape((-1, 3))
            for (idx, value, fix) in s_init:
                for i in range(len(s[idx])):
                    s[idx][i].setInitialValue(i == value)
                    if fix:
                        s[idx][i].fixValue()

        # 3. Objective
        prob = LpProblem("myProblem", LpMinimize)
        ###################################################################
        # computing the node cost(computing cost and communication cost)  #
        ###################################################################
        obj = 0
        for i in range(node_nums):
            assert len(s[i]) == len(c[i])
            assert len(s[i]) == len(d[i])

            obj += lpDot(s[i], c[i]) + lpDot(s[i], d[i])

        #############################################
        # computing the edge cost(resharding cost)  #
        #############################################
        for i in range(len(E)):
            assert len(e[i]) == len(r[i])
            obj += lpDot(e[i], r[i])

        prob += obj

        # 4. Constraints
        # (a). specified by `cat="Binary"`

        # (b)
        #################################################
        # make sure each node only choose one strategy  #
        #################################################
        for i in range(node_nums):
            if s_follow[i] < 0:
                prob += lpSum(s[i]) == 1

        # (c)
        #################################################
        # compute memory consumption with liveness set  #
        #################################################
        if memory_budget > 0:
            for liveness_stage in liveness_set:
                mem = 0
                for live_variable in liveness_stage.unique_live_vars:
                    if live_variable.node not in self.node_index_dict:
                        continue
                    node_index = self.node_index_dict[live_variable.node]
                    mem += lpSum(s[node_index][j] * m[node_index][j] for j in range(len(s[node_index])))
                prob += mem <= memory_budget

        # (d). specified by `cat="Binary"`

        for (idx, (i, j)) in enumerate(E):
            if strategies_len[i] == 1 or strategies_len[j] == 1:
                continue

            # (e)
            prob += lpSum(e[idx]) == 1

            # (f)
            for row in range(len(s[i])):
                C = len(s[j])    # noqa
                prob += lpSum(e[idx][row * C + col] for col in range(0, C)) <= s[i][row]

            # (g)
            for col in range(len(s[j])):
                R = len(s[i])    # noqa
                C = len(s[j])    # noqa
                prob += lpSum(e[idx][row * C + col] for row in range(0, R)) <= s[j][col]

        # (h)
        ######################
        # omit alias set now #
        ######################

        # alias_set = set()
        # for (idx, (i, j)) in enumerate(A):
        #     R = len(s[i])  # noqa
        #     C = len(s[j])  # noqa
        #     if (i, j) in alias_set:
        #         raise ValueError(f"Duplicated edges: {(i, j)}")

        #     alias_set.add((i, j))
        #     alias_set.add((j, i))

        #     for row in range(len(s[i])):
        #         for col in range(len(s[j])):
        #             if v[idx][row * C + col] > 0.5:
        #                 prob += s[i][row] + s[j][col] <= 1

        msg = verbose
        time_limit = 600
        assert "COIN_CMD" in pulp.listSolvers(
            onlyAvailable=True), ("Please install ILP solvers by 'sudo apt install coinor-cbc'")

        solver = pulp.COIN_CMD(mip=True, msg=msg, timeLimit=time_limit, threads=multiprocessing.cpu_count())
        # solver = pulp.GLPK_CMD(mip=True, msg=msg, timeLimit=time_limit)
        prob.solve(solver)

        status = prob.status
        objective = pulp.value(prob.objective)
        objective = float(objective) if objective is not None else -1.0
        if verbose:
            print(f"ILP Status: {LpStatus[status]}\tObjective: {objective}\t"
                  f"Time: {time.time() - tic}")
            print(f"#nodes: {num_nodes},  #edges: {num_edges}")

        if prob.status in [pulp.LpStatusInfeasible]:
            raise RuntimeError("Cannot run the function under the given memory budget. "
                               "Please increase the memory budget.")

        # Get and check results
        s_val = np.full((node_nums,), -1, dtype=np.int32)
        for i in range(node_nums):
            s_val[i] = get_non_zero_index(s[i])

        e_val = np.full((len(E),), -1, dtype=np.int32)
        for (idx, (i, j)) in enumerate(E):
            e_val[idx] = get_non_zero_index(e[idx])
            i_spec_index = e_val[idx] // len(s[j])
            j_spec_index = e_val[idx] % len(s[j])
            assert i_spec_index == s_val[i], f"e_val[{i}][{j}]"
            assert j_spec_index == s_val[j], f"e_val[{i}][{j}]"
            if verbose and r[idx][e_val[idx]] > 0:
                print(f"Edge cost {(i, j)} : {r[idx][e_val[idx]]}")

        self.last_s_val = list(s_val)
        # self._recover_merged_node_strategy()
        self.last_objective = objective

        if objective > INFINITY_COST:
            warnings.warn("Detect unexpected behaviors in the auto-sharding pass.")

        return self.last_s_val, e_val, self.last_objective, status

    def call_solver_serialized_args(self):
        """
        Call the solver with serialized arguments and handle python errors. Additionally,
        we could give a serious of solutions with different memory budget.
        """
        if self.solution_numbers == 1:
            args = self._prepare_data_for_solver()
            ret = self._call_solver_serialized_args(*args)

            return ret

        origin_memory_budget = self.memory_budget
        memory_budget_list = [
            origin_memory_budget * self.memory_increasing_coefficient**i for i in range(self.solution_numbers)
        ]
        ret_list = []
        for memory_budget in memory_budget_list:
            self.memory_budget = memory_budget
            args = self._prepare_data_for_solver()
            ret = self._call_solver_serialized_args(*args)
            ret_list.append(ret)

        return ret_list
