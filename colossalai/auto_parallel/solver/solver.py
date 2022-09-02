import math
import pulp
import time
import numpy as np
import multiprocessing
from pulp import LpVariable, LpProblem, LpMinimize, lpSum, lpDot, LpStatus
from colossalai.auto_parallel.solver.constants import INFINITY_COST


class Solver:

    def __init__(self, graph, strategies_constructor, cost_graph, memory_budget=INFINITY_COST):
        self.graph = graph
        self.strategies_constructor = strategies_constructor
        self.cost_graph = cost_graph
        self.nodes = list(self.graph.nodes)
        self.leaf_strategies = self.strategies_constructor.leaf_strategies
        self.strategy_map = self.strategies_constructor.strategy_map
        self.memory_budget = memory_budget
        # The last solution vector of auto sharding.
        self.last_s_val = None
        # The last objective value of the best ILP solution.
        self.last_objective = None

    def _prepare_data_for_solver(self):

        node_index_dict = {}
        for index, strategies_vector in enumerate(self.leaf_strategies):
            node_index_dict[strategies_vector.node] = index
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
            src_index = node_index_dict[src]
            target_index = node_index_dict[target]
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
            src_node_index = node_index_dict[src_node]
            dst_node_index = node_index_dict[dst_node]
            edge_pairs.append(src_node_index)
            edge_pairs.append(dst_node_index)

            for i in range(strategies_len[src_node_index]):
                for j in range(strategies_len[dst_node_index]):
                    resharding_costs.append(edge_cost[(i, j)])
        edge_pairs = np.array(edge_pairs)
        resharding_costs = np.array(resharding_costs)

        # omit alias_set and liveness_set now
        alias_set = None
        liveness_set = None
        alias_convert_costs = None

        # prepare compute_costs, communication_costs and memory_costs
        compute_costs = []
        communication_costs = []
        memory_costs = []
        for strategies_vector in self.leaf_strategies:
            for strategy in strategies_vector:
                compute_costs.append(strategy.compute_cost)
                communication_costs.append(strategy.communication_cost)
                memory_costs.append(strategy.memory_cost)
        compute_costs = np.array(compute_costs)
        communication_costs = np.array(communication_costs)
        memory_costs = np.array(memory_costs)

        # omit initial value for nodes
        s_init_np = None

        return node_nums, memory_budget, strategies_len, following_nodes, edge_pairs, alias_set, liveness_set, compute_costs, communication_costs, memory_costs, resharding_costs, alias_convert_costs, s_init_np

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
                                     s_init_np=None):
        """Call the solver with serialized arguments."""

        tic = time.time()

        # for x in [strategies_len, edge_pairs, alias_set, liveness_set, compute_costs, communication_costs, memory_costs, resharding_costs, alias_convert_costs]:
        for x in [strategies_len, edge_pairs, compute_costs, communication_costs, memory_costs, resharding_costs]:
            assert isinstance(x, np.ndarray)
        assert len(strategies_len) == node_nums, "strategies_len"

        def get_non_zero_index(binary_vector):
            """Get the index of non-zero item in a vector."""
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
        s = []
        e = []

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

        # 2. Set initial value for warm start
        if s_init_np is not None:
            s_init = s_init_np.reshape((-1, 3))
            for (idx, value, fix) in s_init:
                for i in range(len(s[idx])):
                    s[idx][i].setInitialValue(i == value)
                    if fix:
                        s[idx][i].fixValue()

        # 3. Objective
        prob = LpProblem("myProblem", LpMinimize)
        # compute cost
        obj = 0
        for i in range(node_nums):
            obj += lpDot(s[i], c[i]) + lpDot(s[i], d[i])

        # communication cost
        for i in range(len(E)):
            obj += lpDot(e[i], r[i])

        prob += obj

        # 4. Constraints
        # (a). specified by `cat="Binary"`

        # (b)
        for i in range(node_nums):
            if s_follow[i] < 0:
                prob += lpSum(s[i]) == 1

        # (c)
        ########################################################
        # omit liveness set now, the memory budget is infinity #
        ########################################################

        # if memory_budget > 0:
        #     for t in range(node_nums):
        #         mem = 0
        #         for i in L[t]:
        #             mem += lpSum(s[i][j] * m[i][j] for j in range(len(s[i])))
        #         prob += mem <= memory_budget

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

        verbose = False

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

        self.last_s_val = s_val
        self.last_objective = objective

        if objective > INFINITY_COST:
            warnings.warn("Detect unexpected behaviors in the auto-sharding pass.")

        return s_val, e_val, objective, status

    def call_solver_serialized_args(self):
        """Call the solver with serialized arguments and handle python errors."""

        args = self._prepare_data_for_solver()
        ret = self._call_solver_serialized_args(*args)

        return ret
