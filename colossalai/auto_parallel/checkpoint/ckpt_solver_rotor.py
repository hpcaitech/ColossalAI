from copy import deepcopy
from typing import Dict, List, Tuple

from torch import Tensor
from torch.fx import Graph, Node

from colossalai.fx.codegen.activation_checkpoint_codegen import _find_nested_ckpt_regions
from colossalai.fx.profiler import (
    activation_size,
    calculate_bwd_time,
    calculate_fwd_out,
    calculate_fwd_time,
    calculate_fwd_tmp,
)
from colossalai.logging import get_dist_logger

from .ckpt_solver_base import CheckpointSolverBase
from .operation import Backward, Chain, ForwardCheck, ForwardEnable, ForwardNograd, Function, Loss, Sequence

__all__ = ['CheckpointSolverBase']


class CheckpointSolverRotor(CheckpointSolverBase):

    def __init__(self,
                 graph: Graph,
                 memory_budget: float = -1,
                 parameter_size: float = 0,
                 cnode: List[str] = None,
                 memory_slots: int = 500):
        """This is the simple implementation of dynamic programming algorithm rotor
        in https://hal.inria.fr/hal-02352969. Some code are adapted from
        https://gitlab.inria.fr/hiepacs/rotor.

        Usage:
            Assume that we have a `GraphModule`, and we already applied the `MetaInfoProp`
            to the graph to retrieve all information needed, then we could use the following
            code to find a solution using `CheckpointSolverRotor`:
            >>> solver = CheckpointSolverRotor(gm.graph, memory_budget=memory_budget, parameter_size=parameter_size)
            >>> rotor_graph = solver.solve(force_python=True)   # otherwise use C solver
            >>> gm.graph = rotor_graph    # set the graph to a new graph

        Args:
            graph (Graph): The computing graph to be optimized.
            memory_budget (float, optional): Memory constraint for the solution, unit is byte.
            parameter_size (float, optional): The size of parameter of this model, unit is byte. Use `parameter_size(model)` to estimate.
            cnode (List[str], optional): Common node List, should be the subset of input. Defaults to None.
            memory_slots (int, optional): Number of slots for discretizing memory budget. Defaults to 500.
        """
        super().__init__(graph, memory_budget, parameter_size, True, cnode)
        self.memory_slots = memory_slots

        # construct chain
        unit = self.memory_budget // self.memory_slots
        self.chain = self._construct_chain(self.graph, self.node_list)
        self.chain.discretize_all(unit)

        self.cost_table = None
        self.back_ptr = None
        self.sequence = None

    def solve(self, force_python: bool = False) -> Graph:
        """Solve the checkpointing problem using rotor algorithm.

        Args:
            force_python (bool, optional): Use Python version of solver, else use C version. Defaults to False.

        Returns:
            graph (Graph): The optimized graph, should be a copy of the original graph.
        """
        chain = self.chain

        # compute cost table
        if force_python:
            self.cost_table, self.back_ptr = self._compute_table(chain, self.memory_slots)
        else:
            self.cost_table, self.back_ptr = self._compute_table_c(chain, self.memory_slots)

        # backtrack
        try:
            self.sequence = self._backtrack(chain, 0, chain.length, self.memory_slots, self.cost_table, self.back_ptr)
            self._annotate_from_sequence(self.sequence, self.node_list)
        except RuntimeError as e:
            # using logger to annonce that the solver is failed
            logger = get_dist_logger()
            logger.warning(f'Checkpoint solver failed: {e}')

        return deepcopy(self.graph)

    def print_chain(self):
        print('[input]', self.chain.x[0], self.chain.xbar[0], self.chain.ftmp[0], self.chain.btmp[0])
        for idx in range(len(self.node_list) - 1):
            print(self.node_list[idx], self.chain.x[idx + 1], self.chain.xbar[idx + 1], self.chain.ftmp[idx],
                  self.chain.btmp[idx])
        print(f'Chain = {self.chain}')

    def print_sequence(self):
        print(f'Sequence = {self.sequence}')

    @classmethod
    def _construct_chain(cls, graph: Graph, node_list: List[List[Node]]) -> Chain:
        input_tensors = cls._extract_input(graph)
        fwd_time, bwd_time, ftmp, btmp = list(), list(), list(), list()
        xbar, x = [activation_size(input_tensors)], [activation_size(input_tensors)]

        for idx, node in enumerate(node_list):
            node_info = cls._extract_node_info(node)
            fwd_time.append(node_info[0])
            bwd_time.append(node_info[1])
            x.append(node_info[2])
            xbar.append(node_info[3])
            ftmp.append(node_info[4])
            btmp.append(node_info[5])

        # currently we view loss backward temp as zero
        bwd_time.append(0)
        btmp.append(0)

        return Chain(fwd_time, bwd_time, x, xbar, ftmp, btmp)

    @classmethod
    def _extract_node_info(cls, node: List[Node]) -> Tuple[int, ...]:
        """Extract node info from a list of nodes"""
        xbar = 0
        fwd_time = 0
        bwd_time = 0
        for n in node:
            assert isinstance(n, Node), f'{n} is not a Node'
            xbar += calculate_fwd_tmp(n) + calculate_fwd_out(n)
            # minimum flop count is required
            fwd_time += max(calculate_fwd_time(n), 1.0)
            bwd_time += max(calculate_bwd_time(n), 1.0)

        x = calculate_fwd_out(node[-1])
        xbar = max(x, xbar)
        ftmp = cls._extract_ftmp(node)
        btmp = cls._extract_btmp(node)
        return fwd_time, bwd_time, x, xbar, ftmp, btmp

    @staticmethod
    def _extract_input(graph: Graph) -> Tuple[Tensor, ...]:
        """Extract input tensors from a Graph"""
        input_tensors = []
        for node in graph.nodes:
            if node.op == 'placeholder':
                input_tensors.append(node.meta['fwd_out'])
        return input_tensors

    @staticmethod
    def _extract_ftmp(node: List[Node]) -> int:
        """Extract ftmp from a list of nodes"""
        n = node[-1]
        return activation_size(n.meta['fwd_out']) - calculate_fwd_out(n)

    @staticmethod
    def _extract_btmp(node: List[Node]) -> int:
        """Extract btmp from a list of nodes"""

        def _extract_deps_size():
            deps_size = 0
            for k, v in deps.items():
                k: Node
                if v > 0:
                    deps_size += k.meta['bwd_mem_out']
                if v == float('-inf'):
                    deps_size -= calculate_fwd_tmp(k) + calculate_fwd_out(k)

            return deps_size

        btmp = 0
        deps = {}
        for n in reversed(node):
            deps[n] = len(n.all_input_nodes)
            btmp = max(btmp, _extract_deps_size() + n.meta['bwd_mem_tmp'])
            for child in n.users:
                if child in deps:
                    deps[child] -= 1
                    if deps[child] <= 0:
                        deps[child] = float('-inf')    # free
        return btmp

    @staticmethod
    def _compute_table(chain: Chain, mem_slots: int) -> Tuple:
        """Compute the table using dynamic programming. Returns the cost table and the backtracking pointer.

        Args:
            chain (Chain): A basic linearized structure for solving the dynamic programming problem.
            mem_slots (int): Number of slots for discretizing memory budget.

        Returns:
            cost_table (List[List[Dict[int, Tuple]]]): cost_table[m][lmin][lmax] with lmin = 0...chain.length
                                     and lmax = lmin...chain.length (lmax is not included) and m = 0...mmax
            back_ptr (List[List[Dict[int, Tuple]]]): back_ptr[m][lmin][lmax] is (True,) if the optimal choice
                                     is a chain checkpoint (False, j) if the optimal choice is a leaf checkpoint
                                     of length j
        """

        ftime = chain.ftime + [0.0]
        btime = chain.btime
        x = chain.x + [0]
        xbar = chain.xbar + [0]
        ftmp = chain.ftmp + [0]
        btmp = chain.btmp + [0]

        # Build table
        cost_table = [[{} for _ in range(chain.length + 1)] for _ in range(mem_slots + 1)]
        back_ptr = [[{} for _ in range(chain.length + 1)] for _ in range(mem_slots + 1)]
        # Last one is a dict because its indices go from i to l. Renumbering will wait for C implementation

        # Initialize borders of the tables for lmax-lmin = 0
        for m in range(mem_slots + 1):
            for i in range(chain.length + 1):
                limit = max(x[i + 1] + xbar[i + 1] + ftmp[i], x[i + 1] + xbar[i + 1] + btmp[i])
                if m >= limit:    # Equation (1)
                    cost_table[m][i][i] = ftime[i] + btime[i]
                else:
                    cost_table[m][i][i] = float("inf")

        # Compute everything
        for m in range(mem_slots + 1):
            for d in range(1, chain.length + 1):
                for i in range(chain.length + 1 - d):
                    idx = i + d
                    mmin = x[idx + 1] + x[i + 1] + ftmp[i]
                    if idx > i + 1:
                        mmin = max(mmin, x[idx + 1] + max(x[j] + x[j + 1] + ftmp[j] for j in range(i + 1, idx)))
                    if m < mmin:
                        cost_table[m][i][idx] = float("inf")
                    else:
                        leaf_checkpoints = [(j,
                                             sum(ftime[i:j]) + cost_table[m - x[j]][j][idx] + cost_table[m][i][j - 1])
                                            for j in range(i + 1, idx + 1)
                                            if m >= x[j]]
                        if leaf_checkpoints:
                            best_leaf = min(leaf_checkpoints, key=lambda t: t[1])
                        else:
                            best_leaf = None
                        if m >= xbar[i + 1]:
                            chain_checkpoint = cost_table[m][i][i] + cost_table[m - xbar[i + 1]][i + 1][idx]
                        else:
                            chain_checkpoint = float("inf")
                        if best_leaf and best_leaf[1] <= chain_checkpoint:
                            cost_table[m][i][idx] = best_leaf[1]
                            back_ptr[m][i][idx] = (False, best_leaf[0])
                        else:
                            cost_table[m][i][idx] = chain_checkpoint
                            back_ptr[m][i][idx] = (True,)
        return cost_table, back_ptr

    @staticmethod
    def _compute_table_c(chain: Chain, mem_slots: int) -> Tuple:
        raise NotImplementedError("C implementation not available yet")

    def _backtrack(self, chain: Chain, lmin: int, lmax: int, mem_budget: int, cost_table: List[List[Dict[int, Tuple]]],
                   back_ptr: List[List[Dict[int, int]]]) -> List[int]:
        """Backtrack the cost table and retrieve the optimal checkpointing strategy.

        Args:
            chain (Chain): A basic linearized structure for solving the dynamic programming problem.
            lmin (int): The left index of the interval to backtrack.
            lmax (int): The right index of the interval to backtrack.
            mem_budget (int): The memory budget for processing this interval.
            cost_table (List[List[Dict[int, Tuple]]]): See _compute_table() for definitions
            back_ptr (List[List[Dict[int, Tuple]]]): See _compute_table() for definitions

        Raises:
            ValueError: Can not process the chain.

        Returns:
            sequence (Sequence): The sequence of executing nodes with checkpoints.
        """
        if mem_budget <= 0:
            raise ValueError(f"Can not process a chain with negative memory {mem_budget}")
        elif cost_table[mem_budget][lmin][lmax] == float("inf"):
            raise ValueError(f"Can not process this chain from index {lmin} to {lmax} with memory {mem_budget}")

        sequence = Sequence(Function("Persistent", lmax - lmin, mem_budget))
        if lmin == lmax:
            if lmin == chain.length:
                sequence.insert(Loss())
            else:
                sequence.insert(ForwardEnable(lmin))
                sequence.insert(Backward(lmin))
            return sequence

        if back_ptr[mem_budget][lmin][lmax][0]:
            sequence.insert(ForwardEnable(lmin))
            sequence.insert_sequence(
                self._backtrack(chain, lmin + 1, lmax, mem_budget - chain.xbar[lmin + 1], cost_table, back_ptr))
            sequence.insert(Backward(lmin))
        else:
            j = back_ptr[mem_budget][lmin][lmax][1]
            sequence.insert(ForwardCheck(lmin))
            for k in range(lmin + 1, j):
                sequence.insert(ForwardNograd(k))
            sequence.insert_sequence(self._backtrack(chain, j, lmax, mem_budget - chain.xbar[j], cost_table, back_ptr))
            sequence.insert_sequence(self._backtrack(chain, lmin, j - 1, mem_budget, cost_table, back_ptr))
        return sequence

    @staticmethod
    def _annotate_from_sequence(sequence: Sequence, node_list: List[List[Node]]):
        op_list = sequence.list_operations()
        loss_op = next(op for op in op_list if isinstance(op, Loss))
        fwd_list = op_list[:op_list.index(loss_op)]
        bwd_list = op_list[op_list.index(loss_op) + 1:]
        ckpt_idx = 0
        in_ckpt = False
        ckpt_region = []

        # forward annotation
        for idx, op in enumerate(fwd_list, 0):
            if in_ckpt:
                if isinstance(op, ForwardNograd):
                    ckpt_region.append(idx)

                elif isinstance(op, ForwardEnable):
                    in_ckpt = False
                    for node_idx in ckpt_region:
                        for n in node_list[node_idx]:
                            n.meta['activation_checkpoint'] = [ckpt_idx]

                    ckpt_idx += 1
                    ckpt_region = []

                elif isinstance(op, ForwardCheck):
                    for node_idx in ckpt_region:
                        for n in node_list[node_idx]:
                            n.meta['activation_checkpoint'] = [ckpt_idx]

                    ckpt_idx += 1
                    ckpt_region = [idx]

            else:
                if isinstance(op, ForwardCheck):
                    in_ckpt = True
                    ckpt_region.append(idx)

        # annotate the backward if there is any nested activation checkpoint
        in_recompute = False
        for op in bwd_list:
            if in_recompute:
                if isinstance(op, ForwardNograd):
                    ckpt_region.append(op.index)

                elif isinstance(op, ForwardEnable):
                    for node_idx in ckpt_region:
                        for n in node_list[node_idx]:
                            n.meta['activation_checkpoint'].append(ckpt_idx)

                    ckpt_idx += 1
                    ckpt_region = []

                elif isinstance(op, ForwardCheck):
                    for node_idx in ckpt_region:
                        for n in node_list[node_idx]:
                            n.meta['activation_checkpoint'].append(ckpt_idx)

                    ckpt_idx += 1
                    ckpt_region = [op.index]

                elif isinstance(op, Backward):
                    for node_idx in ckpt_region:
                        for n in node_list[node_idx]:
                            n.meta['activation_checkpoint'].append(ckpt_idx)

                    in_recompute = False

            else:
                if not isinstance(op, Backward):
                    in_recompute = True
                    ckpt_idx = 0
                    ckpt_region = []
                    if isinstance(op, ForwardCheck):
                        ckpt_region.append(op.index)

        # postprocess, make sure every activation checkpoint label in the
        # same activation checkpoint region (level = 0) has the same length
        op_list = []
        for node in node_list:
            op_list += node
        ckpt_regions = _find_nested_ckpt_regions(op_list)
        for (start_idx, end_idx) in ckpt_regions:
            nested_length = max(
                len(op_list[idx].meta['activation_checkpoint']) for idx in range(start_idx, end_idx + 1))
            for idx in range(start_idx, end_idx + 1):
                op_list[idx].meta['activation_checkpoint'] += [None] * (nested_length -
                                                                        len(op_list[idx].meta['activation_checkpoint']))
