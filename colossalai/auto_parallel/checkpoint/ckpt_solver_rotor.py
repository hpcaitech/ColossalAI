from copy import deepcopy
from typing import Any, Dict, List, Tuple

from torch import Tensor
from torch.fx import Graph, Node

from colossalai.auto_parallel.passes.runtime_apply_pass import runtime_apply, runtime_comm_spec_apply
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
from .operation import Backward, Chain, ForwardCheck, ForwardEnable, ForwardNograd, Loss, Sequence

__all__ = ['CheckpointSolverRotor']


class CheckpointSolverRotor(CheckpointSolverBase):

    def __init__(self,
                 graph: Graph,
                 free_memory: float = -1,
                 cnode: List[str] = None,
                 memory_slots: int = 500,
                 optim_multiplier: float = 1.0):
        """This is the simple implementation of dynamic programming algorithm rotor
        in https://hal.inria.fr/hal-02352969. Some code are adapted from
        https://gitlab.inria.fr/hiepacs/rotor.

        Usage:
            Assume that we have a ``GraphModule``, and we have already done the extractions
            to the graph to retrieve all information needed, then we could use the following
            code to find a solution using ``CheckpointSolverRotor``:
            >>> solver = CheckpointSolverRotor(gm.graph, free_memory=torch.cuda.mem_get_info(device=0)[0])
            >>> rotor_graph = solver.solve(force_python=True)   # otherwise use C solver
            >>> gm.graph = rotor_graph    # set the graph to a new graph

        Args:
            graph (Graph): The computing graph to be optimized.
            free_memory (float, optional): Memory constraint for the solution, unit is byte.
                Use ``torch.cuda.mem_get_info(device=0)[0]`` to estimate the free_memory. Defaults to -1.
            cnode (List[str], optional): Common node List, should be the subset of input. Defaults to None.
            memory_slots (int, optional): Number of slots for discretizing memory budget. Defaults to 500.
            optim_multiplier (float, optional): The multiplier of extra weight storage for the
            ``torch.optim.Optimizer``. Default to 1.0.
        """
        super().__init__(graph, free_memory, True, cnode, optim_multiplier)
        self.memory_slots = memory_slots

        # construct chain
        unit = self.free_memory // self.memory_slots
        self.chain = self._construct_chain(self.graph, self.node_list)
        self.chain.discretize_all(unit)

        self.cost_table = None
        self.back_ptr = None
        self.sequence = None

    def solve(self, force_python: bool = False, verbose: bool = False) -> Graph:
        """Solve the checkpointing problem using rotor algorithm.

        Args:
            force_python (bool, optional): Use Python version of solver, else use C version. Defaults to False.
            verbose (bool, optional): Print verbose information. Defaults to False.

        Returns:
            graph (Graph): The optimized graph, should be a copy of the original graph.
        """
        chain = self.chain

        # compute cost table
        if force_python:
            self.cost_table, self.back_ptr = self._compute_table(chain, self.memory_slots)
        else:
            self.cost_table, self.back_ptr = self._compute_table_c(chain, self.memory_slots)

        if verbose:
            self.print_chain()

        # backtrack
        try:
            self.sequence = self._backtrack(chain, 0, len(chain), self.memory_slots - chain.x[0], self.cost_table,
                                            self.back_ptr)
            self._annotate_from_sequence(self.sequence, self.node_list)
        except ValueError as e:
            # using logger to annonce that the solver is failed
            logger = get_dist_logger()
            logger.warning(f'Checkpoint solver failed: {e}')
            raise ValueError

        if verbose:
            self.print_sequence()

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
        ftime, btime, ftmp, btmp = list(), list(), list(), list()
        xbar, x = [activation_size(input_tensors)], [activation_size(input_tensors)]

        for node in node_list:
            node_info = cls._extract_node_info(node)
            ftime.append(node_info[0])
            btime.append(node_info[1])
            x.append(node_info[2])
            xbar.append(node_info[3])
            ftmp.append(node_info[4])
            btmp.append(node_info[5])

        # currently we view loss backward temp as zero
        btime.append(0)
        btmp.append(0)

        return Chain(ftime, btime, x, xbar, ftmp, btmp)

    @classmethod
    def _extract_node_info(cls, node: List[Node]) -> Tuple[int, ...]:
        """Extract node info from a list of nodes"""
        xbar = 0
        ftime = 0
        btime = 0
        fwd_mem_peak = 0
        for n in node:
            assert isinstance(n, Node), f'{n} is not a Node'
            if n.target == runtime_apply or n.target == runtime_comm_spec_apply:
                # in this case we need to calculate memory usage directly based on the statics that hooked in node.meta
                xbar += n.meta['fwd_mem_out']
                fwd_mem_peak = max(fwd_mem_peak, xbar + n.meta['fwd_mem_tmp'])
            else:
                xbar += calculate_fwd_tmp(n) + calculate_fwd_out(n)
                fwd_mem_peak = max(fwd_mem_peak, xbar + n.meta['fwd_mem_tmp'] + cls._extract_unused_output(n))

            # minimum flop count is required
            ftime += max(calculate_fwd_time(n), 1.0)
            btime += max(calculate_bwd_time(n), 1.0)

        x = calculate_fwd_out(node[-1])
        xbar = max(x, xbar)
        ftmp = fwd_mem_peak - xbar
        btmp = cls._extract_btmp(node)
        return ftime, btime, x, xbar, ftmp, btmp

    @staticmethod
    def _extract_input(graph: Graph) -> Tuple[Tensor, ...]:
        """Extract input tensors from a Graph"""
        input_tensors = []
        for node in graph.nodes:
            if node.op == 'placeholder':
                input_tensors.append(node.meta['fwd_out'])
        return input_tensors

    @staticmethod
    def _extract_unused_output(node: Node) -> int:
        """Extract unused output from `torch.fx.Node`"""
        return activation_size(node.meta['fwd_out']) - calculate_fwd_out(node)

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
    def _compute_table(chain: Chain, mmax: int) -> Tuple:
        """Compute the table using dynamic programming. Returns the cost table and the backtracking pointer.

        Args:
            chain (Chain): A basic linearized structure for solving the dynamic programming problem.
            mmax (int): Maximum number of memory slots.

        Returns:
            cost_table (List): cost_table[m][lhs][rhs] with lhs = 0...chain.length
                                     and rhs = lhs...chain.length (lhs is not included) and m = 0...mmax
            back_ptr (List): back_ptr[m][lhs][rhs] is (True,) if the optimal choice
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
        cost_table = [[{} for _ in range(len(chain) + 1)] for _ in range(mmax + 1)]
        back_ptr = [[{} for _ in range(len(chain) + 1)] for _ in range(mmax + 1)]
        # Last one is a dict because its indices go from i to l. Renumbering will wait for C implementation

        # Initialize borders of the tables for lmax-lmin = 0
        for m in range(mmax + 1):
            for i in range(len(chain) + 1):
                limit = max(x[i + 1] + xbar[i + 1] + ftmp[i], x[i + 1] + xbar[i + 1] + btmp[i])
                if m >= limit:    # Equation (1)
                    cost_table[m][i][i] = ftime[i] + btime[i]
                else:
                    cost_table[m][i][i] = float("inf")

        # Compute everything
        for m in range(mmax + 1):
            for d in range(1, len(chain) + 1):
                for i in range(len(chain) + 1 - d):
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
    def _compute_table_c(chain: Chain, mmax: int) -> Tuple:
        try:
            from .rotorc import compute_table

        # build module if module not found
        except ModuleNotFoundError:
            import os
            import subprocess
            import sys
            logger = get_dist_logger()
            logger.info("rotorc hasn't been built! Building library...", ranks=[0])
            this_dir = os.path.dirname(os.path.abspath(__file__))
            result = subprocess.Popen(
                [
                    f"{sys.executable}", f"{os.path.join(this_dir, 'build_c_ext.py')}", "build_ext",
                    f"--build-lib={this_dir}"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.wait() == 0:
                logger.info("rotorc has been built!", ranks=[0])
                from .rotorc import compute_table
            else:
                logger.warning("rotorc built failed! Using python version!", ranks=[0])
                return CheckpointSolverRotor._compute_table(chain, mmax)
        return compute_table(chain, mmax)

    @staticmethod
    def _backtrack(chain: Chain, lhs: int, rhs: int, budget: int, cost_table: List[Any],
                   back_ptr: List[Any]) -> "Sequence":
        """Backtrack the cost table and retrieve the optimal checkpointing strategy.

        Args:
            chain (Chain): A basic linearized structure for solving the dynamic programming problem.
            lhs (int): The left index of the interval to backtrack.
            rhs (int): The right index of the interval to backtrack.
            budget (int): The memory budget for processing this interval.
            cost_table (List[Any]): See ``._compute_table()`` for definitions
            back_ptr (List[Any]): See ``._compute_table()`` for definitions

        Raises:
            ValueError: Can not process the chain.

        Returns:
            sequence (Sequence): The sequence of executing nodes with checkpoints.
        """
        if budget <= 0:
            raise ValueError(f"Can not process a chain with negative memory {budget}")
        elif cost_table[budget][lhs][rhs] == float("inf"):
            raise ValueError(f"Can not process this chain from index {lhs} to {rhs} with memory {budget}")

        sequence = Sequence()
        if rhs == lhs:
            if lhs == len(chain):
                sequence += [Loss()]
            else:
                sequence += [ForwardEnable(lhs), Backward(lhs)]
            return sequence

        if back_ptr[budget][lhs][rhs][0]:
            sequence += [
                ForwardEnable(lhs),
                CheckpointSolverRotor._backtrack(chain, lhs + 1, rhs, budget - chain.xbar[lhs + 1], cost_table,
                                                 back_ptr),
                Backward(lhs),
            ]
        else:
            best_leaf = back_ptr[budget][lhs][rhs][1]
            sequence += [ForwardCheck(lhs)]
            sequence += [ForwardNograd(k) for k in range(lhs + 1, best_leaf)]
            sequence += [
                CheckpointSolverRotor._backtrack(chain, best_leaf, rhs, budget - chain.x[best_leaf], cost_table,
                                                 back_ptr),
                CheckpointSolverRotor._backtrack(chain, lhs, best_leaf - 1, budget, cost_table, back_ptr),
            ]
        return sequence

    @staticmethod
    def _annotate_from_sequence(sequence: Sequence, node_list: List[List[Node]]):
        """Annotate the nodes in the ``node_list`` with activation checkpoint from the sequence.

        Args:
            sequence (Sequence): The sequence of executing nodes with activation checkpoint annotations.
            node_list (List[List[Node]]): The list of nodes to annotate.
        """
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
