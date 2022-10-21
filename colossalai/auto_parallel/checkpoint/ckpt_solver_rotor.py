import math
import sys
from typing import List, Tuple

from torch import Tensor
from torch.fx import Graph, Node

from colossalai.fx.codegen.activation_checkpoint_codegen import _find_nested_ckpt_regions
from colossalai.fx.graph_module import ColoGraphModule
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


class CheckpointSolverRotor(CheckpointSolverBase):

    def __init__(self,
                 graph: Graph,
                 memory_budget: float = -1,
                 parameter_size: float = 0,
                 cnode: List[str] = None,
                 mem_slots: int = 500):
        """This is the simple implementation of dynamic programming algorithm rotor in https://hal.inria.fr/hal-02352969
        Some code are adapted from https://gitlab.inria.fr/hiepacs/rotor

        Args:
            graph (Graph): The computing graph to be optimized.
            memory_budget (float, optional): Memory constraint for the solution.
            parameter_size (float, optional): The size of parameter of this model. Use `parameter_size(model)` to estimate.
            cnode (List[str], optional): Common node List, should be the subset of input. Defaults to None.
            mem_slots (int, optional): Number of slots for discretizing memory budget. Defaults to 500.
        """
        super().__init__(graph, memory_budget, parameter_size, True, cnode)
        self.chain = self._construct_chain(self.graph, self.node_list)
        self.mem_slots = mem_slots

    def solve(self, force_python: bool = False):
        # TODO: implement this
        raise NotImplementedError

    def print_chain(self):
        print('[input]', self.chain.x[0], self.chain.xbar[0], self.chain.ftmp[0], self.chain.btmp[0])
        for idx in range(len(self.node_list) - 1):
            print(self.node_list[idx], self.chain.x[idx + 1], self.chain.xbar[idx + 1], self.chain.ftmp[idx],
                  self.chain.btmp[idx])
        print(f'Chain = {self.chain}')

    @classmethod
    def _construct_chain(cls, graph: Graph, node_list: List[List[Node]]) -> Chain:
        input_tensors = cls._extract_input(graph)
        fwd_time, bwd_time, ftmp, btmp = list(), list(), list(), list()
        xbar, x = [activation_size(input_tensors)], [activation_size(input_tensors)]

        for node in enumerate(node_list):
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

    # this is the python compute table code from adapted from rotor
    @staticmethod
    def _compute_table(chain: Chain, mem_slots: int) -> Tuple:
        """Compute the table using dynamic programming. Returns the optimal table.

        Args:
            chain (Chain): A basic linearized structure for solving the dynamic programming problem.
            mem_slots (int): Number of slots for discretizing memory budget.
        Returns:
            opt (Dict[List[List[int]]]): opt[m][lmin][lmax] with lmin = 0...chain.length
                                     and lmax = lmin...chain.length (lmax is not included) and m = 0...mmax
            what (Dict[List[List[int]]]): what[m][lmin][lmax] is (True,) if the optimal choice is a chain checkpoint
                                     (False, j) if the optimal choice is a leaf checkpoint of length j
        """

        ftime = chain.ftime + [0.0]
        btime = chain.btime
        x = chain.x + [0]
        xbar = chain.xbar + [0]
        ftmp = chain.ftmp + [0]
        btmp = chain.btmp + [0]

        # Build table
        opt = [[{} for _ in range(chain.length + 1)] for _ in range(mem_slots + 1)]
        what = [[{} for _ in range(chain.length + 1)] for _ in range(mem_slots + 1)]
        # Last one is a dict because its indices go from i to l. Renumbering will wait for C implementation

        # Initialize borders of the tables for lmax-lmin = 0
        for m in range(mem_slots + 1):
            for i in range(chain.length + 1):
                #lmax-lmin = 0
                limit = max(x[i + 1] + xbar[i + 1] + ftmp[i], x[i + 1] + xbar[i + 1] + btmp[i])
                if m >= limit:    ## Equation (1)
                    opt[m][i][i] = ftime[i] + btime[i]
                else:
                    opt[m][i][i] = float("inf")

        # Compute everything
        for m in range(mem_slots + 1):
            for d in range(1, chain.length + 1):
                for i in range(chain.length + 1 - d):
                    # for idx in range(i+1, chain.length + 1):
                    idx = i + d
                    mmin = x[idx + 1] + x[i + 1] + ftmp[i]
                    if idx > i + 1:
                        mmin = max(mmin, x[idx + 1] + max(x[j] + x[j + 1] + ftmp[j] for j in range(i + 1, idx)))
                    if m < mmin:
                        opt[m][i][idx] = float("inf")
                    else:
                        leaf_checkpoints = [(j, sum(ftime[i:j]) + opt[m - x[j]][j][idx] + opt[m][i][j - 1])
                                            for j in range(i + 1, idx + 1)
                                            if m >= x[j]]
                        if leaf_checkpoints:
                            best_leaf = min(leaf_checkpoints, key=lambda t: t[1])
                        else:
                            best_leaf = None
                        if m >= xbar[i + 1]:
                            chain_checkpoint = opt[m][i][i] + opt[m - xbar[i + 1]][i + 1][idx]
                        else:
                            chain_checkpoint = float("inf")
                        if best_leaf and best_leaf[1] <= chain_checkpoint:
                            opt[m][i][idx] = best_leaf[1]
                            what[m][i][idx] = (False, best_leaf[0])
                        else:
                            opt[m][i][idx] = chain_checkpoint
                            what[m][i][idx] = (True,)
        return (opt, what)

    @staticmethod
    def _rec(chain: Chain, lmin: int, lmax: int, cmem, opt_table):
        # TODO: implement this
        raise NotImplementedError

    @staticmethod
    def _annotate_from_sequence(sequence: Sequence, node_list: List[List[Node]]):
        # TODO: implement this
        raise NotImplementedError
