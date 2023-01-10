import math
from copy import deepcopy
from typing import List, Set, Tuple

from torch.fx import Graph, Node

from colossalai.fx.profiler import calculate_fwd_in, calculate_fwd_tmp

from .ckpt_solver_base import CheckpointSolverBase

__all__ = ['CheckpointSolverChen']


class CheckpointSolverChen(CheckpointSolverBase):

    def __init__(self, graph: Graph, cnode: List[str] = None, num_grids: int = 6):
        """
        This is the simple implementation of Algorithm 3 in https://arxiv.org/abs/1604.06174.
        Note that this algorithm targets at memory optimization only, using techniques in appendix A.

        Usage:
            Assume that we have a ``GraphModule``, and we have already done the extractions
            to the graph to retrieve all information needed, then we could use the following
            code to find a solution using ``CheckpointSolverChen``:
            >>> solver = CheckpointSolverChen(gm.graph)
            >>> chen_graph = solver.solve()
            >>> gm.graph = chen_graph    # set the graph to a new graph

        Args:
            graph (Graph): The computing graph to be optimized.
            cnode (List[str], optional): Common node List, should be the subset of input. Defaults to None.
            num_grids (int, optional): Number of grids to search for b. Defaults to 6.
        """
        super().__init__(graph, 0, 0, True, cnode)
        self.num_grids = num_grids

    def solve(self) -> Graph:
        """Solve the checkpointing problem using Algorithm 3.

        Returns:
            graph (Graph): The optimized graph, should be a copy of the original graph.
        """
        checkpointable_op = ['call_module', 'call_method', 'call_function', 'get_attr']
        ckpt = self.grid_search()
        for i, seg in enumerate(ckpt):
            for idx in range(*seg):
                nodes = self.node_list[idx]
                for n in nodes:
                    if n.op in checkpointable_op:
                        n.meta['activation_checkpoint'] = i
        return deepcopy(self.graph)

    def run_chen_greedy(self, b: int = 0) -> Tuple[Set, int]:
        """
        This is the simple implementation of Algorithm 3 in https://arxiv.org/abs/1604.06174.
        """
        ckpt_intv = []
        temp = 0
        x = 0
        y = 0
        prev_idx = 2
        for idx, nodes in enumerate(self.node_list):
            for n in nodes:
                n: Node
                temp += calculate_fwd_in(n) + calculate_fwd_tmp(n)
                y = max(y, temp)
            if temp > b and idx > prev_idx:
                x += calculate_fwd_in(nodes[0])
                temp = 0
                ckpt_intv.append((prev_idx, idx + 1))
                prev_idx = idx + 1
        return ckpt_intv, math.floor(math.sqrt(x * y))

    def grid_search(self) -> Set:
        """
        Search ckpt strategy with b = 0, then run the allocation algorithm again with b = √xy.
        Grid search over [√2/2 b, √2 b] for ``ckpt_opt`` over ``num_grids`` as in appendix A.
        """
        _, b_approx = self.run_chen_greedy(0)
        b_min, b_max = math.floor(b_approx / math.sqrt(2)), math.ceil(b_approx * math.sqrt(2))
        b_opt = math.inf
        for b in range(b_min, b_max, (b_max - b_min) // self.num_grids):
            ckpt_intv, b_approx = self.run_chen_greedy(b)
            if b_approx < b_opt:
                b_opt = b_approx
                ckpt_opt = ckpt_intv
        return ckpt_opt
