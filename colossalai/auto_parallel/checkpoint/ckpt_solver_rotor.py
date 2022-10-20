import math
import sys
from typing import List, Tuple

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
        # try to import C version solver if force_python is not set
        logger = get_dist_logger()
        if not force_python:
            try:
                from .rotor_C_solver import compute_table
                CVERSION = True

            # build module if module not found
            except ModuleNotFoundError:
                import os
                import subprocess
                logger.info("rotor_C_solver hasn't been built! Building library...", ranks=[0])
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
                    logger.info("rotor_C_solver has been built!", ranks=[0])
                    from .rotor_C_solver import compute_table
                    CVERSION = True
                else:
                    logger.info("rotor_C_solver built failed! Switching to Python solver!", ranks=[0])
                    CVERSION = False
        else:
            CVERSION = False

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
        """ chain : the class describing the AC graph
            lmin : index of the first forward to execute
            lmax : upper bound index of the last forward to execute (not included)
            cmem : number of available memory slots
            Return the optimal sequence of makespan Opt_hete[cmem][lmin][lmax-lmin]
        Args:
            chain (Chain): _description_
            lmin (_type_): _description_
            lmax (_type_): _description_
            cmem (_type_): _description_
            opt_table (_type_): _description_
        Raises:
            ValueError: _description_
        Returns:
            _type_: _description_
        """
        if cmem <= 0:
            raise ValueError("Can not process a chain with negative memory {cmem}".format(cmem=cmem))
        opt, what = opt_table
        sequence = Sequence(Function("Persistent", lmax - lmin, cmem))
        if opt[cmem][lmin][lmax] == float("inf"):
            # using logger to annonce that the solver is failed
            logger = get_dist_logger()
            logger.info("Can not process this chain from index {lmin} to {lmax} with memory {cmem}".format(lmin=lmin,
                                                                                                           lmax=lmax,
                                                                                                           cmem=cmem))

            # set global indicater SOLVER_FAILED to True
            global SOLVER_FAILED
            SOLVER_FAILED = True
            return sequence

        if lmin == lmax:
            if lmin == chain.length:
                sequence.insert(Loss())
            else:
                sequence.insert(ForwardEnable(lmin))
                sequence.insert(Backward(lmin))
            return sequence

        if what[cmem][lmin][lmax][0]:
            sequence.insert(ForwardEnable(lmin))
            sequence.insert_sequence(
                CheckpointSolverRotor._rec(chain, lmin + 1, lmax, cmem - chain.cbweight[lmin + 1], opt_table))
            sequence.insert(Backward(lmin))
        else:
            j = what[cmem][lmin][lmax][1]
            sequence.insert(ForwardCheck(lmin))
            for k in range(lmin + 1, j):
                sequence.insert(ForwardNograd(k))
            sequence.insert_sequence(CheckpointSolverRotor._rec(chain, j, lmax, cmem - chain.cweight[j], opt_table))
            sequence.insert_sequence(CheckpointSolverRotor._rec(chain, lmin, j - 1, cmem, opt_table))
        return sequence


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
                        setattr(n, "activation_checkpoint", [ckpt_idx])

                ckpt_idx += 1
                ckpt_region = []

            elif isinstance(op, ForwardCheck):
                for node_idx in ckpt_region:
                    for n in node_list[node_idx]:
                        setattr(n, "activation_checkpoint", [ckpt_idx])

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
                        n.activation_checkpoint.append(ckpt_idx)

                ckpt_idx += 1
                ckpt_region = []

            elif isinstance(op, ForwardCheck):
                for node_idx in ckpt_region:
                    for n in node_list[node_idx]:
                        n.activation_checkpoint.append(ckpt_idx)

                ckpt_idx += 1
                ckpt_region = [op.index]

            elif isinstance(op, Backward):
                for node_idx in ckpt_region:
                    for n in node_list[node_idx]:
                        n.activation_checkpoint.append(ckpt_idx)

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
        nested_length = max(len(op_list[idx].activation_checkpoint) for idx in range(start_idx, end_idx + 1))
        for idx in range(start_idx, end_idx + 1):
            op_list[idx].activation_checkpoint += [None] * (nested_length - len(op_list[idx].activation_checkpoint))


def solver_rotor(gm: ColoGraphModule,
                 data,
                 mem_limit: int,
                 mem_slots: int = 500,
                 cnode: List[str] = None,
                 eps: float = 0.0,
                 force_python: bool = False) -> ColoGraphModule:
    """solver that automatically find activation checkpoint in rotor's manner
    Args:
        gm (ColoGraphModule): ColoGraphModule generated by tracing model and MetaInfoProp.
        data (torch.Tensor): input data.
        mem_limit (int): memory budget in Byte.
        mem_slots (int, optional): number of slots for discretizing memory budget. Defaults to 500.
        cnode (List[Node], optional): common node list for linearize. Defaults to None.
        eps (float): epsilon for memory decay. Defaults to 0.0
        force_python (bool): force to use python version of dynamic programs
    Returns:
        ColoGraphModule: annotated ColoGraphModuled with __sequence__ attribute
    """

    # try to import C version solver if force_python is not set
    logger = get_dist_logger()
    if not force_python:
        try:
            from .dynamic_programs_C_version import persistent_compute_table
            CVERSION = True

        # build module if module not found
        except ModuleNotFoundError:
            import os
            import subprocess
            logger.info("dynamic_programs_C_version hasn't been built! Building library...", ranks=[0])
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
                logger.info("dynamic_programs_C_version has been built!", ranks=[0])
                from .dynamic_programs_C_version import persistent_compute_table
                CVERSION = True
            else:
                logger.info("dynamic_programs_C_version built failed! Using python version!", ranks=[0])
                CVERSION = False
    else:
        CVERSION = False

    # linearize the graph
    node_list = linearize(gm, cnode)

    # construct chain
    mem_unit = mem_limit * (1.0 - eps) // mem_slots
    chain: Chain = _construct_chain(node_list, data)
    chain._discretize(mem_unit)

    # use C version if possible
    if CVERSION and not force_python:
        logger.info("Using C version rotor solver!", ranks=[0])
        opt_table = persistent_compute_table(chain, mem_slots)
    else:
        opt_table = _compute_table(chain, mem_slots)
        logger.info("Using python version rotor solver!", ranks=[0])

    # found sequence
    sequence = _rec(chain, 0, chain.length, mem_slots - chain.cweight[0], opt_table)

    # if solver failed, we don't need to annotate the graph
    if not SOLVER_FAILED:
        _annotate_from_sequence(sequence, node_list)

    # set __sequence__ attribute to GraphModule
    if SOLVER_FAILED:
        setattr(gm, "__sequence__", None)
    else:
        setattr(gm, "__sequence__", sequence)

    # set __opttable__ attribute to GraphModule
    setattr(gm, "__opttable__", opt_table[0])
    gm.recompile()
    return
