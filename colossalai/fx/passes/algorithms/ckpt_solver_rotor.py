from typing import List, Set, Tuple, Dict
import torch
from torch.fx import GraphModule, Node
from colossalai.fx.graph_module import ColoGraphModule
import math
from .linearize import linearize
from .utils import *
from colossalai.fx.profiler import profile_function, profile_module
from colossalai.fx.passes.meta_info_prop import MetaInfoProp


# this is the python compute table code from rotor
# https://gitlab.inria.fr/hiepacs/rotor
# paper link: https://hal.inria.fr/hal-02352969
def _compute_table(chain: Chain, mmax) -> Tuple:
    """Returns the optimal table: a tuple containing: 
    Opt[m][lmin][lmax] with lmin = 0...chain.length
         and lmax = lmin...chain.length (lmax is not included) and m = 0...mmax
    what[m][lmin][lmax] is (True,) if the optimal choice is a chain checkpoint
                           (False, j) if the optimal choice is a leaf checkpoint of length j
    The computation uses dynamic programming"""

    fw = chain.fweight + [0]    ## forward time
    bw = chain.bweight    ## backward time, not used
    cw = chain.cweight + [0]    ## size of x (and of y)
    cbw = chain.cbweight + [0]    ## size of xbar
    fwd_tmp = chain.fwd_tmp + [0]
    bwd_tmp = chain.bwd_tmp + [0]

    # Build table
    opt = [[{} for _ in range(chain.length + 1)] for _ in range(mmax + 1)]
    what = [[{} for _ in range(chain.length + 1)] for _ in range(mmax + 1)]
    ## Last one is a dict because its indices go from i to l. Renumbering will wait for C implementation

    # Initialize borders of the tables for lmax-lmin = 0
    for m in range(mmax + 1):
        for i in range(chain.length + 1):
            #lmax-lmin = 0
            limit = max(cw[i + 1] + cbw[i + 1] + fwd_tmp[i], cw[i] + cw[i + 1] + cbw[i + 1] + bwd_tmp[i])
            if m >= limit:    ## Equation (1)
                opt[m][i][i] = fw[i] + bw[i]
            else:
                opt[m][i][i] = float("inf")

    # Compute everything
    for m in range(mmax + 1):
        for d in range(1, chain.length + 1):
            for i in range(chain.length + 1 - d):
                # for idx in range(i+1, chain.length + 1):
                idx = i + d
                mmin = cw[idx + 1] + cw[i + 1] + fwd_tmp[i]
                if idx > i + 1:
                    mmin = max(mmin, cw[idx + 1] + max(cw[j] + cw[j + 1] + fwd_tmp[j] for j in range(i + 1, idx)))
                if m < mmin:
                    opt[m][i][idx] = float("inf")
                else:
                    leaf_checkpoints = [(j, sum(fw[i:j]) + opt[m - cw[j]][j][idx] + opt[m][i][j - 1])
                                        for j in range(i + 1, idx + 1)
                                        if m >= cw[j]]
                    if leaf_checkpoints:
                        best_leaf = min(leaf_checkpoints, key=lambda t: t[1])
                    else:
                        best_leaf = None
                    if m >= cbw[i + 1]:
                        chain_checkpoint = opt[m][i][i] + opt[m - cbw[i + 1]][i + 1][idx]
                    else:
                        chain_checkpoint = float("inf")
                    if best_leaf and best_leaf[1] <= chain_checkpoint:
                        opt[m][i][idx] = best_leaf[1]
                        what[m][i][idx] = (False, best_leaf[0])
                    else:
                        opt[m][i][idx] = chain_checkpoint
                        what[m][i][idx] = (True,)
    return (opt, what)


def _rec(chain: Chain, lmin, lmax, cmem, opt_table):
    """ chain : the class describing the AC graph
        lmin : index of the first forward to execute
        lmax : upper bound index of the last forward to execute (not included)
        cmem : number of available memory slots
        Return the optimal sequence of makespan Opt_hete[cmem][lmin][lmax-lmin]"""
    if cmem <= 0:
        raise ValueError("Can not process a chain with negative memory {cmem}".format(cmem=cmem))
    opt, what = opt_table
    sequence = Sequence(Function("Persistent", lmax - lmin, cmem))
    if opt[cmem][lmin][lmax] == float("inf"):
        raise ValueError("Can not process this chain from index {lmin} to {lmax} with memory {cmem}".format(lmin=lmin,
                                                                                                            lmax=lmax,
                                                                                                            cmem=cmem))
    if lmin == lmax:
        if lmin == chain.length:
            sequence.insert(Loss())
        else:
            sequence.insert(ForwardEnable(lmin))
            sequence.insert(Backward(lmin))
        return sequence

    if what[cmem][lmin][lmax][0]:
        sequence.insert(ForwardEnable(lmin))
        sequence.insert_sequence(_rec(chain, lmin + 1, lmax, cmem - chain.cbweight[lmin + 1], opt_table))
        sequence.insert(Backward(lmin))
    else:
        j = what[cmem][lmin][lmax][1]
        sequence.insert(ForwardCheck(lmin))
        for k in range(lmin + 1, j):
            sequence.insert(ForwardNograd(k))
        sequence.insert_sequence(_rec(chain, j, lmax, cmem - chain.cweight[j], opt_table))
        sequence.insert_sequence(_rec(chain, lmin, j - 1, cmem, opt_table))
    return sequence


def _discretize(mem_unit, values):
    return [math.ceil(value / mem_unit) for value in values]


def _compute_size(obj: torch.Tensor) -> int:
    return obj.numel() * obj.element_size()


def _compute_output_size(node: List[Node]) -> int:
    """Compute the output size of a node

    Args:
        node (List[Node]): node, list of torch.fx.Node

    Returns:
        int: output size
    """

    return node[-1].meta['tensor_meta'].numel * torch.tensor([],
                                                             dtype=node[-1].meta['tensor_meta'].dtype).element_size()


def _get_inplace(node: Node) -> bool:
    """Get the inplace argument from torch.fx.Node

    Args:
        node (Node): torch.fx.Node

    Returns:
        bool: indicates whether this op is inplace
    """

    is_inplace = False
    if node.op == "call_function":
        is_inplace = node.kwargs.get("inplace", False)
    elif node.op == "call_module":
        is_inplace = getattr(node.graph.owning_module.get_submodule(node.target), "inplace", False)

    return is_inplace


def _construct_chain(node_list: List[List[Node]], data, mem_unit: int) -> Chain:

    fwd_time = []
    bwd_time = []

    if isinstance(data, torch.Tensor):
        xbar_sizes = [_compute_size(data)]
        x_sizes = [_compute_size(data)]
    elif isinstance(data, list) or isinstance(data, tuple):
        xbar_sizes = [_compute_size(obj) for obj in data]
        x_sizes = [_compute_size(obj) for obj in data]
    elif isinstance(data, dict):
        xbar_sizes = [_compute_size(obj) for obj in data.values()]
        x_sizes = [_compute_size(obj) for obj in data.values()]

    # currently we can't get the temp memory needed in fwd and bwd
    tmp_fwd = [0] * len(node_list)
    tmp_bwd = [0] * (len(node_list) + 1)

    for idx, node in enumerate(node_list):
        fwd_time.append(0)
        bwd_time.append(0)
        xbar_sizes.append(0)
        x_sizes.append(_compute_output_size(node))

        _check_inplace_flag = 1
        for n in node:
            fwd_time[-1] += max(n.__flops__, 1)

            # currently we haven't patched the backward flops count
            bwd_time[-1] += max(n.__flops__ * 2, 2)
            xbar_sizes[-1] += n.__activation__

            # we need to clear the xbar of previous node as there is
            # one op in the current node that use the previous node's
            # output but applies inplace operation on it
            # NOTE: This process should be done only once as the previous
            # node will only have one output
            if _check_inplace_flag:
                for par in n._input_nodes:
                    if par not in node and _get_inplace(n):
                        xbar_sizes[-2] -= x_sizes[-2]
                        _check_inplace_flag = 0

        xbar_sizes[-1] = max(xbar_sizes[-1], x_sizes[-1])

    bwd_time.append(0)

    xbar_sizes = _discretize(mem_unit, xbar_sizes)
    x_sizes = _discretize(mem_unit, x_sizes)
    tmp_fwd = _discretize(mem_unit, tmp_fwd)
    tmp_bwd = _discretize(mem_unit, tmp_bwd)

    return Chain(fwd_time, bwd_time, x_sizes, xbar_sizes, tmp_fwd, tmp_bwd)


def _annotate_from_sequence(sequence: Sequence, node_list: List[List[Node]]) -> GraphModule:
    op_list = sequence.list_operations()
    loss_op = next(op for op in op_list if isinstance(op, Loss))
    op_list = op_list[:op_list.index(loss_op)]
    ckpt_idx = 0
    in_ckpt = False
    ckpt_region = []
    for idx, op in enumerate(op_list, 0):
        if in_ckpt:
            if isinstance(op, ForwardNograd):
                ckpt_region.append(idx)

            elif isinstance(op, ForwardEnable):
                in_ckpt = False
                for node_idx in ckpt_region:
                    for n in node_list[node_idx]:
                        setattr(n, "activation_checkpoint", ckpt_idx)

                ckpt_idx += 1
                ckpt_region = []

            elif isinstance(op, ForwardCheck):
                for node_idx in ckpt_region:
                    for n in node_list[node_idx]:
                        setattr(n, "activation_checkpoint", ckpt_idx)

                ckpt_idx += 1
                ckpt_region = [idx]

        else:
            if isinstance(op, ForwardCheck):
                in_ckpt = True
                ckpt_region.append(idx)


def solver_rotor(gm: ColoGraphModule,
                 data,
                 mem_limit: int,
                 mem_slots: int = 500,
                 cnode: List[str] = None) -> ColoGraphModule:
    """solver that automatically find activation checkpoint in rotor's manner

    Args:
        gm (ColoGraphModule): ColoGraphModule generated by tracing model.
        data (torch.Tensor): input data.
        mem_limit (int): memory budget in Byte.
        mem_slots (int, optional): number of slots for discretizing memory budget. Defaults to 500.
        cnode (List[Node], optional): common node list for linearize. Defaults to None.

    Returns:
        ColoGraphModule: annotated ColoGraphModuled with __sequence__ attribute
    """

    node_list = linearize(gm, cnode)
    mem_unit = mem_limit // mem_slots
    MetaInfoProp(gm).run(data)
    chain: Chain = _construct_chain(node_list, data, mem_unit)
    opt_table = _compute_table(chain, mem_slots)
    sequence = _rec(chain, 0, chain.length, mem_slots - chain.cweight[0], opt_table)
    _annotate_from_sequence(sequence, node_list)

    # set __sequence__ attribute to GraphModule
    setattr(gm, "__sequence__", sequence)
    return gm
