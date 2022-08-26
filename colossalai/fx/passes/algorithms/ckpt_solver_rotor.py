from typing import List, Set, Tuple, Dict
import torch
from torch.fx import GraphModule, Node
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


def _construct_chain(node_dict: Dict[int, Node], data: torch.Tensor, mem_unit: int) -> Chain:

    fwd_time = []
    bwd_time = []
    xbar_sizes = [data.numel() * data.element_size()]
    x_sizes = [data.numel() * data.element_size()]

    # currently we can't get the temp memory needed in fwd and bwd
    tmp_fwd = [0] * len(node_dict)
    tmp_bwd = [0] * (len(node_dict) + 1)

    for key in node_dict.keys():
        fwd_time.append(0)
        bwd_time.append(0)
        xbar_sizes.append(0)
        x_sizes.append(node_dict[key][-1].meta['tensor_meta'].numel *
                       torch.tensor([], dtype=node_dict[key][-1].meta['tensor_meta'].dtype).element_size())
        for node in node_dict[key]:
            fwd_time[-1] += node.__flops__

            # currently we haven't patched the backward flops count
            bwd_time[-1] += node.__flops__ * 2

            xbar_sizes[-1] += node.__activation__

        xbar_sizes[-1] = max(xbar_sizes[-1], x_sizes[-1])

    bwd_time.append(0)

    fwd_time = _discretize(mem_unit, fwd_time)
    bwd_time = _discretize(mem_unit, bwd_time)
    xbar_sizes = _discretize(mem_unit, xbar_sizes)
    x_sizes = _discretize(mem_unit, x_sizes)
    tmp_fwd = _discretize(mem_unit, tmp_fwd)
    tmp_bwd = _discretize(mem_unit, tmp_bwd)

    return Chain(fwd_time, bwd_time, x_sizes, xbar_sizes, tmp_fwd, tmp_bwd)


def _annotate_from_sequence(sequence: Sequence, node_dict: Dict[int, Node]) -> GraphModule:
    op_list = sequence.list_operations()
    loss_op = [op for op in op_list if isinstance(op, Loss)][0]
    op_list = op_list[:op_list.index(loss_op)]
    ckpt_idx = 0
    in_ckpt = False
    ckpt_region = []
    for idx, op in enumerate(op_list, 1):
        if in_ckpt:
            if isinstance(op, ForwardNograd):
                ckpt_region.append(idx)

            elif isinstance(op, ForwardEnable):
                in_ckpt = False
                for idx in ckpt_region:
                    for node in node_dict[idx]:
                        setattr(node, "activation_checkpoint", ckpt_idx)

                ckpt_idx += 1
                ckpt_region = []

            elif isinstance(op, ForwardCheck):
                for idx in ckpt_region:
                    for node in node_dict[idx]:
                        setattr(node, "activation_checkpoint", ckpt_idx)

                ckpt_idx += 1
                ckpt_region = [idx]

        else:
            if isinstance(op, ForwardCheck):
                in_ckpt = True
                ckpt_region.append(idx)


def solver_rotor(gm: GraphModule, data: torch.Tensor, mem_limit: int, mem_slots: int = 500) -> GraphModule:
    node_dict = linearize(gm)
    mem_unit = mem_limit // mem_slots
    MetaInfoProp(gm).run(data)
    chain: Chain = _construct_chain(node_dict, data, mem_unit)
    opt_table = _compute_table(chain, mem_slots)
    sequence = _rec(chain, 0, chain.length, mem_slots - chain.cweight[0], opt_table)
    _annotate_from_sequence(sequence, node_dict)
    return gm
