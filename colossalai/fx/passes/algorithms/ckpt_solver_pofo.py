import copy
import math
from typing import List, Tuple

import torch
from colossalai.fx import is_compatible_with_meta
from colossalai.fx.codegen.activation_checkpoint_codegen import \
    _find_nested_ckpt_regions
from colossalai.fx.graph_module import ColoGraphModule
from colossalai.fx.passes.algorithms.ckpt_solver_rotor import (_compute_table, _construct_chain, _rec)
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.fx.profiler import parameter_size
from torch.fx import GraphModule, Node

from .linearize import linearize
from .operation import (Backward, Chain, ForwardCheck, ForwardEnable, ForwardNograd, Function, Loss, Offload, Prefetch,
                        Sequence)

INF = float("inf")


def _normalize_flops(chain: Chain, flops) -> Chain:
    """
    Normalize flops
    """
    for i in range(chain.length):
        chain.fweight[i] /= flops
        chain.bweight[i] /= flops

    return chain


class PofoTable:
    """PofoTable
    The PofoTable contains the necessary components to store intermediate results
    of dynamic programming and the operations alone the way.
    """

    def __init__(self, chain_length: int, mem_slots: int):
        """Init pofo table
        The pofo table contains two tables, opt and what, indicating values and
        operations.

        Args:
            chain_length (int): chain length
            mem_slots (int): number of memory slots
        """

        self.length = chain_length
        self.mem_slots = mem_slots

        # initializing tables
        # the first bool indicates whether the input has bar
        # opt table is for value, opt[True/False][i][A][(df, db)] = OCx(i, A, df, db)
        # what table is for decision, what[True/False][i][A][(df, db)] = (is_enable, is_offload, index)
        # where is_enable indicates whether we enable the gradient, is_offload indicates whether we
        # offload the input, index indicates the end of F_\empty sequence if is_enable = False
        self.opt = {
            False: [[{} for _ in range(mem_slots + 1)] for _ in range(self.length + 1)],
            True: [[{} for _ in range(mem_slots + 1)] for _ in range(self.length + 1)]
        }
        self.what = {
            False: [[{} for _ in range(mem_slots + 1)] for _ in range(self.length + 1)],
            True: [[{} for _ in range(mem_slots + 1)] for _ in range(self.length + 1)]
        }

    def _get_value(self, state, table, default):
        i, act_size, df, db, input_has_bar = state
        if act_size + df > self.mem_slots or act_size + db > self.mem_slots:
            return default

        try:
            return table[input_has_bar][i][act_size][(df, db)]
        except KeyError:
            print(f"state not found {state}")

    def get_opt(self, state):
        return self._get_value(state, self.opt, INF)

    def get_what(self, state):
        return self._get_value(state, self.what, INF)

    def set_value(self, state, opt, what):
        i, act_size, df, db, input_has_bar = state
        self.opt[input_has_bar][i][act_size][(df, db)] = opt
        self.what[input_has_bar][i][act_size][(df, db)] = what


class PofoSolver:
    """PofoSolver that executes algorithm mentioned in https://proceedings.neurips.cc/paper/2021/hash/c8461bf13fca8a2b9912ab2eb1668e4b-Abstract.html
    The new pofo solver is based on paper Efficient Combination of Rematerialization and Offloading for Training DNNs 
    and it's code given in the supplemental. Currently we doesn't use the whole set up in the original paper and reuse 
    rotor solver for the backward sequence as suggested in supplemental. The solver now is able to find strategy with offload. 
    """

    def __init__(self, chain: Chain, max_memory: int, bandwidth, mem_slots: int) -> None:
        self.chain = chain
        self.length = chain.length
        self.max_memory = max_memory
        self.mem_slots = mem_slots
        self.mem_unit = max_memory / mem_slots
        self.bandwidth = bandwidth

        self.disc_chain = copy.deepcopy(self.chain)
        self.disc_chain._discretize(self.mem_unit)

        self.rotor_table = _compute_table(self.disc_chain, mem_slots)
        self._compute_pofo_table()

    def _discretize(self, *values) -> Tuple:
        return tuple(math.ceil(value / self.mem_unit) for value in values)

    def _undiscretize(self, *discrete_values) -> Tuple:
        if len(discrete_values) == 1:
            return discrete_values[0] * self.mem_unit
        else:
            return tuple(d * self.mem_unit for d in discrete_values)

    def _mmax_all(self, idx: int):
        """
        Calculate the maximum memory usage of Fi_all
        """

        return self.chain.cbweight[idx + 1] + self.chain.fwd_mem_tmp[idx]

    def _mmax_b(self, idx: int):
        """
        Calculate the maximum memory usage of Bi
        """

        return self.chain.cbweight[idx +
                                   1] + self.chain.cweight[idx +
                                                           1] + self.chain.cweight[idx] + self.chain.bwd_mem_tmp[idx]

    def _mmax_ng(self, i: int, j: int):
        """
        Calculate the maximum memory usage of CF_i, F_i+1\empty, ... F_j\empty
        """

        res = self.chain.cweight[j + 1] + self.chain.fwd_mem_tmp[j]
        if j > i:
            res += self.chain.cweight[j]
        return res

    def _rotor_estimated_bwd(self, i, j, m, delta):
        compute = self.rotor_table[0][math.floor((m - self.chain.cweight[i]) / self.mem_unit)][i][j]
        comm = delta / self.bandwidth
        return (max(compute, comm) + compute + comm) / 2

    def _rotor_estimated_bwd_sequence(self, i, j, m, delta):
        return _rec(self.disc_chain, i, j, math.floor((m - self.chain.cweight[i]) / self.mem_unit), self.rotor_table)

    def _common_values_enable(self, state: Tuple):

        idx, act_size, df, db, input_has_bar = state
        input_size = self.chain.cbweight[idx] if input_has_bar else self.chain.cweight[idx]
        mf = act_size + df + input_size
        mb = act_size + db + input_size
        mem_avail = self.max_memory - act_size - input_size
        f_usage = self._mmax_all(idx)
        b_usage = self._mmax_b(idx)

        # infeasible
        if f_usage > mem_avail or b_usage > mem_avail:
            return None

        # calculate idle time
        eps_f_beta = max(0, f_usage - self.max_memory + mf)
        eps_b_beta = max(0, b_usage - self.max_memory + mb)
        idle_time = (eps_f_beta + eps_b_beta) / self.bandwidth

        # calculate offload and prefetch data
        offload_data = self.chain.fweight[idx] * self.bandwidth + eps_f_beta
        prefetch_data = self.chain.bweight[idx] * self.bandwidth + eps_b_beta

        # total_time
        total_time = self.chain.fweight[idx] + self.chain.bweight[idx] + idle_time

        return (offload_data, prefetch_data, total_time, idle_time)

    def _common_values_nograd(self, state: Tuple, j: int, iterative: bool = False):

        i, act_size, df, db, input_has_bar = state

        # compute new epsilon_tmp and sum_fwds
        if iterative:
            self.epsilon_tmp = max(self.epsilon_tmp, self._mmax_ng(i, j) - self.bandwidth * self.sum_fwds)
            self.sum_fwds += self.chain.fweight[j]
        else:
            self.epsilon_tmp = max(
                self._mmax_ng(i, k) - self.bandwidth * sum(self.chain.fweight[i:k]) for k in range(i, j + 1))
            self.sum_fwds = sum(self.chain.fweight[i:j + 1])

        input_size = self.chain.cbweight[i] if input_has_bar else self.chain.cweight[i]
        mf = act_size + df + input_size
        mem_avail = self.max_memory - act_size - input_size

        # if infeasible
        if max(self._mmax_ng(i, k) for k in range(i, self.length)) > mem_avail:
            return None

        eps_f_beta = max(0, self.epsilon_tmp - self.max_memory + mf)
        offload_data = self.sum_fwds * self.bandwidth + eps_f_beta

        # TODO: Implement the precise backward recompute sequence mentioned in the paper
        # currently we will use an approximate way to get the backward time
        time_backward = self._rotor_estimated_bwd(i, j, mem_avail, db)

        prefetch_data = time_backward * self.bandwidth
        idle_time = eps_f_beta / self.bandwidth
        total_time = self.sum_fwds + idle_time + time_backward

        return (offload_data, prefetch_data, total_time, idle_time)

    def _new_values(self, state: Tuple, do_offload: bool, common_values: Tuple) -> Tuple:
        """Generate new values for next state

        Args:
            state (Tuple): undiscretized states
            do_offload (bool): bool type indicates whether we need to do offload
            common_values (Tuple): common values (offload_data, prefetch_data, total_time, idle_time)

        Returns:
            Tuple: (new_act_size, new_df, new_db)
        """
        idx, act_size, df, db, input_has_bar = state
        offload_data, prefetch_data, *_ = common_values
        input_size = self.chain.cbweight[idx] if input_has_bar else self.chain.cweight[idx]
        if do_offload:
            new_act_size = act_size
            new_df = max(0, df + input_size - offload_data)
            new_db = max(0, db - prefetch_data) + input_size
        else:
            new_act_size = act_size + input_size
            new_df = max(0, df - offload_data)
            new_db = max(0, db - prefetch_data)

        return (new_act_size, new_df, new_db)

    def _compute_pofo_table(self):
        self.table = PofoTable(self.length, self.mem_slots)

        # initializing the loss
        for act_size in range(self.mem_slots + 1):
            for df in range(self.mem_slots - act_size + 1):
                for db in range(self.mem_slots - act_size + 1):
                    # undiscretize for idle time calculation
                    origin_values = self._undiscretize(act_size, df, db)

                    for input_has_bar in (False, True):
                        disc_state = (self.length, act_size, df, db, input_has_bar)
                        state = (self.length, *origin_values, input_has_bar)
                        common_values = self._common_values_enable(state)

                        # if no feasible choice
                        if common_values is None:
                            self.table.set_value(disc_state, INF, None)
                            continue

                        # if there is feasible choice
                        new_act_size, new_df, new_db = self._new_values(state, False, common_values)
                        eps_g = (new_df + new_db) / self.bandwidth
                        total_time = common_values[2] + eps_g
                        self.table.set_value(disc_state, total_time, (True, False))

        # main loop
        for i in reversed(range(self.length)):
            for act_size in range(self.mem_slots + 1):
                for df in range(self.mem_slots - act_size + 1):
                    for db in range(self.mem_slots - act_size + 1):
                        # undiscretize for idle time calculation
                        origin_values = self._undiscretize(act_size, df, db)

                        for input_has_bar in (False, True):
                            best_result = INF
                            best_choice = None
                            disc_state = (i, act_size, df, db, input_has_bar)
                            state = (i, *origin_values, input_has_bar)

                            # case 1: start with F_all
                            vals_enable = self._common_values_enable(state)
                            if vals_enable is not None:
                                for do_offload in (True, False):
                                    new_state = self._new_values(state, do_offload, vals_enable)
                                    new_state = (i + 1, *self._discretize(*new_state), True)
                                    total_time = vals_enable[2]
                                    results_all = self.table.get_opt(new_state) + total_time
                                    if results_all < best_result:
                                        best_result = results_all
                                        best_choice = (True, do_offload)

                            # case 2: start with F_ck
                            self.sum_fwds = 0
                            self.epsilon_tmp = 0
                            for j in range(i, self.length):
                                vals_nograd = self._common_values_nograd(state, j, True)

                                # if infeasible
                                if vals_nograd is None:
                                    continue

                                for do_offload in (True, False):
                                    new_state = self._new_values(state, do_offload, vals_nograd)
                                    new_state = (j + 1, *self._discretize(*new_state), False)
                                    total_time = vals_nograd[2]
                                    result_nograd = total_time + self.table.get_opt(new_state)
                                    if result_nograd < best_result:
                                        best_result = result_nograd
                                        best_choice = (False, do_offload, j)

                            self.table.set_value(disc_state, best_result, best_choice)

    def pofo_rec(self, disc_state):
        i, act_size, df, db, input_has_bar = disc_state
        result = Sequence(Function("pofo", *disc_state))
        what = self.table.get_what(disc_state)
        state = self._undiscretize(act_size, df, db)
        state = (i, *state, input_has_bar)
        i, act_size, df, db, input_has_bar = state

        if what is None:
            return None

        # if loss
        if i == self.length:
            result.insert(Loss())
            return result

        if what[0]:
            do_offload = what[1]
            values = self._common_values_enable(state)
            new_state = self._discretize(*self._new_values(state, do_offload, values))
            new_state = (i + 1, *new_state, True)
            if do_offload:
                result.insert(Offload(i, input_has_bar))
            result.insert(ForwardEnable(i))
            result.insert_sequence(self.pofo_rec(new_state))
            if do_offload:
                result.insert(Prefetch(i, input_has_bar))
            result.insert(Backward(i))

        else:
            _, do_offload, j = what
            values = self._common_values_nograd(state, j)
            new_state = self._discretize(*self._new_values(state, do_offload, values))
            new_state = (j + 1, *new_state, False)
            if do_offload:
                result.insert(Offload(i, input_has_bar))
            result.insert(ForwardCheck(i))
            for k in range(i + 1, j + 1):
                result.insert(ForwardNograd(k))
            result.insert_sequence(self.pofo_rec(new_state))
            if do_offload:
                result.insert(Prefetch(i, input_has_bar))
            m = self.max_memory - act_size - (self.chain.cbweight[i] if input_has_bar else self.chain.cweight[i])

            #TODO: Implement the precise backward recompute sequence mentioned in the paper
            result.insert_sequence(self._rotor_estimated_bwd_sequence(i, j, m, db))

        return result


def _annotate_from_pofo_sequence(sequence: Sequence, node_list: List[List[Node]]):
    op_list = sequence.list_operations()
    loss_op = next(op for op in op_list if isinstance(op, Loss))
    fwd_list = op_list[:op_list.index(loss_op)]
    bwd_list = op_list[op_list.index(loss_op) + 1:]
    ckpt_idx = 0
    in_ckpt = False
    ckpt_region = []

    # forward annotation
    for op in fwd_list:
        if in_ckpt:
            if isinstance(op, ForwardNograd):
                ckpt_region.append(op.index)

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
                ckpt_region = [op.index]

        else:
            if isinstance(op, ForwardCheck):
                in_ckpt = True
                ckpt_region.append(op.index)

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

    # annotate the offload
    offload_idx = 0
    for idx, op in enumerate(fwd_list):
        if isinstance(op, Offload):
            # corner case: offload input
            if op.index == 0:
                if isinstance(fwd_list[idx + 1], ForwardCheck):
                    for n in node_list[op.index]:
                        setattr(n, "activation_offload", True)
                else:
                    for n in node_list[op.index]:
                        setattr(n, "activation_offload", (offload_idx, True, False))
                    offload_idx += 1

            else:
                if op.has_bar:
                    # annotate previous node
                    if hasattr(node_list[op.index - 1][0], "activation_offload"):
                        for n in node_list[op.index - 1]:
                            n.activation_offload[-1] = True
                    else:
                        for n in node_list[op.index - 1]:
                            setattr(n, "activation_offload", [offload_idx, False, True])

                        offload_idx += 1

                # annotate this node
                if isinstance(fwd_list[idx + 1], ForwardCheck):
                    for n in node_list[op.index]:
                        setattr(n, "activation_offload", True)
                else:
                    for n in node_list[op.index]:
                        setattr(n, "activation_offload", [offload_idx, True, False])

                    offload_idx += 1


def solver_pofo(gm: ColoGraphModule,
                data,
                bandwidth,
                flops,
                mem_limit: int,
                mem_slots: int = 50,
                cnode: List[str] = None,
                eps: float = 0.0) -> ColoGraphModule:
    """Solver that combine offload and activation checkpoint
    Reference: https://proceedings.neurips.cc/paper/2021/hash/c8461bf13fca8a2b9912ab2eb1668e4b-Abstract.html

    Args:
        gm (ColoGraphModule): ColoGraphModule derived from tracer
        data: input of the model
        bandwidth: offload bandwidth, unit Byte/s
        flops: FLOPS of device, unit FLOPs/s
        mem_limit (int): memory limit, unit Byte
        mem_slots (int, optional): number of memory slots. Defaults to 500.
        cnode (List[str], optional): common node for linearize. Defaults to None.
        eps (float, optional): epsilon for memory decay. Defaults to 0.02.

    Returns:
        ColoGraphModule: annotated graph module
    """

    node_list = linearize(gm, cnode)
    mem_limit -= parameter_size(gm)

    # prepare data
    if is_compatible_with_meta():
        from colossalai.fx.profiler import MetaTensor
        data = MetaTensor(data, fake_device=next(gm.parameters()).device)
    MetaInfoProp(gm).run(data)
    chain: Chain = _construct_chain(node_list, data)
    chain = _normalize_flops(chain, flops)
    # currently we view loss as an op without expense
    chain.cbweight.append(0)
    chain.cweight.append(0)
    chain.fwd_mem_tmp.append(0)
    chain.bwd_mem_tmp.append(0)
    chain.fweight.append(0)
    chain.bweight.append(0)

    solver = PofoSolver(chain, mem_limit, bandwidth, mem_slots)
    first_state = (0, 0, 0, 0, False)
    sequence = solver.pofo_rec(first_state)
    if sequence == None:
        raise ValueError(f"Cannot solve sequence with {mem_limit} Bytes memory")

    _annotate_from_pofo_sequence(sequence, node_list)
    setattr(gm, "__sequence__", sequence)
    return gm
