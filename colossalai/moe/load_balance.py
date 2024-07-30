from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed import ProcessGroup

from colossalai.cluster import ProcessGroupMesh
from colossalai.moe.manager import MOE_MANAGER
from colossalai.shardformer.layer.moe import MLPExperts
from colossalai.zero.low_level import LowLevelZeroOptimizer


class LoadBalancer:
    def __init__(
        self,
        experts: MLPExperts,
        gate: nn.Parameter,
        local_expert_num: int,
        expert_num: int,
        ep_group: ProcessGroup,
        dp_group: ProcessGroup,
        tolerance: Optional[float] = 0.1,
        beam_width: Optional[int] = 8,
        group_swap_factor: Optional[float] = 0.4,
    ) -> None:
        self.experts: MLPExperts = experts
        self.gate: nn.Parameter = gate
        self.moe_ep_group: ProcessGroup = ep_group
        self.moe_ep_ranks = MOE_MANAGER.parallel_info_dict[dist.get_world_size(self.moe_ep_group)].ep_group_ranks
        self.moe_dp_group: ProcessGroup = dp_group
        self.tolerance = tolerance
        self.beam_width = beam_width
        self.group_swap_factor = group_swap_factor
        self.local_expert_num = local_expert_num
        self.expert_num = expert_num
        self.local_load = None
        # TODO: use a global process group mesh
        pp_size = 1 if MOE_MANAGER.pp_size is None else MOE_MANAGER.pp_size
        global_dp_group = ProcessGroupMesh(pp_size, dist.get_world_size() // pp_size)
        self.global_dp_group = global_dp_group.get_group_along_axis(1)
        self.global_dp_rank = dist.get_rank(self.global_dp_group)
        self.global_dp_size = dist.get_world_size(self.global_dp_group)

    def _clear_load(self) -> None:
        self.local_load = None

    def _sync_load(self) -> Tensor:
        new_load = self.local_load.clone().detach()
        # all reduce load between ep group
        dist.all_reduce(new_load, group=self.moe_ep_group)
        # all reduce load between dp group
        dist.all_reduce(new_load, group=self.moe_dp_group)
        return new_load

    @staticmethod
    def _get_diff_from_avg(data: List, group: int, avg: float) -> float:
        return abs(sum(data[group]) / len(data[group]) - avg)

    @staticmethod
    def _swap_data(data: List, group_i: int, index_i: int, group_j: int, index_j: int) -> None:
        data[group_i][index_i], data[group_j][index_j] = (
            data[group_j][index_j],
            data[group_i][index_i],
        )

    @staticmethod
    def _normalize_data(data: List) -> List:
        max_value = max(max(sublist) for sublist in data)
        data = [[i / max_value for i in sublist] for sublist in data]
        return data

    @staticmethod
    def _get_swap_loss(
        group_swap_factor: float,
        swap_list: List,
        group_i: int,
        index_i: int,
        group_j: int,
        index_j: int,
    ) -> float:
        """
        Get swap loss. The swap loss is used to avoid the situation that
        the same index is swapped twice and the same group is swapped for multiple times.
        """
        swap_loss = 0
        for swap in swap_list:
            for group_id, index_id in zip([group_i, group_j], [index_i, index_j]):
                # the group has been swapped
                if group_id in [swap[0], swap[2]]:
                    # the index has been swapped
                    # we want to avoid the situation that the same index is swapped twice
                    if index_id in [swap[1], swap[3]]:
                        swap_loss += 1e5
                    # the index has not been swapped
                    # this is acceptable but as less as possible
                    else:
                        swap_loss += group_swap_factor
        return swap_loss

    @staticmethod
    def _check_convergence(data: List, avg: float, tolerance: float):
        """
        Check whether the data is converged after swap.
        """
        for sublist in data:
            if abs(sum(sublist) / len(sublist) - avg) > tolerance * avg:
                return False
        return True

    def _beam_search(
        self,
        inputs: Tuple[List, float, List],
        beam_width: int,
        avg: float,
        group_swap_factor: float,
    ) -> List:
        """
        Beam search for the best swap combination.
        Specifically, we swap two elements from two groups and calculate the score.
        The score is the difference between the origin group sum and the new group sum.
        The larger the score, the better the swap combination.

        Args:
            inputs (Tuple): (data, origin_score, swap_list)
            beam_width (int): beam width for beam search
            avg (float): average value of the data
            group_swap_factor (float): group loss for group swap loss

        Returns:
            List: results list
        """
        data, origin_score, swap_list = inputs
        results = []
        group_num = len(data)
        group_size = len(data[0])
        origin_diff_list = [self._get_diff_from_avg(data, i, avg) for i in range(group_num)]

        for group_num_i in range(group_num):
            for group_size_i in range(group_size):
                for group_num_j in range(group_num_i + 1, group_num):
                    for group_size_j in range(group_size):
                        new_data = deepcopy(data)
                        # calculate origin group sum
                        origin_diff = origin_diff_list[group_num_i] + origin_diff_list[group_num_j]
                        # swap data
                        self._swap_data(
                            new_data,
                            group_num_i,
                            group_size_i,
                            group_num_j,
                            group_size_j,
                        )
                        # calculate new group sum
                        new_diff = self._get_diff_from_avg(new_data, group_num_i, avg) + self._get_diff_from_avg(
                            new_data, group_num_j, avg
                        )
                        # caculate score
                        new_score = origin_diff - new_diff
                        if new_score > 0:
                            new_score = origin_score + new_score
                            # get swap loss
                            swap_loss = self._get_swap_loss(
                                group_swap_factor,
                                swap_list,
                                group_num_i,
                                group_size_i,
                                group_num_j,
                                group_size_j,
                            )
                            new_score = new_score - swap_loss
                            # update swap list
                            new_swap_list = swap_list + [(group_num_i, group_size_i, group_num_j, group_size_j)]
                            results.append((new_data, new_score, new_swap_list))
        # sort results
        results.sort(key=lambda x: x[1], reverse=True)
        # select top k results
        results = results[:beam_width]
        return results

    def _load_to_list(self, load: Tensor) -> List:
        load_len = len(load)
        assert load_len % self.local_expert_num == 0
        load_list = []
        tmp_list = []
        for i in range(len(load)):
            tmp_list.append(float(load[i]))
            if (i + 1) % self.local_expert_num == 0:
                load_list.append(tmp_list)
                tmp_list = []
        return load_list

    def _search_balance(
        self,
        data: List,
        tolerance: Optional[float] = 0.1,
        beam_width: Optional[int] = 8,
        group_swap_factor: Optional[float] = 0.4,
        return_swapped_data: Optional[bool] = False,
    ) -> Tuple[List, List]:
        """
        Search for the best swap combination to balance the data within the specified tolerance.
        And return the balanced data and the swap list. The swap list is used to record the swap.
        The swap list is a list of tuples. Each tuple is a swap operation.

        Args:
            data (List): expert load list.
                E.g. [[9.2, 8.3], [2.3, 10.0], [6.1, 7.2], [5.3, 3.2]]
                This means there are 4 devices and each devices has 2 experts.
                The value is the load of the expert.
            tolerance (float): tolerance for balance.
            beam_width (int): beam width for beam search.
            group_swap_factor (float): group swap factor for group swap loss.
                The bigger it is, the less times a group will be swapped.
            return_swapped_data (bool): whether to return the swapped data.

        Returns:
            Tuple: (balanced data, swap list).
                The swap list is a list of tuples. Each tuple is a swap operation.
                E.g. [(0, 0, 1, 0), (...), (...)]. The first tuple means
                the first expert of the first device is swapped with the first expert
                of the second device.
        """
        norm_data = self._normalize_data(data)
        avg = sum(sum(sublist) / len(sublist) for sublist in norm_data) / len(norm_data)
        results = [(norm_data, 0, [])]
        stop_flag = False

        while stop_flag == False:
            new_results = []
            best_score = results[0][1]
            for i in range(len(results)):
                new_results.extend(self._beam_search(results[i], beam_width, avg, group_swap_factor))
            if len(new_results) == 0:
                stop_flag = True
                break
            new_results.sort(key=lambda x: x[1], reverse=True)
            new_best_score = new_results[0][1]
            if new_best_score == best_score:
                stop_flag = True
                break
            new_results = new_results[:beam_width]
            results = new_results
            for i in results:
                if self._check_convergence(results[0][0], avg, tolerance):
                    stop_flag = True
                    break

        swap_list = results[0][2]
        if return_swapped_data:
            out = deepcopy(data)
            for swap in swap_list:
                self._swap_data(out, *swap)
            return out, swap_list
        else:
            return swap_list

    @staticmethod
    def _swap_expert_single_tensor(
        weight: nn.Parameter,
        expert_idx: int,
        comm_group: ProcessGroup,
        send_first: bool,
        comm_rank: int,
    ):
        # exchange weight
        local_weight = weight.data[expert_idx]
        new_weight = torch.empty_like(local_weight)
        if send_first:
            dist.send(local_weight, dst=comm_rank, group=comm_group)
            dist.recv(new_weight, src=comm_rank, group=comm_group)
        else:
            dist.recv(new_weight, src=comm_rank, group=comm_group)
            dist.send(local_weight, dst=comm_rank, group=comm_group)
        weight.data[expert_idx] = new_weight

    def _swap_expert_param_and_optim(
        self,
        weight: nn.Parameter,
        expert_idx: int,
        comm_group: ProcessGroup,
        send_first: bool,
        comm_rank: int,
        optim: LowLevelZeroOptimizer,
    ):
        # need to update master and working param if master param exists
        # else just update working param
        if weight in optim.optim.state:
            master_weight_ptr = None
            working_weight_ptr = weight
            exp_avg_ptr = optim.optim.state[working_weight_ptr]["exp_avg"]
            exp_avg_sq_ptr = optim.optim.state[working_weight_ptr]["exp_avg_sq"]
        else:
            master_weight_ptr = optim.working_to_master_param[id(weight)]
            working_weight_ptr = weight
            exp_avg_ptr = optim.optim.state[master_weight_ptr]["exp_avg"]
            exp_avg_sq_ptr = optim.optim.state[master_weight_ptr]["exp_avg_sq"]

        # exchange weight
        self._swap_expert_single_tensor(
            working_weight_ptr,
            expert_idx,
            comm_group,
            send_first,
            comm_rank,
        )
        if master_weight_ptr is not None:
            # TODO: exchange master weight, skip for now
            # master weight is shared by dp group
            tmp = working_weight_ptr.view(-1).split(
                working_weight_ptr.numel() // dist.get_world_size(self.moe_dp_group)
            )[dist.get_rank(self.moe_dp_group)]
            master_weight_ptr.data.copy_(tmp.clone().detach().to(master_weight_ptr.device).to(master_weight_ptr.dtype))
        # exchange optim
        self._swap_expert_single_tensor(exp_avg_ptr, expert_idx, comm_group, send_first, comm_rank)
        self._swap_expert_single_tensor(exp_avg_sq_ptr, expert_idx, comm_group, send_first, comm_rank)

    def _gather_global_dp_group(self, data: Tensor) -> Tensor:
        data_list = [torch.zeros_like(data) for _ in range(self.global_dp_size)]
        dist.all_gather(data_list, data, group=self.global_dp_group)
        data_list = torch.cat(data_list, dim=0)
        return data_list

    def _swap_moe_param(self, swap_list: List, optim: LowLevelZeroOptimizer) -> None:
        """
        Swap moe param and optim.
        We use different strategies to swap expert and gate.
        For expert, we exchange the param and optim of the expert by p2p.
        For gate, we all gather the gate choose the part we want.

        Args:
            swap_list (List)
            optim (LowLevelZeroOptimizer)
        """
        # get all experts weights
        local_rank = dist.get_rank(self.moe_ep_group)
        if self.experts.gated:
            weight_list = [self.experts.wi_up, self.experts.wi_gate]
        else:
            weight_list = [self.experts.wi]
        weight_list.append(self.experts.wo)

        # gate optim should be obtained first
        gate_shape = self.gate.shape
        # get master weight and optim
        master_gate_weight = optim.working_to_master_param[id(self.gate)]
        gate_exp_avg = optim.optim.state[master_gate_weight]["exp_avg"]
        gate_exp_avg_sq = optim.optim.state[master_gate_weight]["exp_avg_sq"]
        # gather
        global_master_gate_weight = self._gather_global_dp_group(master_gate_weight).view(gate_shape)
        global_gate_exp_avg = self._gather_global_dp_group(gate_exp_avg).view(gate_shape)
        global_gate_exp_avg_sq = self._gather_global_dp_group(gate_exp_avg_sq).view(gate_shape)
        assert (
            self.gate.shape
            == global_master_gate_weight.shape
            == global_gate_exp_avg.shape
            == global_gate_exp_avg_sq.shape
        )

        for swap in swap_list:
            source_group, source_idx, target_group, target_idx = swap
            source_rank = self.moe_ep_ranks[source_group]
            target_rank = self.moe_ep_ranks[target_group]
            # exchange expert
            if local_rank in [source_group, target_group]:
                for weight in weight_list:
                    if local_rank == source_group:
                        self._swap_expert_param_and_optim(
                            weight,
                            source_idx,
                            self.moe_ep_group,
                            True,
                            target_rank,
                            optim,
                        )
                    elif local_rank == target_group:
                        self._swap_expert_param_and_optim(
                            weight,
                            target_idx,
                            self.moe_ep_group,
                            False,
                            source_rank,
                            optim,
                        )
            # exchange gate
            source_expert_pos = source_group * self.local_expert_num + source_idx
            target_expert_pos = target_group * self.local_expert_num + target_idx
            for gate in [
                self.gate,
                global_master_gate_weight,
                global_gate_exp_avg,
                global_gate_exp_avg_sq,
            ]:
                origin_source = gate.data[source_expert_pos].clone().detach()
                origin_target = gate.data[target_expert_pos].clone().detach()
                gate.data[source_expert_pos], gate.data[target_expert_pos] = (
                    origin_target,
                    origin_source,
                )

        # update gate
        global_master_gate_weight = global_master_gate_weight.view(-1).split(
            global_master_gate_weight.numel() // self.global_dp_size
        )[self.global_dp_rank]
        master_gate_weight.data.copy_(global_master_gate_weight)
        global_gate_exp_avg = global_gate_exp_avg.view(-1).split(global_gate_exp_avg.numel() // self.global_dp_size)[
            self.global_dp_rank
        ]
        gate_exp_avg.data.copy_(global_gate_exp_avg)
        global_gate_exp_avg_sq = global_gate_exp_avg_sq.view(-1).split(
            global_gate_exp_avg_sq.numel() // self.global_dp_size
        )[self.global_dp_rank]
        gate_exp_avg_sq.data.copy_(global_gate_exp_avg_sq)

    @torch.no_grad()
    def update_load(self, load: Tensor) -> None:
        if len(load) != self.expert_num:
            padding_size = self.expert_num - len(load)
            padding = torch.zeros(padding_size, dtype=load.dtype, device=load.device)
            load = torch.cat((load, padding), dim=0)
        if self.local_load is None:
            self.local_load = load
        else:
            self.local_load += load

    @torch.no_grad()
    def balance_load(self, optim: LowLevelZeroOptimizer) -> None:
        # prepare load
        load = self._sync_load()
        load = self._load_to_list(load)
        # search balance
        swap_list = self._search_balance(load)
        if dist.get_rank() == 0:
            if len(swap_list) > 0:
                print(f"[Load Balance] Applying expert swap...")
            else:
                print(f"[Load Balance] Invalid swap, skip...")
        # swap expert and gate
        self._swap_moe_param(swap_list, optim)
        # clear load
        self._clear_load()
