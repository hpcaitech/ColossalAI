from typing import List
import torch
from torch.fx.graph import Graph
from torch.fx.node import Node
from colossalai.utils.cuda import get_current_device
from colossalai.fx.profiler import (calculate_fwd_out, calculate_fwd_tmp)
from colossalai.auto_parallel.param_offload.strategies_constructor import OffloadStrategiesConstructor
from colossalai.auto_parallel.param_offload.offload_strategy import SystemConfig

class Solver:

    def __init__(self,
                 graph: Graph,
                 strategies_constructor: OffloadStrategiesConstructor,
                 memory_budget: float = -1.0):
        self.graph = graph
        self.strategies_constructor = strategies_constructor
        self.leaf_strategies = self.strategies_constructor.leaf_strategies
        self.nodes = [strategies_vector.node for strategies_vector in self.leaf_strategies]
        self.memory_budget = memory_budget if memory_budget > 0 \
            else torch.cuda.get_device_properties(get_current_device()).total_memory

    def _compute_mem_saving(self):

        peak_mem = 0
        total_mem_saving = 0
        runtime_mem = 0

        # calculate runtime memory of each node during forward pass
        for node in self.graph.nodes:
            runtime_mem = runtime_mem + calculate_fwd_tmp(node) + calculate_fwd_out(node)
            # upload parameter
            runtime_mem += node.node_info.param_size
            total_mem_saving += max(node.node_info.runtime_fwd_mem - runtime_mem, 0)
            node.node_info.runtime_fwd_mem = runtime_mem

            peak_mem = max(runtime_mem, peak_mem)
            if node.node_info.offload_param_flag:
                runtime_mem -= node.node_info.param_size

        grad_in_computed = {}
        # calculate runtime memory of each node during backward pass
        for node in self.graph.nodes.__reversed__():
            runtime_mem -= calculate_fwd_out(node)
            runtime_mem = runtime_mem + node.meta['bwd_mem_tmp'] + node.meta['bwd_mem_out']
            if node.node_info.has_param:
                if node.node_info.offload_param_flag:
                    # upload parameter
                    runtime_mem += node.node_info.param_size
                # add gradient memory
                runtime_mem += node.node_info.param_size

                total_mem_saving += max(node.node_info.runtime_bwd_mem - runtime_mem, 0)
                node.node_info.runtime_bwd_mem = runtime_mem

                peak_mem = max(runtime_mem, peak_mem)

                # release parameter and offload gradient
                runtime_mem -= 2 * node.node_info.param_size

            peak_mem = max(runtime_mem, peak_mem)
            runtime_mem = runtime_mem - node.meta['bwd_mem_tmp'] - calculate_fwd_tmp(node)

            # release grad_in of current node
            for grad_in in node.meta["fwd_out"]:
                if isinstance(grad_in, torch.Tensor):
                    runtime_mem -= grad_in.numel() * grad_in.element_size()

            for in_node in list(node._input_nodes.keys()):
                # map multiple gradients of output to one tensor
                if grad_in_computed.get(in_node, False):
                    runtime_mem -= calculate_fwd_out(in_node)
                    grad_in_computed[in_node] = True

        return peak_mem, total_mem_saving

    def _call_solver_greedy_v1(self):
        """
        offload solution based on greedy algorithm.
        """
        peak_mem, total_mem_saving = self._compute_mem_saving()
        assert total_mem_saving == 0
        while peak_mem > self.memory_budget:
            offload_node = None
            max_profit = 0
            reduced_peak_mem = peak_mem
            for node in self.nodes:
                if (not node.node_info.offload_param_flag) and node.node_info.has_param:
                    node.node_info.offload_param_flag = True
                    tmp_peak_mem, tmp_total_mem_saving = self._compute_mem_saving()
                    profit = (peak_mem - tmp_peak_mem) / node.strategies_vector[0].comm_cost
                    if profit > max_profit:
                        offload_node = node
                        max_profit = profit
                        reduced_peak_mem = tmp_peak_mem
                    node.node_info.offload_param_flag = False
            offload_node.node_info.offload_param_flag = True
            peak_mem = reduced_peak_mem

    def _call_solver_l2l(self):
        """
        a layer to layer offload solution.
        """
        for node in self.nodes:
            node.node_info.offload_param_flag = True


class AsynGreedySolver:

    def __init__(self,
                 graph: Graph,
                 memory_budget: float = -1.0):
        self.graph = graph
        self.nodes = list(self.graph.nodes)
        self.memory_budget = memory_budget if memory_budget > 0 \
            else torch.cuda.get_device_properties(get_current_device()).total_memory
        # used to record computation start and end time stamp of each node
        self.node_compute_stream: List[List[float, float]] = []
        # used to record prefetch operation start and end time stamp of each node
        self.param_prefetch_stream: List[List[float, float]] = []

        self.peak_mem = -1

    def _init_compute_stream(self):
        compute_timestamp = 0
        for node in self.graph.nodes:
            if node.node_info.has_param:
                # upload parameter
                compute_timestamp += node.node_info.param_size / SystemConfig.BANDWIDTH
            self.node_compute_stream.append(
                [compute_timestamp, compute_timestamp + node.meta.get('fwd_flop', 0) / SystemConfig.COMPUTE_POWER])
            compute_timestamp += node.meta.get('fwd_flop', 0) / SystemConfig.COMPUTE_POWER

        for node in self.graph.nodes.__reversed__():
            self.node_compute_stream.append(
                [compute_timestamp, compute_timestamp + node.meta.get('bwd_flop', 0) / SystemConfig.COMPUTE_POWER])
            compute_timestamp += node.meta.get('bwd_flop', 0) / SystemConfig.COMPUTE_POWER
            if node.node_info.has_param:
                # offload gradient
                compute_timestamp += node.node_info.param_size / SystemConfig.BANDWIDTH


    def _call_solver_greedy(self):
        peak_mem_saving, total_mem_saving = self._compute_mem_saving()
        assert peak_mem_saving == 0 and total_mem_saving < 0
        while self.peak_mem > self.memory_budget:
            node_to_offload = None
            max_offload_profit = (0,)
            # record corresponding host node which prefetch the node to be offloaded
            node_to_node_map = {}
            # record the memory saving from the node to be offloaded
            node_to_mem_saving_map = {}

            # search which node should be offloaded
            for node in self.nodes:
                if node.node_info.has_param and (not node.node_info.offload_param_flag):
                    node_idx = self.nodes.index(node)
                    max_prefetch_profit = (0,)

                    # search when to prefetch the node offloaded
                    for following_node in self.nodes[node_idx+1:]:
                        if following_node.node_info.node_to_prefetch is not None:
                            continue
                        tmp_peak_mem_saving, tmp_total_mem_saving = self._compute_mem_saving(following_node, node)

                        if tmp_peak_mem_saving <= 0:
                            continue

                        extra_comm_cost = self._compute_extra_comm_cost(following_node, node)
                        tmp_profit = self._compute_offload_profit(tmp_peak_mem_saving, extra_comm_cost)

                        if self._compare_profit(tmp_profit, max_prefetch_profit):
                            node_to_node_map[node] = following_node
                            node_to_mem_saving_map[node] = tmp_peak_mem_saving
                            max_prefetch_profit = tmp_profit
                            if tmp_profit[0] == float('inf'):
                                break

                    if self._compare_profit(max_prefetch_profit, max_offload_profit):
                        node_to_offload = node
                        max_offload_profit = max_prefetch_profit

            node_to_node_map[node_to_offload].node_info.node_to_prefetch = node_to_offload
            node_to_offload.node_info.offload_param_flag = True
            self.peak_mem -= node_to_mem_saving_map[node_to_offload]

            self._update_rumtime_mem_for_node()
            self._update_exec_stream_and_node_info()

            node_to_node_map.clear()
            node_to_mem_saving_map.clear()


    def _update_rumtime_mem_for_node(self):
        self._compute_mem_saving(update_flag=True)

    def _update_exec_stream_and_node_info(self):

        self.node_compute_stream.clear()
        self.param_prefetch_stream.clear()

        compute_timestamp = 0
        prefetch_timestamp = 0

        # forward
        for node in self.graph.nodes:
            if node.node_info.has_param:
                # upload parameter
                compute_timestamp += node.node_info.param_size / SystemConfig.BANDWIDTH

            self.node_compute_stream.append(
                [compute_timestamp, compute_timestamp + node.meta.get('fwd_flop', 0) / SystemConfig.COMPUTE_POWER])
            compute_timestamp += node.meta.get('fwd_flop', 0) / SystemConfig.COMPUTE_POWER

        # backward
        for node in self.graph.nodes.__reversed__():

            if node.node_info.offload_param_flag:
                # wait parameter prefetch
                assert node.node_info.prefetch_end_timestamp != 0
                compute_timestamp = max(node.node_info.prefetch_end_timestamp, compute_timestamp)

            # prefetch parameter, which is parallel to node computation
            node_to_prefetch = node.node_info.node_to_prefetch
            if node_to_prefetch is not None:
                prefetch_timestamp = max(prefetch_timestamp, compute_timestamp)
                self.param_prefetch_stream.append(
                    [prefetch_timestamp,
                     prefetch_timestamp + node_to_prefetch.node_info.param_size / SystemConfig.BANDWIDTH])
                prefetch_timestamp += node_to_prefetch.node_info.param_size / SystemConfig.BANDWIDTH
                node_to_prefetch.node_info.prefetch_end_timestamp = prefetch_timestamp

            self.node_compute_stream.append(
                [compute_timestamp, compute_timestamp + node.meta.get('bwd_flop', 0) / SystemConfig.COMPUTE_POWER])
            compute_timestamp += node.meta.get('bwd_flop', 0) / SystemConfig.COMPUTE_POWER

            if node.node_info.has_param:
                # offload gradient
                compute_timestamp += node.node_info.param_size / SystemConfig.BANDWIDTH


    def _compute_offload_profit(self, mem_saving: float, extra_cost: float):
        if extra_cost == 0:
            # If the prefetch operation can be completely overlapped,
            # then will provide memory saving information to downstream
            return (float('inf'), mem_saving)
        return (mem_saving/extra_cost, )

    def _compare_profit(self, profit_a: tuple, profit_b: tuple):
        for val1, val2 in zip(profit_a, profit_b):
            if val1 != val2:
                return val1 > val2
        return False

    def _compute_mem_saving(self,
                            host_node_for_prefetch: Node=None,
                            node_to_offload: Node=None,
                            update_flag=False):
        cur_peak_mem = 0
        total_mem_saving = 0
        runtime_mem = 0

        # forward
        for node in self.graph.nodes:
            runtime_mem = runtime_mem + calculate_fwd_tmp(node) + calculate_fwd_out(node)
            # upload parameter
            runtime_mem += node.node_info.param_size
            total_mem_saving += max(node.node_info.runtime_fwd_mem - runtime_mem, 0)

            if update_flag:
                node.node_info.runtime_fwd_mem = runtime_mem

            cur_peak_mem = max(runtime_mem, cur_peak_mem)
            if node.node_info.offload_param_flag or (node == node_to_offload):
                runtime_mem -= node.node_info.param_size

        # backward
        grad_in_computed = {}
        for node in self.graph.nodes.__reversed__():
            runtime_mem -= calculate_fwd_out(node)

            # param prefetch
            node_to_prefetch = node.node_info.node_to_prefetch
            if node == host_node_for_prefetch:
                node_to_prefetch = node_to_offload
            if node_to_prefetch is not None:
                runtime_mem += node_to_prefetch.node_info.param_size

            runtime_mem = runtime_mem + node.meta['bwd_mem_tmp'] + node.meta['bwd_mem_out']
            if node.node_info.has_param:
                # There is no need to add up the parameter size because it may be prefetched or not offloaded.

                # add the gradient of the parameter
                runtime_mem += node.node_info.param_size

                # The memory savings of a node may be negative due to parameter prefetch.
                total_mem_saving += (node.node_info.runtime_bwd_mem - runtime_mem)

                if update_flag:
                    node.node_info.runtime_bwd_mem = runtime_mem

                cur_peak_mem = max(runtime_mem, cur_peak_mem)

                # release parameter and offload gradient
                runtime_mem -= 2 * node.node_info.param_size
            cur_peak_mem = max(runtime_mem, cur_peak_mem)
            runtime_mem = runtime_mem - node.meta['bwd_mem_tmp'] - calculate_fwd_tmp(node)

            # release grad_in of current node
            for grad_in in node.meta["fwd_out"]:
                if isinstance(grad_in, torch.Tensor):
                    runtime_mem -= grad_in.numel() * grad_in.element_size()

            for in_node in list(node._input_nodes.keys()):
                # # release fwd_in (fwd_out) of current node (input nodes)
                # if calculate_fwd_out(in_node) > 0 and (not fwd_out_released[in_node]):
                #     runtime_mem -= calculate_fwd_out(in_node)
                #     fwd_out_released[in_node] = True

                # map multiple gradients of output to one tensor
                if grad_in_computed.get(in_node, False):
                    runtime_mem -= calculate_fwd_out(in_node)
                    grad_in_computed[in_node] = True

        if (host_node_for_prefetch is None) and (node_to_offload is None):
            if update_flag:
                assert self.peak_mem == cur_peak_mem
            else:
                assert self.peak_mem < 0
                self.peak_mem = cur_peak_mem
        peak_mem_saving = self.peak_mem - cur_peak_mem
        return peak_mem_saving, total_mem_saving

    def _compute_extra_comm_cost(self, host_node_for_prefetch: Node, node_to_offload: Node):

        compute_start_timestamp = self.node_compute_stream[len(self.nodes)][0]
        prefetch_start_timestamp = compute_start_timestamp
        for node in self.graph.nodes.__reversed__():

            if node.node_info.offload_param_flag or (node == node_to_offload):
                # wait parameter prefetch
                assert node.node_info.prefetch_end_timestamp != 0
                compute_start_timestamp = max(node.node_info.prefetch_end_timestamp, compute_start_timestamp)

            # prefetch parameter, which is parallel to node computation
            node_to_prefetch = node.node_info.node_to_prefetch
            if node == host_node_for_prefetch:
                assert node.node_info.node_to_prefetch is None
                node_to_prefetch = node_to_offload
            if node_to_prefetch is not None:
                prefetch_start_timestamp = max(prefetch_start_timestamp, compute_start_timestamp)
                prefetch_start_timestamp += node_to_prefetch.node_info.param_size / SystemConfig.BANDWIDTH
                node_to_prefetch.node_info.prefetch_end_timestamp = prefetch_start_timestamp

            compute_start_timestamp += node.meta.get('bwd_flop', 0) / SystemConfig.COMPUTE_POWER

            if node.node_info.has_param:
                # offload gradient
                compute_start_timestamp += node.node_info.param_size / SystemConfig.BANDWIDTH

        # restore node info
        node_to_offload.node_info.prefetch_end_timestamp = 0

        return max(compute_start_timestamp-self.node_compute_stream[-1][1], 0)


    def plot_execution_stream(self):
        # plot
        x1 = self.node_compute_stream
        x2 = self.param_prefetch_stream
        pass