import torch
from torch.fx.graph import Graph
from colossalai.utils.cuda import get_current_device
from colossalai.fx.profiler import (calculate_fwd_out, calculate_fwd_tmp)
from colossalai.auto_parallel.param_offload.strategies_constructor import OffloadStrategiesConstructor

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