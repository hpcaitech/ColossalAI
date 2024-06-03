import time
from abc import ABC, abstractmethod
from typing import Dict, List, Type

NOT_NVML = False
try:
    from pynvml import *
except:
    NOT_NVML = True

import torch
from torch.fx.node import Node

from colossalai.accelerator import get_accelerator

from .region import Region
from .training_simulator import AsynTrainingSimulator, SynTrainingSimulator, TrainingSimulator
from .util import NodeInfo, NvDevicePower


def benchmark_func(func, number=1, repeat=1, warmup=3):
    """
    benchmark data transfer cost.
    """

    for i in range(warmup):
        func()

    costs = []

    for i in range(repeat):
        torch.cuda.synchronize()
        begin = time.time()
        for i in range(number):
            func()
        torch.cuda.synchronize()
        costs.append((time.time() - begin) / number)

    return sum(costs) / len(costs)


class Solver(ABC):
    """
    The parameter offload solver.

    Args:
        region_list (List[Region]): represents the linearized DNN computing graph.
        memory_budget (float): the given memory budget.
        error_factor (float): the error factor.
            It is used to reduce the memory budget. Due to some errors in the estimation of peak memory and execution time.
    """

    def __init__(self, region_list: List[Region], memory_budget: float = -1.0, error_factor: float = 0.95) -> None:
        self.region_list = region_list

        self.error_factor: float = error_factor
        if memory_budget > 0:
            self.memory_budget = memory_budget * self.error_factor
        else:
            self.memory_budget = (
                torch.cuda.get_device_properties(get_accelerator().get_current_device()).total_memory
                * self.error_factor
            )

        self.link_to_bandwidth: Dict[str, Dict[float, float]] = self._profile_bandwidth()
        self.comp_power: float = self._extract_computing_power()

    @abstractmethod
    def _call_solver(self):
        raise NotImplementedError

    @abstractmethod
    def _try_to_offload(self, *args):
        raise NotImplementedError

    @abstractmethod
    def _eval_one_choice(self, *args):
        raise NotImplementedError

    def _compute_offload_profit(self, total_mem_saving: float, peak_mem_saving: float, extra_cost: float):
        """
        Compute the profits of the offload strategies,
        which packages the memory savings information for subsequent comparisons.

        Args:
            total_mem_saving (float): the total memory saving of the offload strategy.
            peak_mem_saving (float): the peak memory saving of the offload strategy.
            extra_cost (float): extra data transfer cost.

        Returns:
            tuple: profit information, the first term represents memory savings per unit of time.
        """

        if extra_cost == 0:
            # means data transfer overhead can be completely overlapped
            return (float("inf"), total_mem_saving, peak_mem_saving)
        return (total_mem_saving / extra_cost, total_mem_saving, peak_mem_saving)

    def _compare_profit(self, profit_a: tuple, profit_b: tuple) -> bool:
        """
        Compare the profits of the two offload strategies using the dictionary order algorithm.

        Args:
            profit_a (tuple): the profit of a offload strategy.
            profit_b (tuple): the profit of another offload strategy.

        Returns:
            bool: whether profit_a is greater than profit_b.
        """

        for val1, val2 in zip(profit_a, profit_b):
            if val1 != val2:
                return val1 > val2
        return False

    def _update_state(self, best_ts: TrainingSimulator):
        """
        Update the solver state.
        """

        self.best_ts = best_ts
        self._update_node_mem_info(best_ts.fwd_node_mem, best_ts.bwd_node_mem)

    def _update_node_mem_info(self, fwd_mem_info: Dict[Node, float], bwd_mem_info: Dict[Node, float]):
        """
        Update the runtime memory information of the node.

        Args:
            fwd_mem_info (Dict[Node, float]): the runtime memory of each node in forward pass.
            bwd_mem_info (Dict[Node, float]): the runtime memory of each node in backward pass.
        """

        for node, mem in fwd_mem_info.items():
            assert hasattr(node, "node_info") and isinstance(node.node_info, NodeInfo)
            node.node_info.runtime_fwd_mem = mem
        for node, mem in bwd_mem_info.items():
            assert hasattr(node, "node_info") and isinstance(node.node_info, NodeInfo)
            node.node_info.runtime_bwd_mem = mem

    def _extract_computing_power(self):
        """
        return the FP16 computing performance of the current NVIDIA GPU.

        Raises:
            TypeError: Unknown NVIDIA GPU device.
        """

        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        device_name = nvmlDeviceGetName(handle)
        units = 1e12

        if device_name.__contains__("RTX 3080"):
            return NvDevicePower.RTX3080_FP16 * units
        elif device_name.__contains__("RTX 3090"):
            return NvDevicePower.RTX3090_FP16 * units
        elif device_name.__contains__("V100"):
            return NvDevicePower.V100_FP16 * units
        elif device_name.__contains__("A100"):
            return NvDevicePower.A100_FP16 * units
        else:
            raise TypeError(f"Unknown NVIDIA GPU device name {device_name}")

    def _profile_bandwidth(self):
        """
        Profile the bidirectional communication bandwidth between CPU and GPU
        using data volumes ranging from 1KB to 1GB.
        """

        print("profiling bandwidth ......")
        link_to_bandwidth = {}
        links = ["h2d", "d2h"]

        for link in links:
            t_size = 1024
            size_to_bandwidth = {}

            # from 1KB to 1GB
            for i in range(21):
                if link == "h2d":
                    src_tensor = torch.ones(int(t_size), dtype=torch.int8, pin_memory=True)
                    dst_tensor = torch.ones((int(t_size)), dtype=torch.int8, device="cuda")
                elif link == "d2h":
                    src_tensor = torch.ones(int(t_size), dtype=torch.int8, device="cuda")
                    dst_tensor = torch.ones((int(t_size)), dtype=torch.int8, pin_memory=True)

                def func():
                    dst_tensor.copy_(src_tensor)

                size_to_bandwidth[t_size] = t_size / benchmark_func(func, number=5, repeat=3)
                print(
                    f"size: {t_size / 1024 ** 2:.3f} MB, "
                    f"{src_tensor.device.type}-to-{dst_tensor.device.type} "
                    f"bandwidth: {size_to_bandwidth[t_size] / 1024 ** 3:.3f} GB/s"
                )

                t_size *= 2

            link_to_bandwidth[link] = size_to_bandwidth
        return link_to_bandwidth


class SynGreedySolver(Solver):
    def __init__(self, region_list: List[Region], memory_budget: float = -1.0) -> None:
        super().__init__(region_list, memory_budget)

        self.best_ts: SynTrainingSimulator = None
        self._init_state()

    def _init_state(self):
        """
        Initialize the solver state when without offloading.
        """

        ts = SynTrainingSimulator(self.region_list, self.comp_power, self.link_to_bandwidth)
        ts.execute()
        self._update_state(ts)

    def _call_solver(self):
        """
        Call the solver to search an efficient parameter offloading strategy for the linearized graph.
        The solver adopts greedy algorithm.

        Raises:
            NotImplementedError: Unable to find a solution for the given memory budget.
        """

        print("search offloading strategy ......")
        while self.best_ts.peak_mem > self.memory_budget:
            offload_region = None
            best_ts = None
            max_profit = (0,)

            # search which region should be offloaded,
            # the last region does not need to be offloaded.
            for region in self.region_list[:-1]:
                if region.param_size and not region.need_offload:
                    temp_ts, profit = self._try_to_offload(region)
                    if self._compare_profit(profit, max_profit):
                        offload_region = region
                        max_profit = profit
                        best_ts = temp_ts

            if offload_region is not None and best_ts is not None:
                offload_region.need_offload = True
                offload_region.is_syn = True
                self._update_state(best_ts)
            else:
                raise NotImplementedError(
                    f"can't find the offload strategy met the memory budget {self.memory_budget / 1024 ** 2} MB, "
                    f"it needs {self.best_ts.peak_mem / 1024 ** 2:.3f} MB at least!"
                )

    def _call_solver_l2l(self):
        """
        The layer-wise offload strategy.
        """

        for region in self.region_list[:-1]:
            region.need_offload = True
            region.is_syn = True

    def _try_to_offload(self, offload_region: Region):
        # record previous information
        orig_need_offload = offload_region.need_offload
        assert not orig_need_offload
        offload_region.need_offload = True

        ts, profit = self._eval_one_choice(offload_region)

        # restore previous information
        offload_region.need_offload = orig_need_offload
        return ts, profit

    def _eval_one_choice(self, offload_region: Region):
        """
        Evaluate the profit of a strategy choice.

        Args:
            offload_region (Region): the offload region of current choice.

        Returns:
            SynTrainingSimulator: the training simulator corresponding to the current strategy.
            tuple: contains memory saving and cost information of the current strategy.
        """

        ts = SynTrainingSimulator(self.region_list, self.comp_power, self.link_to_bandwidth)
        ts.execute()

        extra_comm_cost = 2.0 * ts._get_communication_overhead("h2d", offload_region.param_size)
        # the shared region needs to be moved twice
        if offload_region.r_id < offload_region.shared_rid:
            extra_comm_cost *= 2.0
        profit = self._compute_offload_profit(ts.total_mem_saving, self.best_ts.peak_mem - ts.peak_mem, extra_comm_cost)

        return ts, profit


class AsynGreedySolver(Solver):
    def __init__(self, region_list: List[Region], memory_budget: float = -1.0, search_window_size: int = 3):
        super().__init__(region_list, memory_budget)

        self.search_window_size = search_window_size
        # Records the prefetch execution location of the offloaded region
        self.region_to_region_map = {}
        self.best_ts: AsynTrainingSimulator = None

        self._init_state()

    def _init_state(self):
        """
        Initialize the solver state when without offloading.
        """

        ts = AsynTrainingSimulator(self.region_list, self.comp_power, self.link_to_bandwidth)
        ts.execute()
        self._update_state(ts)
        print("init peak memory", self.best_ts.peak_mem / 1024**2, "MB")

    def _call_solver(self):
        """
        Call the solver to search an efficient parameter offloading strategy for the linearized graph.
        The solver adopts greedy algorithm.

        Raises:
            NotImplementedError: Unable to find a solution for the given memory budget.
        """

        print("search for offloading strategy ......")
        # Records the prefetch execution location of the offloaded region
        region_to_region_map = {}
        while self.best_ts.peak_mem > self.memory_budget:
            region_to_offload = None
            max_offload_profit = (0,)
            best_offl_ts = None

            # search which region should be offloaded,
            # the last region does not need to be offloaded
            for region in self.region_list[:-1]:
                if region.param_size and not region.need_offload:
                    max_prefetch_profit = (0,)
                    best_pref_ts = None

                    # search when to prefetch the region offloaded
                    for host_region in self.region_list[region.r_id + 1 : region.r_id + 1 + self.search_window_size]:
                        if host_region.bwd_prefetch_region is not None:
                            continue

                        temp_ts, profit = self._try_to_offload(host_region, region)

                        if self._compare_profit(profit, max_prefetch_profit):
                            region_to_region_map[region.r_id] = host_region
                            max_prefetch_profit = profit
                            best_pref_ts = temp_ts
                            if profit[0] == float("inf"):
                                break

                    if self._compare_profit(max_prefetch_profit, max_offload_profit):
                        region_to_offload = region
                        max_offload_profit = max_prefetch_profit
                        best_offl_ts = best_pref_ts

            if (region_to_offload is not None) and (best_offl_ts is not None):
                region_to_offload.need_offload = True
                if region_to_region_map[region_to_offload.r_id] == region_to_offload:
                    region_to_offload.is_syn = True
                else:
                    region_to_region_map[region_to_offload.r_id].bwd_prefetch_region = region_to_offload
                    self.region_to_region_map[region_to_offload.r_id] = region_to_region_map[region_to_offload.r_id]

                self._update_state(best_offl_ts)

            elif self.region_to_region_map.__len__() > 0:
                self._repair_strategy()
            else:
                raise NotImplementedError(
                    f"can't find the offload strategy met the memory budget {self.memory_budget / 1024 ** 2} MB, "
                    f"it needs {self.best_ts.peak_mem / 1024 ** 2:.3f} MB at least!"
                )

            region_to_region_map.clear()

    def _try_to_offload(self, host_region: Region, offload_region: Region):
        """
        Attempts to offload the region and prefetch it in backward pass.
        """

        # record previous information
        orig_prefetch = host_region.bwd_prefetch_region
        orig_is_syn = offload_region.is_syn
        orig_need_offload = offload_region.need_offload

        if host_region == offload_region:
            offload_region.is_syn = True
        else:
            host_region.bwd_prefetch_region = offload_region
        offload_region.need_offload = True

        ts, profit = self._eval_one_choice()

        # restore previous information
        host_region.bwd_prefetch_region = orig_prefetch
        offload_region.is_syn = orig_is_syn
        offload_region.need_offload = orig_need_offload

        return ts, profit

    def _try_convert_to_syn_upload(self, host_region: Region, offload_region: Region):
        """
        Attempts to convert asynchronous prefetch into synchronous upload operations.
        """

        # record previous information
        orig_prefetch = host_region.bwd_prefetch_region
        orig_is_syn = offload_region.is_syn
        assert orig_prefetch is not None and not orig_is_syn

        host_region.bwd_prefetch_region = None
        offload_region.is_syn = True

        ts, profit = self._eval_one_choice()

        # restore previous information
        host_region.bwd_prefetch_region = orig_prefetch
        offload_region.is_syn = orig_is_syn

        return ts, profit

    def _repair_strategy(self):
        """
        Repair offload strategy.
        It attempts to convert asynchronous prefetch into synchronous upload operations and selects the best one.
        The repair process does not end until peak memory is reduced or there is no asynchronous prefetch operation.
        """
        print("repair strategy ......")

        peak_mem_saving = 0
        while len(self.region_to_region_map) and peak_mem_saving <= 0:
            max_profit = (0,)
            best_ts = None
            undo_host_region = None
            undo_offload_region = None

            for offload_region_id, host_region in self.region_to_region_map.items():
                offload_region = self.region_list[offload_region_id]
                assert host_region.bwd_prefetch_region == offload_region
                assert offload_region.need_offload
                assert not offload_region.is_syn

                ts, profit = self._try_convert_to_syn_upload(host_region, offload_region)

                if self._compare_profit(profit, max_profit):
                    undo_host_region = host_region
                    undo_offload_region = offload_region
                    max_profit = profit
                    best_ts = ts

            if best_ts is None:
                raise NotImplementedError("repair error!")

            assert not undo_offload_region.is_syn
            undo_offload_region.is_syn = True
            undo_host_region.bwd_prefetch_region = None

            peak_mem_saving = self.best_ts.peak_mem - best_ts.peak_mem

            self._update_state(best_ts)
            self.region_to_region_map.pop(undo_offload_region.r_id)

        return best_ts

    def _eval_one_choice(self):
        """
        Evaluate the profit of a strategy choice.

        Returns:
            AsynTrainingSimulator: the training simulator corresponding to the current strategy.
            tuple: contains memory saving and cost information of the current strategy.
        """

        ts = AsynTrainingSimulator(self.region_list, self.comp_power, self.link_to_bandwidth)
        ts.execute()

        extra_comm_cost = max(ts.iter_end_time - self.best_ts.iter_end_time, 0)
        profit = self._compute_offload_profit(ts.total_mem_saving, self.best_ts.peak_mem - ts.peak_mem, extra_comm_cost)

        return ts, profit


class SolverFactory:
    solvers: Dict[str, Type[Solver]] = {"syn": SynGreedySolver, "asyn": AsynGreedySolver}

    @staticmethod
    def create(solver_name: str) -> Type[Solver]:
        if solver_name not in SolverFactory.solvers:
            raise TypeError(f"Unknown parameter offload policy {solver_name}")
        return SolverFactory.solvers[solver_name]

    @staticmethod
    def get_solver_names():
        return tuple(SolverFactory.solvers.keys())
