import bisect
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List

from torch.fx.node import Node

from .region import Region
from .util import *


@dataclass
class ExecutionPeriod:
    start_time: float = 0
    end_time: float = 0


class TrainingSimulator(ABC):
    """
    The Training Simulator is used to simulate the training process.
    It records computation, communication, and runtime memory during forward and backward passes.

    Args:
        region_list (List[Region]): represents the linearized DNN computing graph.
        comp_power (float): the NVIDIA GPU FP16 computing power.
        link_to_bw (Dict[str, Dict[float, float]]): communication links and the corresponding bandwidth.
    """

    def __init__(self, region_list: List[Region], comp_power: float, link_to_bw: Dict[str, Dict[float, float]]) -> None:
        self.region_list = region_list
        self.region_num = len(region_list)

        self.runtime_mem: int = 0
        self.peak_mem: int = 0
        self.total_mem_saving: int = 0

        self.fwd_node_mem: Dict[Node, float] = {}
        self.bwd_node_mem: Dict[Node, float] = {}

        # Node dependencies in backward pass
        self.bwd_node_deps: Dict[Node, int] = {}

        self.comp_power: float = comp_power
        self.link_to_bandwidth: Dict[str, Dict[float, float]] = link_to_bw

    @abstractmethod
    def execute(self):
        raise NotImplementedError

    @abstractmethod
    def _eval_fwd_mem_per_region(self, region: Region):
        raise NotImplementedError

    @abstractmethod
    def _eval_bwd_mem_per_region(self, region: Region):
        raise NotImplementedError

    def _get_bandwidth(self, link: str, comm_volumn: float) -> float:
        """
        Get the data transfer bandwidth.

        Args:
            link (str): the data transfer link.
            comm_volumn (float): the amount of data transferred.

        Returns:
            float: the data transfer bandwidth.
        """

        assert len(self.link_to_bandwidth)
        if link not in self.link_to_bandwidth:
            raise TypeError(f"Unknown data transfer link {link}")

        # size_list = sorted(list(map(float, self.link_to_bandwidth[link].keys())))
        size_list = sorted(self.link_to_bandwidth[link].keys())
        d_idx = bisect.bisect_left(size_list, comm_volumn)
        return self.link_to_bandwidth[link][size_list[d_idx]]

    def _get_communication_overhead(self, link: str, comm_volumn: float) -> float:
        return comm_volumn / self._get_bandwidth(link, comm_volumn)

    def _get_computing_overhead(self, flop: float) -> float:
        return flop / self.comp_power


class SynTrainingSimulator(TrainingSimulator):
    def __init__(self, region_list: List[Region], comp_power: float, link_to_bw: Dict[str, Dict[float, float]]) -> None:
        super().__init__(region_list, comp_power, link_to_bw)

    def execute(self):
        """
        Simulate synchronous training process.
        """

        for reg in self.region_list:
            self._eval_fwd_mem_per_region(reg)

        for reg in self.region_list.__reversed__():
            self._eval_bwd_mem_per_region(reg)

    def _eval_fwd_mem_per_region(self, region: Region):
        """
        Evaluate the runtime and peak memory when the forward execution reaches the current region.
        """

        # upload parameters of the current region
        if requires_upload_p_in_fwd(self.region_list[region.shared_rid]):
            self.runtime_mem += region.param_size

        for node in region.nodes:
            self.runtime_mem += calculate_fwd_tmp(node) + calculate_fwd_out(node)
            self.fwd_node_mem[node] = self.runtime_mem
            self.peak_mem = max(self.runtime_mem, self.peak_mem)
            self.total_mem_saving += node.node_info.runtime_fwd_mem - self.runtime_mem

        if region.need_offload:
            self.runtime_mem -= region.param_size

    def _eval_bwd_mem_per_region(self, region: Region):
        """
        Evaluate the runtime and peak memory when the backward execution reaches the current region.
        """

        # upload parameters of the current region
        if region.need_offload:
            self.runtime_mem += region.param_size

        # add the gradient of the parameter
        if region.r_id < region.shared_rid:
            # gradient accumulation is required for shared parameters
            self.runtime_mem += 2.0 * region.param_size
        else:
            self.runtime_mem += region.param_size

        for node in region.nodes.__reversed__():
            self.runtime_mem -= calculate_fwd_out(node)
            self.runtime_mem += node.meta["bwd_mem_tmp"] + node.meta["bwd_mem_out"]
            self.peak_mem = max(self.runtime_mem, self.peak_mem)

            # The memory savings of a node may be negative due to parameter prefetch.
            self.total_mem_saving += node.node_info.runtime_bwd_mem - self.runtime_mem
            self.bwd_node_mem[node] = self.runtime_mem

            self.runtime_mem -= node.meta["bwd_mem_tmp"] + calculate_fwd_tmp(node)

            # free bwd_mem_out
            self.bwd_node_deps[node] = len(node.all_input_nodes)
            for user_node in node.users:
                if user_node in self.bwd_node_deps:
                    self.bwd_node_deps[user_node] -= 1
                    if self.bwd_node_deps[user_node] <= 0:
                        self.runtime_mem -= user_node.meta["bwd_mem_out"]

            if self.runtime_mem < 0:
                raise ValueError(
                    f"region id: {region.r_id}, node name: {node.name}, "
                    f"runtime_mem: {self.runtime_mem / 1024 ** 2:.3f}MB ---"
                    f"runtime memory computed less than 0, which is miscalculated!"
                )

        # release parameter and offload gradient in region
        if region.r_id == region.shared_rid:
            self.runtime_mem -= 2.0 * region.param_size
        elif region.r_id < region.shared_rid:
            self.runtime_mem -= 3.0 * region.param_size
        elif self.region_list[region.shared_rid].need_offload:
            self.runtime_mem -= region.param_size


class AsynTrainingSimulator(TrainingSimulator):
    def __init__(self, region_list: List[Region], comp_power: float, link_to_bw: Dict[str, Dict[float, float]]) -> None:
        super().__init__(region_list, comp_power, link_to_bw)

        self.iter_end_time: int = 0
        # the last computation execution period
        self.last_comp: ExecutionPeriod = ExecutionPeriod(start_time=0, end_time=0)
        # the last parameter prefetch execution period
        self.last_h2d: ExecutionPeriod = ExecutionPeriod(start_time=0, end_time=0)
        # the last gradient offload execution period
        self.last_d2h: ExecutionPeriod = ExecutionPeriod(start_time=0, end_time=0)
        # the forward computation execution period of the region
        self.fwd_reg_to_comp: OrderedDict[int, ExecutionPeriod] = OrderedDict()
        # the forward parameter prefetch execution period of the region
        self.fwd_reg_to_pref: OrderedDict[int, ExecutionPeriod] = OrderedDict()
        # the backward computation execution period of the region
        self.bwd_reg_to_comp: OrderedDict[int, ExecutionPeriod] = OrderedDict()
        # the backward parameter prefetch execution period of the region
        self.bwd_reg_to_pref: OrderedDict[int, ExecutionPeriod] = OrderedDict()
        # the gradient offload execution period of the region
        # which is divided into those that are waiting and those that have been released
        self.bwd_reg_to_offl_waiting: OrderedDict[int, ExecutionPeriod] = OrderedDict()
        self.bwd_reg_to_offl_freed: OrderedDict[int, ExecutionPeriod] = OrderedDict()
        # the region buffer, which records regions that are offloaded but not released
        self.reg_buffer_to_free: List[int] = []

        # node dependencies in backward pass
        self.bwd_node_deps: Dict[Node, int] = {}

        # the region execution flow,
        # where fwd_reg_flow[i,j] denotes whether the parameters of j-th region are in the GPU
        # when the execution reaches the i-th region.
        self.fwd_reg_flow = torch.zeros((self.region_num, self.region_num)).bool()
        self.bwd_reg_flow = torch.zeros((self.region_num, self.region_num)).bool()

    def execute(self):
        """
        Simulate asynchronous training process.
        In forward pass, parameter prefetching is advanced by one region.
        In backward pass, parameter prefetching is executed at the specified location,
            and gradient offloading is urgent.
        """

        for reg in self.region_list:
            if reg.param_size and reg.r_id < self.region_num - 1:
                for nr in self.region_list[reg.r_id + 1 :]:
                    if nr.param_size and requires_upload_p_in_fwd(self.region_list[nr.shared_rid]):
                        reg.fwd_prefetch_region = nr
                        break
            self._eval_fwd_cost_per_region(reg)
            self._eval_fwd_mem_per_region(reg)

        for reg in self.region_list.__reversed__():
            self._eval_bwd_cost_per_region(reg)
            self._eval_bwd_mem_per_region(reg)

        # release remaining grads
        for reg_id, offl_exec in self.bwd_reg_to_offl_waiting.items():
            self.bwd_reg_to_offl_freed[reg_id] = offl_exec
            self.runtime_mem -= self.region_list[reg_id].param_size
        self.bwd_reg_to_offl_waiting.clear()

        self.iter_end_time = max(self.last_comp.end_time, self.last_d2h.end_time)

    def _insert_h2d_exec(self, region: Region, is_fwd: bool = True):
        """
        Insert parameter prefetch execution period of the current region to the end of the h2d stream
        """

        pref_start_time = max(self.last_h2d.end_time, self.last_comp.end_time)
        pref_end_time = pref_start_time + 2.0 * self._get_communication_overhead("h2d", region.param_size)
        pref_ep = ExecutionPeriod(start_time=pref_start_time, end_time=pref_end_time)
        if is_fwd:
            self.fwd_reg_to_pref[region.r_id] = pref_ep
        else:
            self.bwd_reg_to_pref[region.r_id] = pref_ep
        self.last_h2d = pref_ep

    def _insert_comp_exec(self, region: Region, is_fwd: bool = True):
        """
        Insert computation execution period of the current region to the end of the computing stream
        """

        if is_fwd:
            reg_to_comp = self.fwd_reg_to_comp
            reg_to_pref = self.fwd_reg_to_pref
            flop_key = "fwd_flop"
        else:
            reg_to_comp = self.bwd_reg_to_comp
            reg_to_pref = self.bwd_reg_to_pref
            flop_key = "bwd_flop"
        comp_start_time = max(self.last_comp.end_time, reg_to_pref.get(region.r_id, ExecutionPeriod(0, 0)).end_time)
        comp_end_time = comp_start_time + sum(
            [self._get_computing_overhead(node.meta.get(flop_key, 0)) for node in region.nodes]
        )
        comp_ep = ExecutionPeriod(start_time=comp_start_time, end_time=comp_end_time)
        reg_to_comp[region.r_id] = comp_ep
        self.last_comp = comp_ep

    def _insert_d2h_exec(self, region: Region):
        """
        Insert gradient offload execution period of the current region to the end of the d2h stream
        """

        offl_start_time = max(self.last_d2h.end_time, self.last_comp.end_time)
        offl_end_time = offl_start_time + self._get_communication_overhead("d2h", region.param_size)
        offl_ep = ExecutionPeriod(start_time=offl_start_time, end_time=offl_end_time)
        self.bwd_reg_to_offl_waiting[region.r_id] = offl_ep
        self.last_d2h = offl_ep

    def _eval_fwd_cost_per_region(self, region: Region):
        """
        Evaluate computation and communication execution period of the region in forward pass.
        """

        # upload parameters of the first region
        if region.r_id == 0:
            self._insert_h2d_exec(region)

        # prefetch parameters of the next region
        fwd_prefetch_region = region.fwd_prefetch_region
        if fwd_prefetch_region and requires_upload_p_in_fwd(self.region_list[fwd_prefetch_region.shared_rid]):
            self._insert_h2d_exec(fwd_prefetch_region)

        # execute computation
        self._insert_comp_exec(region)

    def _eval_fwd_mem_per_region(self, region: Region):
        """
        Evaluate the runtime and peak memory when the forward execution reaches the current region.
        """

        # upload parameters of the current region
        if region.r_id <= 0:
            self.runtime_mem += region.param_size
            self.fwd_reg_flow[region.r_id, region.r_id] = True
        else:
            self.fwd_reg_flow[region.r_id] = self.fwd_reg_flow[region.r_id - 1]
            self.fwd_reg_flow[region.r_id, self.reg_buffer_to_free] = False
            self.reg_buffer_to_free.clear()

        # prefetch parameters of the next region
        fwd_prefetch_region = region.fwd_prefetch_region
        if fwd_prefetch_region and requires_upload_p_in_fwd(self.region_list[fwd_prefetch_region.shared_rid]):
            self.runtime_mem += fwd_prefetch_region.param_size
            self.fwd_reg_flow[region.r_id, fwd_prefetch_region.r_id] = True

        for node in region.nodes:
            self.runtime_mem += calculate_fwd_tmp(node) + calculate_fwd_out(node)
            self.peak_mem = max(self.runtime_mem, self.peak_mem)

            self.total_mem_saving += node.node_info.runtime_fwd_mem - self.runtime_mem
            self.fwd_node_mem[node] = self.runtime_mem

        if region.need_offload:
            self.runtime_mem -= region.param_size

            assert len(self.reg_buffer_to_free) <= 1, f"{len(self.reg_buffer_to_free)}"
            self.reg_buffer_to_free.append(region.r_id)

    def _eval_bwd_cost_per_region(self, region: Region):
        """
        Evaluate computation and communication execution period of the region in backward pass.
        """

        # upload parameters of the current region
        if region.is_syn:
            assert region.need_offload
            self._insert_h2d_exec(region, is_fwd=False)

        # prefetch parameters of the region choiced, which is parallel to computation
        if region.bwd_prefetch_region is not None:
            self._insert_h2d_exec(region.bwd_prefetch_region, is_fwd=False)

        # execute computation
        self._insert_comp_exec(region, is_fwd=False)

        # offload gradient
        if requires_offload_g_in_bwd(region):
            self._insert_d2h_exec(region)

        assert len(self.reg_buffer_to_free) == 0
        for reg_id, offl_exec in self.bwd_reg_to_offl_waiting.items():
            if offl_exec.end_time >= self.last_comp.start_time:
                break
            self.reg_buffer_to_free.append(reg_id)
            self.bwd_reg_to_offl_freed[reg_id] = offl_exec

        for reg_id in self.reg_buffer_to_free:
            self.bwd_reg_to_offl_waiting.pop(reg_id)

    def _eval_bwd_mem_per_region(self, region: Region):
        """
        Evaluate the runtime and peak memory when the backward execution reaches the current region.
        """

        if region.r_id + 1 < self.region_num:
            self.bwd_reg_flow[region.r_id] = self.bwd_reg_flow[region.r_id + 1]
        else:
            self.bwd_reg_flow[region.r_id] = self.fwd_reg_flow[-1]
        self.bwd_reg_flow[region.r_id, self.reg_buffer_to_free] = False

        # free gradients in the buffer
        while len(self.reg_buffer_to_free):
            reg_id = self.reg_buffer_to_free.pop(0)
            self.runtime_mem -= self.region_list[reg_id].param_size

        # upload parameters of the current region
        if region.is_syn:
            self.runtime_mem += region.param_size
            self.bwd_reg_flow[region.r_id, region.r_id] = True

        # prefetch parameters of the region choiced
        bwd_prefetch_region = region.bwd_prefetch_region
        if bwd_prefetch_region:
            self.runtime_mem += bwd_prefetch_region.param_size
            self.bwd_reg_flow[region.r_id, bwd_prefetch_region.r_id] = True

        # add the gradient of the parameter
        if region.r_id < region.shared_rid:
            # gradient accumulation is required for shared parameters
            self.runtime_mem += 2.0 * region.param_size
        else:
            self.runtime_mem += region.param_size

        for node in region.nodes.__reversed__():
            self.runtime_mem -= calculate_fwd_out(node)
            self.runtime_mem += node.meta["bwd_mem_tmp"] + node.meta["bwd_mem_out"]
            self.peak_mem = max(self.runtime_mem, self.peak_mem)

            # The memory savings of a node may be negative due to parameter prefetch.
            self.total_mem_saving += node.node_info.runtime_bwd_mem - self.runtime_mem

            self.bwd_node_mem[node] = self.runtime_mem

            self.runtime_mem -= node.meta["bwd_mem_tmp"] + calculate_fwd_tmp(node)

            # free bwd_mem_out
            self.bwd_node_deps[node] = len(node.all_input_nodes)
            for user_node in node.users:
                if user_node in self.bwd_node_deps:
                    self.bwd_node_deps[user_node] -= 1
                    if self.bwd_node_deps[user_node] <= 0:
                        self.runtime_mem -= user_node.meta["bwd_mem_out"]

            if self.runtime_mem < 0:
                raise ValueError(
                    f"region id: {region.r_id}, node name: {node.name}, "
                    f"runtime_mem: {self.runtime_mem / 1024 ** 2:.3f}MB ---"
                    f"runtime memory computed less than 0, which is miscalculated!"
                )

        # release parameters of the region
        if requires_release_p_in_bwd(self.region_list[region.shared_rid]):
            self.runtime_mem -= region.param_size
