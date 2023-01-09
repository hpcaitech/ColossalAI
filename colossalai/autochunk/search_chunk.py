import copy
from typing import Dict, List, Tuple

from torch.fx.node import Node

from .estimate_memory import EstimateMemory
from .reorder_graph import ReorderGraph
from .select_chunk import SelectChunk
from .trace_flow import TraceFlow
from .trace_index import TraceIndex
from .utils import (
    get_node_shape,
    is_non_compute_node,
    is_non_compute_node_except_placeholder,
)


class SearchChunk(object):
    """
    This is the core class for AutoChunk.

    It defines the framework of the strategy of AutoChunk.
    Chunks will be selected one by one utill search stops.

    The chunk search is as follows:
    1. find the peak memory node
    2. find the max chunk region according to the peak memory node
    3. find all possible chunk regions in the max chunk region
    4. find the best chunk region for current status
    5. goto 1

    Attributes:
        gm: graph model
        print_mem (bool): print estimated memory
        trace_index: trace the flow of every dim of every node to find all free dims
        trace_flow: determine the region chunk strategy
        reorder_graph: reorder nodes to improve chunk efficiency
        estimate_memory: estimate memory with chunk
        select_chunk: select the best chunk region

    Args:
        gm: graph model
        max_memory (int): max memory in MB
        print_mem (bool): print estimated memory
    """

    def __init__(self, gm, max_memory=None, print_mem=False) -> None:
        self.gm = gm
        self.print_mem = print_mem
        self.trace_index = TraceIndex(list(gm.graph.nodes))
        self.trace_index.trace_index()
        self.trace_flow = TraceFlow(self.trace_index)
        self.reorder_graph = ReorderGraph(self.trace_index)
        self.estimate_memory = EstimateMemory()
        self.select_chunk = SelectChunk(
            self.trace_index,
            self.estimate_memory,
            self.reorder_graph,
            max_memory=max_memory,
        )

    def _find_peak_node(self, mem_peak):
        max_value = max(mem_peak)
        max_idx = mem_peak.index(max_value)
        return max_idx

    def _get_free_var_idx(self) -> List:
        """
        Get free var index

        Returns:
            free_var_idx (List): all indexs of free vars
        """
        free_var_idx = []
        for idx, n in enumerate(self.trace_index.node_list):
            if n.op == "placeholder":
                free_var_idx.append(idx)
        return free_var_idx

    def _search_max_chunk_region(
        self, active_node: List, peak_node: Node, chunk_regions: List
    ) -> Tuple:
        """
        Search max chunk region according to peak memory node

        Chunk region starts extending from the peak node, stops where free var num is min

        Args:
            active_node (List): active node status for every node
            peak_node (Node): peak memory node
            chunk_regions (List): chunk region info

        Returns:
            chunk_region_start (int)
            chunk_region_end (int)
        """
        free_vars = self._get_free_var_idx()
        free_var_num = len(free_vars)
        active_node_num = [len(i) for i in active_node]
        min_active_node_num = min(active_node_num[free_var_num:])
        threshold = max(free_var_num, min_active_node_num)

        # from peak_node to free_var
        inside_flag = False
        chunk_region_start = free_var_num
        for i in range(peak_node, -1, -1):
            if active_node_num[i] <= threshold:
                inside_flag = True
            if inside_flag and active_node_num[i] > threshold:
                chunk_region_start = i + 1
                break

        # from peak_node to len-2
        inside_flag = False
        chunk_region_end = len(active_node) - 1
        for i in range(peak_node, len(active_node)):
            if active_node_num[i] <= threshold:
                inside_flag = True
            if inside_flag and active_node_num[i] > threshold:
                chunk_region_end = i
                break

        for i in chunk_regions:
            region = i["region"]
            if chunk_region_start >= region[0] and chunk_region_end <= region[1]:
                return None
            elif (
                region[0] <= chunk_region_start <= region[1]
                and chunk_region_end > region[1]
            ):
                chunk_region_start = region[1] + 1
            elif (
                region[0] <= chunk_region_end <= region[1]
                and chunk_region_start < region[0]
            ):
                chunk_region_end = region[0] - 1
        return chunk_region_start, chunk_region_end

    def _find_chunk_info(self, input_trace, output_trace, start_idx, end_idx) -> List:
        """
        Find chunk info for a region.

        We are given the region start and region end, and need to find out all chunk info for it.
        We first loop every dim of start node and end node, to see if we can find dim pair,
        which is linked in a flow and not computed.
        If found, we then search flow in the whole region to find out all chunk infos.

        Args:
            input_trace (List): node's input trace in region
            output_trace (List): node's output trace in region
            start_idx (int): region start node index
            end_idx (int): region end node index

        Returns:
            chunk_infos: possible regions found
        """
        start_traces = input_trace[start_idx]
        end_trace = output_trace[end_idx]
        end_node = self.trace_index.node_list[end_idx]
        chunk_infos = []
        for end_dim, _ in enumerate(end_trace["idx"]):
            if len(start_traces) > 1:
                continue
            for start_node, start_trace in start_traces.items():
                for start_dim, _ in enumerate(start_trace["idx"]):
                    # dim size cannot be 1
                    if (
                        get_node_shape(end_node)[end_dim] == 1
                        or get_node_shape(start_node)[start_dim] == 1
                    ):
                        continue
                    # check index source align
                    if not self.trace_flow.check_index_source(
                        start_dim, start_node, start_idx, end_dim, end_node
                    ):
                        continue
                    # check index copmute
                    if not self.trace_flow.check_index_compute(
                        start_idx, end_dim, end_node, end_idx
                    ):
                        continue
                    # flow search
                    chunk_info = self.trace_flow.flow_search(
                        start_idx, start_dim, end_idx, end_dim
                    )
                    if chunk_info is None:
                        continue
                    # check index copmute
                    if not self.trace_flow.check_index_duplicate(chunk_info):
                        continue
                    chunk_infos.append(chunk_info)
        return chunk_infos

    def _search_possible_chunk_regions(
        self, max_chunk_region: Tuple, peak_node: Node
    ) -> List:
        """
        Search every possible region within the max chunk region.

        Args:
            max_chunk_region (Tuple)
            peak_node (Node): peak memory node

        Returns:
            possible_chunk_region (List)
        """
        possible_chunk_region = []
        output_trace = copy.deepcopy(self.trace_index.idx_trace_list)
        input_trace = []  # trace of a node's input nodes
        for _, n in enumerate(self.trace_index.node_list):
            cur_trace = {}
            for arg in n.args:
                if type(arg) == type(n) and not is_non_compute_node_except_placeholder(
                    arg
                ):
                    cur_trace[arg] = self.trace_index._find_trace_from_node(arg)
            input_trace.append(cur_trace)

        for start_idx in range(max_chunk_region[0], peak_node + 1):
            for end_idx in range(peak_node, max_chunk_region[1] + 1):
                # skip non compute nodes
                if is_non_compute_node(
                    self.trace_index.node_list[start_idx]
                ) or is_non_compute_node(self.trace_index.node_list[end_idx]):
                    continue

                # select free dim
                chunk_info = self._find_chunk_info(
                    input_trace, output_trace, start_idx, end_idx
                )
                if len(chunk_info) > 0:
                    possible_chunk_region.extend(chunk_info)
        return possible_chunk_region

    def _step_search(
        self,
        mem_peak: List[float],
        active_node: List[List[Node]],
        chunk_infos: List[Dict],
    ) -> Dict:
        """
        Find one chunk region

        The chunk search is as follows:
        1. find the peak memory node
        2. find the max chunk region according to the peak memory node
        3. find all possible chunk regions in the max chunk region
        4. find the best chunk region for current status

        Args:
            mem_peak (List): peak memory for every node
            active_node (List[List[Node]]): active node for every node
            chunk_infos (List[Dict]): all chunk info

        Returns:
            best_chunk_region (Dict)
        """
        peak_node = self._find_peak_node(mem_peak)
        max_chunk_region = self._search_max_chunk_region(
            active_node, peak_node, chunk_infos
        )
        if max_chunk_region == None:
            return None
        possible_chunk_regions = self._search_possible_chunk_regions(
            max_chunk_region, peak_node
        )
        best_chunk_region = self.select_chunk._select_best_chunk_region(
            possible_chunk_regions, chunk_infos, peak_node, max_chunk_region, mem_peak
        )
        best_chunk_region = self.reorder_graph.reorder_all(best_chunk_region)
        return best_chunk_region

    def _stop_search(self, init_mem_peak, mem_peak):
        sorted_init_mem_peak = sorted(init_mem_peak)
        if max(mem_peak) < sorted_init_mem_peak[int(len(sorted_init_mem_peak) * 0.5)]:
            return True
        return False

    def search_region(self) -> Dict:
        """
        Search all chunk regions:
        1. Estimate current memory
        2. Find best chunk for current memory
        3. goto 1

        Returns:
            chunk_infos (Dict)
        """
        chunk_infos = []
        (
            init_mem_peak,
            _,
            active_node,
        ) = self.estimate_memory.estimate_chunk_inference_mem(
            self.trace_index.node_list
        )
        mem_peak = init_mem_peak

        while True:
            chunk_info = self._step_search(mem_peak, active_node, chunk_infos)
            if chunk_info is None:
                break
            chunk_infos.append(chunk_info)

            (
                mem_peak,
                _,
                active_node,
            ) = self.estimate_memory.estimate_chunk_inference_mem(
                self.trace_index.node_list, chunk_infos
            )
            if self._stop_search(init_mem_peak, mem_peak):
                break
        if self.print_mem:
            self.print_mem = False
            self.estimate_memory.estimate_chunk_inference_mem(
                self.trace_index.node_list, chunk_infos, print_mem=True
            )
        return chunk_infos
