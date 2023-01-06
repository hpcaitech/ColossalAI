import copy

from .select_chunk import SelectChunk
from .trace_index import TraceIndex, ReorderGraph
from .estiamte_memory import EstimateMemory
from .utils import (
    get_node_shape,
    is_non_compute_node,
    is_non_compute_node_except_placeholder,
)


class SearchChunk(object):
    def __init__(self, gm, max_memory=None, print_mem=False) -> None:
        self.gm = gm
        self.print_mem = print_mem
        self.index_tracer = TraceIndex(list(gm.graph.nodes))
        self.index_tracer.trace_index()
        self.reorder_graph = ReorderGraph(self.index_tracer)
        self.memory_estimator = EstimateMemory()
        self.chunk_selector = SelectChunk(
            self.index_tracer, self.memory_estimator, self.reorder_graph, max_memory=max_memory
        )

    def _find_peak_node(self, mem_peak):
        max_value = max(mem_peak)
        max_idx = mem_peak.index(max_value)
        return max_idx

    def _get_free_var(self):
        free_var_idx = []
        for idx, n in enumerate(self.index_tracer.node_list):
            if n.op == "placeholder":
                free_var_idx.append(idx)
        return free_var_idx

    def _get_min_free_var(self, active_node_list, free_vars):
        min_len = 999
        for idx, n in enumerate(active_node_list):
            if idx in free_vars:
                continue
            if len(n) < min_len:
                min_len = len(n)
        return min_len

    def _search_max_chunk_region(self, active_node, peak_node, chunk_regions):
        free_vars = self._get_free_var()
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

    def _is_not_compute(self, trace, chunk_range, dim_idx):
        if trace["idx"][dim_idx] not in trace["compute"]:
            return True
        if trace["idx"][dim_idx] in trace["compute"] and all(
            i < chunk_range[0] or i > chunk_range[1]
            for i in trace["compute"][trace["idx"][dim_idx]]
        ):
            return True
        return False

    def _find_free_dim(self, input_trace, output_trace, start_idx, end_idx):
        start_traces = input_trace[start_idx]
        end_trace = output_trace[end_idx]
        end_node = self.index_tracer.node_list[end_idx]
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
                    if not self.index_tracer.check_index_source(
                        start_dim, start_node, start_idx, end_dim, end_node
                    ):
                        continue
                    # check index copmute
                    if not self.index_tracer.check_index_compute(
                        start_idx, end_dim, end_node, end_idx
                    ):
                        continue
                    # flow search
                    chunk_info = self.index_tracer.flow_search(
                        start_idx, start_dim, end_idx, end_dim
                    )
                    if chunk_info is None:
                        continue
                    # check index copmute
                    if not self.index_tracer.check_index_duplicate(chunk_info):
                        continue
                    chunk_infos.append(chunk_info)
        return chunk_infos

    def _search_possible_chunk_regions(self, max_chunk_region, peak_node):
        possible_chunk_region = []
        output_trace = copy.deepcopy(self.index_tracer.idx_trace_list)
        input_trace = []  # trace of a node's input nodes
        for _, n in enumerate(self.index_tracer.node_list):
            cur_trace = {}
            for arg in n.args:
                if type(arg) == type(n) and not is_non_compute_node_except_placeholder(
                    arg
                ):
                    cur_trace[arg] = self.index_tracer._find_trace_from_node(arg)
            input_trace.append(cur_trace)

        for start_idx in range(max_chunk_region[0], peak_node + 1):
            for end_idx in range(peak_node, max_chunk_region[1] + 1):
                # skip non compute nodes
                if is_non_compute_node(
                    self.index_tracer.node_list[start_idx]
                ) or is_non_compute_node(self.index_tracer.node_list[end_idx]):
                    continue

                # select free dim
                chunk_info = self._find_free_dim(
                    input_trace, output_trace, start_idx, end_idx
                )
                if len(chunk_info) > 0:
                    possible_chunk_region.extend(chunk_info)
        return possible_chunk_region

    def _step_search(self, mem_peak, active_node, chunk_regions):
        peak_node = self._find_peak_node(mem_peak)
        max_chunk_region = self._search_max_chunk_region(
            active_node, peak_node, chunk_regions
        )
        if max_chunk_region == None:
            return None
        possible_chunk_regions = self._search_possible_chunk_regions(
            max_chunk_region, peak_node
        )
        best_chunk_region = self.chunk_selector._select_best_chunk_region(
            possible_chunk_regions, chunk_regions, peak_node, max_chunk_region, mem_peak
        )
        best_chunk_region = self.reorder_graph.reorder_all(best_chunk_region)
        return best_chunk_region

    def _stop_search(self, init_mem_peak, mem_peak):
        sorted_init_mem_peak = sorted(init_mem_peak)
        if max(mem_peak) < sorted_init_mem_peak[int(len(sorted_init_mem_peak) * 0.5)]:
            return True
        return False

    def search_region(self):
        chunk_infos = []
        (
            init_mem_peak,
            _,
            active_node,
        ) = self.memory_estimator.estimate_chunk_inference_mem(
            self.index_tracer.node_list
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
            ) = self.memory_estimator.estimate_chunk_inference_mem(
                self.index_tracer.node_list, chunk_infos
            )
            if self._stop_search(init_mem_peak, mem_peak):
                break
        if self.print_mem:
            self.print_mem = False
            self.memory_estimator.estimate_chunk_inference_mem(
                self.index_tracer.node_list, chunk_infos, print_mem=True
            )
        return chunk_infos
