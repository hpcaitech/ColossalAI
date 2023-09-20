from .estimate_memory import EstimateMemory
from .reorder_graph import ReorderGraph
from .trace_indice import TraceIndice
from .utils import NodeMgr, is_non_compute_node


class SelectChunk(object):
    def __init__(
        self,
        trace_indice: TraceIndice,
        estimate_memory: EstimateMemory,
        reorder_graph: ReorderGraph,
        node_mgr: NodeMgr,
        max_memory=None,
    ):
        self.trace_indice = trace_indice
        self.estimate_memory = estimate_memory
        self.reorder_graph = reorder_graph
        self.node_mgr = node_mgr
        if max_memory is not None:
            self.stratge = "fit_memory"
            self.max_memory = max_memory  # MB
        else:
            self.stratge = "min_memory"

    def _select_best_chunk_region(self, possible_chunk_regions, chunk_infos, mem_peak):
        if self.stratge == "min_memory":
            best_region = self._select_min_memory_chunk_region(possible_chunk_regions, chunk_infos)
        elif self.stratge == "fit_memory":
            best_region = self._select_fit_memory_chunk_region(possible_chunk_regions, chunk_infos, mem_peak)
        else:
            raise RuntimeError()
        return best_region

    def _select_fit_memory_chunk_region(self, possible_chunk_regions, chunk_infos, mem_peak):
        # stop chunk if max memory satisfy memory limit
        if max(mem_peak) < self.max_memory:
            return None

        # remove illegal regions
        illegal_regions = []
        for i in possible_chunk_regions:
            if not self._is_legal_region(i, chunk_infos):
                illegal_regions.append(i)
        for i in illegal_regions:
            if i in possible_chunk_regions:
                possible_chunk_regions.remove(i)

        if len(possible_chunk_regions) == 0:
            return None

        # get mem for chunk region
        regions_dict = []
        for region in possible_chunk_regions:
            cur_region = region.copy()
            cur_node_list, cur_region = self.reorder_graph.tmp_reorder(self.node_mgr.get_node_list(), cur_region)
            cur_chunk_infos = chunk_infos + [cur_region]
            cur_mem = self.estimate_memory.estimate_chunk_inference_mem(cur_node_list, cur_chunk_infos)[0]
            cur_chunk_region_peak = cur_mem[cur_region["region"][0] : cur_region["region"][1] + 1]
            cur_chunk_region_max_peak = max(cur_chunk_region_peak)
            if cur_chunk_region_max_peak < self.max_memory:
                regions_dict.append(
                    {
                        "chunk_info": region,
                        "chunk_max_mem": cur_chunk_region_max_peak,
                        "chunk_len": self._get_compute_node_num(region["region"][0], region["region"][1]),
                        "reorder_chunk_info": cur_region,
                        "reorder_node_list": cur_node_list,
                    }
                )
        # no region found
        if len(regions_dict) == 0:
            raise RuntimeError("Search failed. Try a larger memory threshold.")

        # select the min chunk len
        chunk_len = [i["chunk_len"] for i in regions_dict]
        best_region_idx = chunk_len.index(min(chunk_len))
        best_region = regions_dict[best_region_idx]

        # get max chunk size
        best_region = self._get_fit_chunk_size(best_region, chunk_infos)
        return best_region

    def _get_fit_chunk_size(self, chunk_region_dict, chunk_infos):
        chunk_size = 1
        reorder_chunk_info = chunk_region_dict["reorder_chunk_info"]
        reorder_chunk_info["chunk_size"] = chunk_size
        cur_chunk_max_mem = 0
        # search a region
        while cur_chunk_max_mem < self.max_memory:
            chunk_size *= 2
            reorder_chunk_info["chunk_size"] = chunk_size
            cur_chunk_infos = chunk_infos + [reorder_chunk_info]
            cur_mem_peak = self.estimate_memory.estimate_chunk_inference_mem(
                chunk_region_dict["reorder_node_list"], cur_chunk_infos
            )[0]
            cur_chunk_max_mem = max(cur_mem_peak[reorder_chunk_info["region"][0] : reorder_chunk_info["region"][1] + 1])
        # search exact size
        chunk_info = chunk_region_dict["chunk_info"]
        chunk_info["chunk_size"] = self._chunk_size_binary_search(
            chunk_size // 2, chunk_size, chunk_region_dict, chunk_infos
        )
        return chunk_info

    def _chunk_size_binary_search(self, left, right, chunk_region_dict, chunk_infos):
        if left >= 16:
            gap = 4
        else:
            gap = 1
        chunk_info = chunk_region_dict["reorder_chunk_info"]
        while right >= left + gap:
            mid = int((left + right) / 2 + 0.5)
            chunk_info["chunk_size"] = mid
            cur_chunk_infos = chunk_infos + [chunk_info]
            cur_mem_peak = self.estimate_memory.estimate_chunk_inference_mem(
                chunk_region_dict["reorder_node_list"], cur_chunk_infos
            )[0]
            cur_chunk_max_mem = max(cur_mem_peak[chunk_info["region"][0] : chunk_info["region"][1] + 1])
            if cur_chunk_max_mem >= self.max_memory:
                right = mid - gap
            else:
                left = mid + gap
        return left

    def _get_compute_node_num(self, start, end):
        count = 0
        for i in self.node_mgr.get_node_slice_by_idx(start, end + 1):
            if not is_non_compute_node(i):
                count += 1
        return count

    def _select_min_memory_chunk_region(self, possible_chunk_regions, chunk_infos):
        # remove illegal regions
        illegal_regions = []
        for i in possible_chunk_regions:
            if not self._is_legal_region(i, chunk_infos):
                illegal_regions.append(i)
        for i in illegal_regions:
            if i in possible_chunk_regions:
                possible_chunk_regions.remove(i)

        if len(possible_chunk_regions) == 0:
            return None

        # get max possible chunk region
        max_possible_chunk_region = (
            min([i["region"][0] for i in possible_chunk_regions]),
            max([i["region"][1] for i in possible_chunk_regions]),
        )

        # get mem for chunk region
        regions_dict_list = []
        for region in possible_chunk_regions:
            cur_region = region.copy()
            cur_node_list, cur_region = self.reorder_graph.tmp_reorder(self.node_mgr.get_node_list(), cur_region)
            cur_chunk_infos = chunk_infos + [cur_region]
            cur_mem_peak = self.estimate_memory.estimate_chunk_inference_mem(cur_node_list, cur_chunk_infos)[0]
            cur_chunk_region_peak = cur_mem_peak[max_possible_chunk_region[0] : max_possible_chunk_region[1] + 1]
            cur_chunk_region_max_peak = max(cur_chunk_region_peak)
            regions_dict_list.append(
                {
                    "chunk_info": region,
                    "chunk_max_mem": cur_chunk_region_max_peak,
                    "chunk_len": self._get_compute_node_num(region["region"][0], region["region"][1]),
                    "reorder_chunk_info": cur_region,
                    "reorder_node_list": cur_node_list,
                }
            )

        # select the min mem
        chunk_max_mem = [i["chunk_max_mem"] for i in regions_dict_list]
        best_region_idx = chunk_max_mem.index(min(chunk_max_mem))
        best_region = regions_dict_list[best_region_idx]["chunk_info"]
        if best_region is not None:
            best_region["chunk_size"] = 1
        return best_region

    def _is_legal_region(self, cur_chunk_info, chunk_infos):
        (chunk_region_start, chunk_region_end) = cur_chunk_info["region"]
        if cur_chunk_info in chunk_infos:
            return False
        if chunk_region_end < chunk_region_start:
            return False
        for i in chunk_infos:
            region = i["region"]
            if not (
                (chunk_region_start > region[1] and chunk_region_end > region[1])
                or (chunk_region_start < region[0] and chunk_region_end < region[0])
            ):
                return False
        return True
