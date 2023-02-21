from .trace_indice import TraceIndice
from .utils import NodeMgr


class ReorderGraph(object):
    """
    Reorder node list and indice trace list
    """

    def __init__(self, trace_indice: TraceIndice, node_mgr: NodeMgr) -> None:
        self.trace_indice = trace_indice
        self.node_mgr = node_mgr
        self.all_reorder_map = {i: i for i in range(len(self.node_mgr.get_node_list()))}

    def _get_reorder_map(self, chunk_info):
        reorder_map = {i: i for i in range(len(self.node_mgr.get_node_list()))}

        chunk_region_start = chunk_info["region"][0]
        chunk_region_end = chunk_info["region"][1]
        chunk_prepose_nodes = chunk_info["args"]["prepose_nodes"]
        chunk_prepose_nodes_idx = [self.node_mgr.find_node_idx(i) for i in chunk_prepose_nodes]
        # put prepose nodes ahead
        for idx, n in enumerate(chunk_prepose_nodes):
            n_idx = chunk_prepose_nodes_idx[idx]
            reorder_map[n_idx] = chunk_region_start + idx
        # put other nodes after prepose nodes
        for n in self.node_mgr.get_node_slice_by_idx(chunk_region_start, chunk_region_end + 1):
            if n in chunk_prepose_nodes:
                continue
            n_idx = self.node_mgr.find_node_idx(n)
            pos = sum([n_idx < i for i in chunk_prepose_nodes_idx])
            reorder_map[n_idx] = n_idx + pos

        return reorder_map

    def _reorder_chunk_info(self, chunk_info, reorder_map):
        # update chunk info
        chunk_info["region"] = (
            chunk_info["region"][0] + len(chunk_info["args"]["prepose_nodes"]),
            chunk_info["region"][1],
        )
        new_inputs_dim = []
        for _, input_dim in enumerate(chunk_info["inputs_dim"]):
            new_input_dim = {}
            for k, v in input_dim.items():
                new_input_dim[reorder_map[k]] = v
            new_inputs_dim.append(new_input_dim)
        chunk_info["inputs_dim"] = new_inputs_dim
        return chunk_info

    def _update_all_reorder_map(self, reorder_map):
        for origin_idx, map_idx in self.all_reorder_map.items():
            self.all_reorder_map[origin_idx] = reorder_map[map_idx]

    def _reorder_self_node_list(self, reorder_map):
        new_node_list = [None for _ in range(len(self.node_mgr.get_node_list()))]
        for old_idx, new_idx in reorder_map.items():
            new_node_list[new_idx] = self.node_mgr.get_node_by_idx(old_idx)
        self.node_mgr.update_node_list(new_node_list)

    def _reorder_idx_trace(self, reorder_map):
        # reorder list
        new_idx_trace_list = [None for _ in range(len(self.trace_indice.indice_trace_list))]
        for old_idx, new_idx in reorder_map.items():
            new_idx_trace_list[new_idx] = self.trace_indice.indice_trace_list[old_idx]
        self.trace_indice.indice_trace_list = new_idx_trace_list
        # update compute
        for idx_trace in self.trace_indice.indice_trace_list:
            compute = idx_trace["compute"]
            for dim_compute in compute:
                for idx, i in enumerate(dim_compute):
                    dim_compute[idx] = reorder_map[i]
        # update source
        for idx_trace in self.trace_indice.indice_trace_list:
            source = idx_trace["source"]
            for dim_idx, dim_source in enumerate(source):
                new_dim_source = {}
                for k, v in dim_source.items():
                    new_dim_source[reorder_map[k]] = v
                source[dim_idx] = new_dim_source

    def reorder_all(self, chunk_info):
        if chunk_info is None:
            return chunk_info
        if len(chunk_info["args"]["prepose_nodes"]) == 0:
            return chunk_info
        reorder_map = self._get_reorder_map(chunk_info)
        self._update_all_reorder_map(reorder_map)
        self._reorder_idx_trace(reorder_map)
        self._reorder_self_node_list(reorder_map)
        chunk_info = self._reorder_chunk_info(chunk_info, reorder_map)
        return chunk_info

    def reorder_node_list(self, node_list):
        new_node_list = [None for _ in range(len(node_list))]
        for old_idx, new_idx in self.all_reorder_map.items():
            new_node_list[new_idx] = node_list[old_idx]
        return new_node_list

    def tmp_reorder(self, node_list, chunk_info):
        if len(chunk_info["args"]["prepose_nodes"]) == 0:
            return node_list, chunk_info
        reorder_map = self._get_reorder_map(chunk_info)

        # new tmp node list
        new_node_list = [None for _ in range(len(node_list))]
        for old_idx, new_idx in reorder_map.items():
            new_node_list[new_idx] = node_list[old_idx]

        chunk_info = self._reorder_chunk_info(chunk_info, reorder_map)
        return new_node_list, chunk_info
