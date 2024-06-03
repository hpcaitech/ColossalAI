from typing import Dict, List

import torch
from torch.fx.node import Node

from .utils import NodeMgr, get_node_shape, is_non_memory_node


class EstimateMemory(object):
    """
    Estimate memory with chunk
    """

    def __init__(self) -> None:
        pass

    def _get_node_size(self, x: Node) -> float:
        """
        return node size in MB
        """
        x = x.meta["tensor_meta"]
        if not hasattr(x, "numel"):
            out = sum([i.numel * torch.tensor([], dtype=i.dtype).element_size() for i in x])
        else:
            out = x.numel * torch.tensor([], dtype=x.dtype).element_size()
        out = float(out) / 1024**2
        return out

    def _add_active_node(self, n: Node, active_nodes: Dict, chunk_ratio: float) -> None:
        """
        add an active node and its shape to active node dict
        """
        if get_node_shape(n) is None:
            return
        if n.op == "placeholder":
            return
        if n not in active_nodes:
            node_size = self._get_node_size(n) * chunk_ratio
            active_nodes[n] = node_size

    def _build_delete_node_dict(self, node_mgr: NodeMgr) -> Dict:
        """
        build delete node dict, means node should be deleted at what time
        """
        delete_node_dict = {}
        for idx, node in enumerate(node_mgr.get_node_list()):
            # skip non shape node
            if get_node_shape(node) is None:
                continue
            # dont remove free nodes
            elif node.op == "placeholder":
                delete_node_dict[node] = len(node_mgr.get_node_list())
            # node no user
            elif len(node.users) == 0:
                delete_node_dict[node] = idx
            # log max use
            else:
                node_user_idx = [node_mgr.find_node_idx(i) for i in node.users.keys()]
                delete_node_dict[node] = max(node_user_idx)
        return delete_node_dict

    def _remove_deactive_node(
        self, user_idx: int, user: Node, active_nodes: List, delete_node_dict: List, kept_nodes: List = None
    ) -> None:
        """
        remove deactivate nodes from active nodes
        """
        if kept_nodes is None:
            kept_nodes = []
        if user.op in ("output",):
            return

        for node in list(active_nodes.keys()):
            # dont delete kept nodes
            if node in kept_nodes:
                continue
            # should be deleted
            if delete_node_dict[node] <= user_idx:
                active_nodes.pop(node)

    def _get_tmp_memory(self, node, not_contiguous_list, delete=False):
        mem = 0
        not_contiguous_ops = ["permute"]

        if node.op == "call_function" and any(n in node.name for n in ["matmul", "reshape"]):
            for n in node.args:
                if n in not_contiguous_list:
                    # matmul won't change origin tensor, but create a tmp copy
                    mem += self._get_node_size(n)
        elif node.op == "call_module":
            for n in node.args:
                if n in not_contiguous_list:
                    # module will just make origin tensor to contiguous
                    if delete:
                        not_contiguous_list.remove(n)
        elif node.op == "call_method" and any(i in node.name for i in not_contiguous_ops):
            if node not in not_contiguous_list:
                not_contiguous_list.append(node)
        return mem

    def _get_chunk_ratio(self, node, chunk_node_dim, chunk_size):
        if node not in chunk_node_dim:
            return 1.0
        node_shape = get_node_shape(node)
        chunk_dim = chunk_node_dim[node]["chunk_dim"]
        if chunk_dim is None:
            return 1.0
        else:
            return chunk_size / float(node_shape[chunk_dim])

    def _print_compute_op_mem_log(self, log, nodes, title=None):
        if title:
            print(title)
        for idx, (l, n) in enumerate(zip(log, nodes)):
            if n.op in ["placeholder", "get_attr", "output"]:
                continue
            if any(i in n.name for i in ["getitem", "getattr"]):
                continue
            print("%s:%.2f \t" % (n.name, l), end="")
            if (idx + 1) % 3 == 0:
                print("")
        print("\n")

    def _add_active_nodes_from_list(self, active_nodes: List, nodes: List) -> List:
        """
        add active nodes from nodes
        """
        for n in nodes:
            self._add_active_node(n, active_nodes, 1)

    def _get_memory_from_active_nodes(self, active_nodes: Dict) -> float:
        """
        sum all memory of active nodes
        """
        out = [i for i in active_nodes.values()]
        out = sum(out)
        return out

    def estimate_chunk_inference_mem(self, node_list: List, chunk_infos: Dict = None, print_mem: bool = False):
        """
        Estimate inference memory with chunk

        Args:
            node_list (List): _description_
            chunk_infos (Dict): Chunk information. Defaults to None.
            print_mem (bool): Wether to print peak memory of every node. Defaults to False.

        Returns:
            act_memory_peak_log (List): peak memory of every node
            act_memory_after_node_log (List): memory after executing every node
            active_node_list_log (List): active nodes of every node. active nodes refer to
                nodes generated but not deleted.
        """
        act_memory = 0.0
        act_memory_peak_log = []
        act_memory_after_node_log = []
        active_nodes = {}
        active_nodes_log = []
        not_contiguous_list = []
        node_mgr = NodeMgr(node_list)
        delete_node_dict = self._build_delete_node_dict(node_mgr)

        use_chunk = True if chunk_infos is not None else False
        chunk_within = False
        chunk_region_idx = None
        chunk_ratio = 1  # use it to estimate chunk mem
        chunk_inputs_all = []

        if use_chunk:
            chunk_regions = [i["region"] for i in chunk_infos]
            chunk_starts = [i[0] for i in chunk_regions]
            chunk_ends = [i[1] for i in chunk_regions]
            chunk_inputs = [i["inputs"] for i in chunk_infos]
            chunk_inputs_non_chunk = [i["inputs_non_chunk"] for i in chunk_infos]
            chunk_inputs_all = [j for i in chunk_inputs for j in i] + [j for i in chunk_inputs_non_chunk for j in i]
            chunk_outputs = [i["outputs"] for i in chunk_infos]
            chunk_node_dim = [i["node_chunk_dim"] for i in chunk_infos]
            chunk_sizes = [i["chunk_size"] if "chunk_size" in i else 1 for i in chunk_infos]

        for idx, node in enumerate(node_mgr.get_node_list()):
            # if node in chunk start nodes, change chunk ratio and add chunk_tensor
            if use_chunk and idx in chunk_starts:
                chunk_within = True
                chunk_region_idx = chunk_starts.index(idx)
                self._add_active_nodes_from_list(active_nodes, chunk_outputs[chunk_region_idx])

            # determine chunk ratio for current node
            if chunk_within:
                chunk_ratio = self._get_chunk_ratio(
                    node, chunk_node_dim[chunk_region_idx], chunk_sizes[chunk_region_idx]
                )

            # add current node as active node
            self._add_active_node(node, active_nodes, chunk_ratio)
            act_memory = self._get_memory_from_active_nodes(active_nodes)

            # if node is placeholder, just add the size of the node
            if node.op == "placeholder":
                act_memory_peak_log.append(act_memory)
            # skip output
            elif node.op == "output":
                continue
            # no change for non compute node
            elif is_non_memory_node(node):
                act_memory_peak_log.append(act_memory)
            # node is a compute op, calculate tmp
            else:
                # forward memory
                # TODO: contiguous_memory still not accurate for matmul, view, reshape and transpose
                tmp_memory = self._get_tmp_memory(node, not_contiguous_list, delete=True) * chunk_ratio
                # record max act memory
                act_memory_peak_log.append(act_memory + tmp_memory)

            # remove_deactive_node
            self._remove_deactive_node(idx, node, active_nodes, delete_node_dict, kept_nodes=chunk_inputs_all)

            # if node in chunk end nodes, restore chunk settings
            if use_chunk and idx in chunk_ends:
                self._remove_deactive_node(idx, node, active_nodes, delete_node_dict)  # dont provide kept nodes now
                chunk_within = False
                chunk_ratio = 1
                chunk_region_idx = None

            act_memory = self._get_memory_from_active_nodes(active_nodes)
            act_memory_after_node_log.append(act_memory)
            active_nodes_log.append(active_nodes.copy())

        if print_mem:
            print("with chunk" if use_chunk else "without chunk")
            self._print_compute_op_mem_log(act_memory_peak_log, node_mgr.get_node_list(), "peak")

        # param_memory = parameter_size(gm)
        # all_memory = act_memory + param_memory
        return act_memory_peak_log, act_memory_after_node_log, active_nodes_log
