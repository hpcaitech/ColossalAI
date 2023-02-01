import copy
from typing import Any, Callable, Dict, Iterable, List, Tuple

import torch
from torch.fx.node import Node, map_arg

from colossalai.fx.profiler import activation_size, parameter_size

from .utils import NodeMgr, delete_free_var_from_last_use, get_node_shape, is_non_memory_node


class EstimateMemory(object):
    """
    Estimate memory with chunk
    """

    def __init__(self, node_mgr: NodeMgr) -> None:
        self.node_mgr = node_mgr

    def _get_meta_node_size(self, x):
        x = x.meta["tensor_meta"]
        x = x.numel * torch.tensor([], dtype=x.dtype).element_size()
        return x

    def _get_output_node(self, n):
        out_size = activation_size(n.meta["fwd_out"])
        out_node = [n.name] if out_size > 0 else []
        return out_size, out_node

    def _get_output_node_size(self, n):
        return self._get_output_node(n)[0]

    def _add_active_node(self, n, active_list):
        new_active = self._get_output_node(n)[1]
        if n.op == "placeholder" and get_node_shape(n) is not None:
            new_active.append(n.name)
        for i in new_active:
            if i not in active_list and get_node_shape(n) is not None:
                active_list.append(i)

    def _get_delete_node(self, user, user_to_last_uses, to_keep=None):
        delete_size = 0
        delete_node = []
        if user.op not in ("output",):
            nodes_to_delete = user_to_last_uses.get(user, [])
            if len(user.users) == 0:
                nodes_to_delete.append(user)
            if to_keep is not None:
                keep_list = []
                for n in nodes_to_delete:
                    if n.name in to_keep:
                        keep_list.append(n)
                for n in keep_list:
                    if n in nodes_to_delete:
                        nodes_to_delete.remove(n)
            if len(nodes_to_delete):
                out_node = [self._get_output_node(i) for i in nodes_to_delete]
                delete_size = sum([i[0] for i in out_node])
                for i in range(len(out_node)):
                    if out_node[i][0] > 0:
                        delete_node.append(out_node[i][1][0])
                    elif nodes_to_delete[i].op == "placeholder":
                        delete_node.append(nodes_to_delete[i].name)
                    # elif any(j in nodes_to_delete[i].name for j in ['transpose', 'permute', 'view']):
                    #     delete_node.append(nodes_to_delete[i].name)
        return delete_size, delete_node

    def _get_delete_node_size(self, user, user_to_last_uses, to_keep):
        return self._get_delete_node(user, user_to_last_uses, to_keep)[0]

    def _remove_deactive_node(self, user, user_to_last_uses, active_list):
        delete_node = self._get_delete_node(user, user_to_last_uses)[1]
        for i in delete_node:
            if i in active_list:
                active_list.remove(i)

    def _get_chunk_inputs_size(self, chunk_inputs, chunk_inputs_non_chunk, node_list, chunk_end_idx):
        nodes_to_delete = []
        for chunk_input in chunk_inputs + chunk_inputs_non_chunk:
            chunk_input_users = chunk_input.users.keys()
            chunk_input_users_idx = [self.node_mgr.find_node_idx(i) for i in chunk_input_users]
            if all(i <= chunk_end_idx for i in chunk_input_users_idx):
                if chunk_input not in nodes_to_delete:
                    nodes_to_delete.append(chunk_input)
        out_node = [self._get_output_node(i) for i in nodes_to_delete]
        delete_size = sum([i[0] for i in out_node])
        return delete_size

    def _get_last_usr(self, nodes):
        node_to_last_use: Dict[Node, Node] = {}
        user_to_last_uses: Dict[Node, List[Node]] = {}

        def register_last_uses(n: Node, user: Node):
            if n not in node_to_last_use:
                node_to_last_use[n] = user
                user_to_last_uses.setdefault(user, []).append(n)

        for node in reversed(nodes):
            map_arg(node.args, lambda n: register_last_uses(n, node))
            map_arg(node.kwargs, lambda n: register_last_uses(n, node))
        return user_to_last_uses

    def _get_contiguous_memory(self, node, not_contiguous_list, delete=False):
        mem = 0
        not_contiguous_ops = ["permute"]
        inherit_contiguous_ops = ["transpose", "view"]

        if node.op == "call_function" and any(n in node.name for n in ["matmul", "reshape"]):
            for n in node.args:
                if n in not_contiguous_list:
                    # matmul won't change origin tensor, but create a tmp copy
                    mem += self._get_output_node_size(n)
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
            return float(chunk_size) / node_shape[chunk_dim]

    def _get_chunk_delete_node_size(self, user, user_to_last_uses, chunk_ratio, chunk_inputs_names):
        # if any(j in user.name for j in ['transpose', 'permute', 'view']):
        #     return 0
        if user.op in ("placeholder", "output"):
            return 0
        nodes_to_delete = user_to_last_uses.get(user, [])
        if len(user.users) == 0:
            nodes_to_delete.append(user)
        delete_size = 0
        for n in nodes_to_delete:
            if n.name in chunk_inputs_names:
                continue
            delete_size += self._get_output_node_size(n) * chunk_ratio
        return delete_size

    def _print_mem_log(self, log, nodes, title=None):
        if title:
            print(title)
        for idx, (l, n) in enumerate(zip(log, nodes)):
            print("%s:%.2f \t" % (n.name, l), end="")
            if (idx + 1) % 3 == 0:
                print("")
        print("\n")

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

    def estimate_chunk_inference_mem(
        self,
        node_list: List,
        chunk_infos=None,
        print_mem=False,
    ):
        """
        Estimate inference memory with chunk

        Args:
            node_list (List): _description_
            chunk_infos (Dict): Chunk information. Defaults to None.
            print_mem (bool): Wether to print peak memory of every node. Defaults to False.

        Returns:
            act_memory_peak_log (List): peak memory of every node
            act_memory_after_node_log (List): memory after excuting every node
            active_node_list_log (List): active nodes of every node. active nodes refer to
                nodes generated but not deleted.
        """
        act_memory = 0.0
        act_memory_peak_log = []
        act_memory_after_node_log = []
        active_node_list = []
        active_node_list_log = []
        not_contiguous_list = []
        user_to_last_uses = self._get_last_usr(node_list)
        user_to_last_uses_no_free_var = self._get_last_usr(node_list)
        delete_free_var_from_last_use(user_to_last_uses_no_free_var)

        use_chunk = True if chunk_infos is not None else False
        chunk_within = False
        chunk_region_idx = None
        chunk_ratio = 1    # use it to estimate chunk mem
        chunk_inputs_names = []

        if use_chunk:
            chunk_regions = [i["region"] for i in chunk_infos]
            chunk_starts = [i[0] for i in chunk_regions]
            chunk_ends = [i[1] for i in chunk_regions]
            chunk_inputs = [i["inputs"] for i in chunk_infos]
            chunk_inputs_non_chunk = [i["inputs_non_chunk"] for i in chunk_infos]
            chunk_inputs_names = [j.name for i in chunk_inputs for j in i
                                 ] + [j.name for i in chunk_inputs_non_chunk for j in i]
            chunk_outputs = [i["outputs"] for i in chunk_infos]
            chunk_node_dim = [i["node_chunk_dim"] for i in chunk_infos]
            chunk_sizes = [i["chunk_size"] if "chunk_size" in i else 1 for i in chunk_infos]

        for idx, node in enumerate(node_list):
            # if node in chunk start nodes, change chunk ratio and add chunk_tensor
            if use_chunk and idx in chunk_starts:
                chunk_within = True
                chunk_region_idx = chunk_starts.index(idx)
                act_memory += sum(self._get_output_node_size(i) for i in chunk_outputs[chunk_region_idx]) / (1024**2)

            # determine chunk ratio for current node
            if chunk_within:
                chunk_ratio = self._get_chunk_ratio(
                    node,
                    chunk_node_dim[chunk_region_idx],
                    chunk_sizes[chunk_region_idx],
                )

            # if node is placeholder, just add the size of the node
            if node.op == "placeholder":
                act_memory += self._get_meta_node_size(node) * chunk_ratio / (1024**2)
                act_memory_peak_log.append(act_memory)
            # skip output
            elif node.op == "output":
                continue
            # no change for non compute node
            elif is_non_memory_node(node):
                act_memory_peak_log.append(act_memory)
            # node is a compute op
            # calculate tmp, output node and delete node memory
            else:
                # forward memory
                # TODO: contiguous_memory still not accurate for matmul, view, reshape and transpose
                act_memory += (self._get_contiguous_memory(node, not_contiguous_list) * chunk_ratio / (1024**2))
                act_memory += (self._get_output_node_size(node) * chunk_ratio / (1024**2))
                # record max act memory
                act_memory_peak_log.append(act_memory)
                # delete useless memory
                act_memory -= (self._get_contiguous_memory(node, not_contiguous_list, delete=True) * chunk_ratio /
                               (1024**2))
                # delete unused vars not in chunk_input_list
                # we can't delete input nodes until chunk ends
                if chunk_within:
                    act_memory -= self._get_chunk_delete_node_size(
                        node,
                        user_to_last_uses_no_free_var,
                        chunk_ratio,
                        chunk_inputs_names,
                    ) / (1024**2)
                else:
                    act_memory -= self._get_delete_node_size(node, user_to_last_uses_no_free_var,
                                                             chunk_inputs_names) / (1024**2)

            # log active node, only effective without chunk
            self._add_active_node(node, active_node_list)
            self._remove_deactive_node(node, user_to_last_uses, active_node_list)

            # if node in chunk end nodes, restore chunk settings
            if use_chunk and idx in chunk_ends:
                act_memory -= (self._get_output_node_size(node) * chunk_ratio / (1024**2))
                act_memory -= self._get_chunk_inputs_size(
                    chunk_inputs[chunk_region_idx],
                    chunk_inputs_non_chunk[chunk_region_idx],
                    node_list,
                    chunk_regions[chunk_region_idx][1],
                ) / (1024**2)
                chunk_within = False
                chunk_ratio = 1
                chunk_region_idx = None

            act_memory_after_node_log.append(act_memory)
            active_node_list_log.append(copy.deepcopy(active_node_list))

        if print_mem:
            print("with chunk" if use_chunk else "without chunk")
            # self._print_mem_log(act_memory_peak_log, node_list, "peak")
            # self._print_mem_log(act_memory_after_node_log, node_list, "after")
            self._print_compute_op_mem_log(act_memory_peak_log, node_list, "peak")
            # self._print_compute_op_mem_log(
            #     act_memory_after_node_log, node_list, "after"
            # )

        # param_memory = parameter_size(gm)
        # all_memory = act_memory + param_memory
        return act_memory_peak_log, act_memory_after_node_log, active_node_list_log

    def get_active_nodes(self, node_list: List) -> List:
        """
        Get active nodes for every node

        Args:
            node_list (List): _description_

        Returns:
            active_node_list_log (List): active nodes of every node. active nodes refer to
                nodes generated but not deleted.
        """
        active_node_list = []
        active_node_list_log = []
        user_to_last_uses = self._get_last_usr(node_list)
        user_to_last_uses_no_free_var = self._get_last_usr(node_list)
        delete_free_var_from_last_use(user_to_last_uses_no_free_var)
        for _, node in enumerate(node_list):
            # log active node, only effective without chunk
            self._add_active_node(node, active_node_list)
            self._remove_deactive_node(node, user_to_last_uses, active_node_list)
            active_node_list_log.append(copy.deepcopy(active_node_list))
        return active_node_list_log
