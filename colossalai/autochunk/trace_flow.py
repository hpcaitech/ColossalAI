from typing import Dict, List, Tuple

from torch.fx.node import Node

from .trace_indice import TraceIndice
from .utils import (
    NodeMgr,
    find_chunk_all_input_nodes,
    find_chunk_compute_input_and_output_nodes,
    find_tensor_shape_node,
    flat_list,
    get_node_name,
    get_node_shape,
    is_non_compute_node,
)


class TraceFlow(object):
    def __init__(self, trace_indice: TraceIndice, node_mgr: NodeMgr) -> None:
        self.trace_indice = trace_indice
        self.node_mgr = node_mgr

    def check_index_source(self, start_dim, start_node, start_idx, end_dim, end_node):
        """
        Check 2 given index: one index should be source of the other
        Args:
            start_idx(int): start node chunk dim
            start_node(node): start node
            end_idx(int): end node chunk dim
            end_node(node): end node

        Returns:
            bool: True if check pass
        """
        # we use start_node_idx instead of real chunk index
        start_node_idx = self.node_mgr.find_node_idx(start_node)
        end_node_trace = self.trace_indice._find_trace_from_node(end_node)
        end_node_trace_source = end_node_trace["source"][end_dim]
        sorted_source = sorted(end_node_trace_source.items(), key=lambda d: d[0], reverse=True)
        for node_idx, node_dim in sorted_source:
            if node_idx == start_node_idx and start_dim in node_dim:
                return True
            # it means we meet a node outside the loop, and the node is not input node
            if node_idx < start_node_idx:
                return False
        return False

    def check_index_compute(self, start_idx, end_dim, end_node, end_idx):
        """
        Check 2 given index: check they haven't been computed in the source trace.
        Args:
            start_idx(int): start node chunk dim
            start_node(node): start node
            end_idx(int): end node chunk dim
            end_node(node): end node

        Returns:
            bool: True if check pass
        """
        end_node_trace = self.trace_indice._find_trace_from_node(end_node)
        end_node_compute = end_node_trace["compute"][end_dim]
        if any(start_idx <= i <= end_idx for i in end_node_compute):
            return False
        return True

    def _assign_single_node_flow(
        self,
        arg_node: Node,
        start_idx: int,
        end_idx: int,
        cur_node: Node,
        cur_node_dim: int,
        cur_node_compute: Dict,
        cur_node_source: Dict,
        cur_node_fix_dim: List,
        all_node_info: Dict,
        next_node_list: List,
    ) -> bool:
        """
        Given the current node and one of its arg node,
        this function finds out arg node's chunk dim and fix dim

        Args:
            arg_node (Node): input node
            start_idx (int): chunk region start
            end_idx (int): chunk region end
            cur_node_dim (int): current node chunk dim
            cur_node_compute (Dict): current node compute dict
            cur_node_source (Dict): current node source dict
            cur_node_fix_dim (List): current node fix dim
            all_node_info (Dict): all node chunk info in the chunk region
            next_node_list (List)

        Returns:
            bool: True if this node can be added to the flow, vice versa.
        """
        arg_idx = self.node_mgr.find_node_idx(arg_node)
        # arg in chunk range or be inputs
        if not (start_idx <= arg_idx < end_idx):
            return True

        # get fix dim
        arg_fix_dim = []
        if cur_node_dim is not None:
            for i in cur_node_fix_dim:
                fix_dim_source = cur_node_source[i]
                if arg_idx in fix_dim_source:
                    arg_fix_dim.append(fix_dim_source[arg_idx][0])
        if arg_node in all_node_info:
            arg_fix_dim = list(set(all_node_info[arg_node]["fix_dim"] + arg_fix_dim))

        # find arg dim
        if cur_node_dim is not None:
            # dim is computed
            if arg_idx in cur_node_compute[cur_node_dim]:
                return False
            if arg_idx not in cur_node_source[cur_node_dim]:
                arg_dim = None
            else:
                arg_dim = cur_node_source[cur_node_dim][arg_idx][0]
                # chunk dim cannot be in fix dims
                if arg_dim in arg_fix_dim:
                    return False
                # chunk dim should be None if shape size is 1
                if get_node_shape(arg_node)[arg_dim] == 1:
                    arg_dim = None
                # chunk shape should equal cur node
                elif get_node_shape(arg_node)[arg_dim] != 1:
                    if cur_node_dim is not None and get_node_shape(cur_node)[cur_node_dim] != 1:
                        if get_node_shape(arg_node)[arg_dim] != get_node_shape(cur_node)[cur_node_dim]:
                            return False
        else:
            arg_dim = None

        # add arg rest dim as fix dim
        arg_fix_dim = list(range(len(get_node_shape(arg_node))))
        if arg_dim is not None:
            arg_fix_dim.remove(arg_dim)

        # if already in node_info, arg dim must be same
        if arg_node in all_node_info:
            if all_node_info[arg_node]["chunk_dim"] != arg_dim:
                return False
            all_node_info[arg_node]["fix_dim"] = arg_fix_dim
        # else add it to list
        else:
            all_node_info[arg_node] = {"chunk_dim": arg_dim, "fix_dim": arg_fix_dim}

        next_node_list.append(arg_node)
        return True

    def _get_all_node_info(self, end_dim, start_idx, end_idx):
        cur_node_list = [self.node_mgr.get_node_by_idx(end_idx)]  # start from the last node
        all_node_info = {cur_node_list[0]: {"chunk_dim": end_dim, "fix_dim": []}}

        while len(cur_node_list) > 0:
            next_node_list = []

            for cur_node in cur_node_list:
                # get cur node info
                cur_node_chunk_dim = all_node_info[cur_node]["chunk_dim"]
                cur_node_fix_dim = all_node_info[cur_node]["fix_dim"]
                if cur_node_chunk_dim is not None:
                    cur_node_compute = self.trace_indice._find_compute_trace_from_node(cur_node)
                    cur_node_source = self.trace_indice._find_source_trace_from_node(cur_node)
                else:
                    cur_node_compute = cur_node_source = None

                # get all valid args
                arg_list = []
                for arg in cur_node.all_input_nodes:
                    if type(arg) != type(cur_node):
                        continue
                    if is_non_compute_node(arg):
                        continue
                    if get_node_shape(arg) is None:
                        continue
                    arg_list.append(arg)
                    flow_flag = self._assign_single_node_flow(
                        arg,
                        start_idx,
                        end_idx,
                        cur_node,
                        cur_node_chunk_dim,
                        cur_node_compute,
                        cur_node_source,
                        cur_node_fix_dim,
                        all_node_info,
                        next_node_list,
                    )
                    if flow_flag == False:
                        return None

            cur_node_list = next_node_list
        return all_node_info

    def _get_input_nodes_dim(self, inputs: List[Node], start_idx: int, end_idx: int, all_node_info: Dict) -> Tuple:
        """
        Get chunk dim for every input node for their every entry, remove unchunked nodes

        Args:
            inputs (List[Node]): input nodes
            all_node_info (Dict): describe all node's chunk dim and fix dim
            start_idx (int): chunk start idx
            end_idx (int): chunk end idx

        Returns:
            inputs (List(Node)): new inputs
            inputs_dim (List): chunk dim for inputs
        """
        inputs_dim = []
        remove_inputs = []
        for input_node in inputs:
            input_dict = {}
            input_node_idx = self.node_mgr.find_node_idx(input_node)
            for user in input_node.users.keys():
                # skip non compute
                if is_non_compute_node(user):
                    continue
                # untraced node, mostly non compute
                if user not in all_node_info:
                    continue
                user_idx = self.node_mgr.find_node_idx(user)
                if start_idx <= user_idx <= end_idx:
                    chunk_dim = all_node_info[user]["chunk_dim"]
                    if chunk_dim is not None:
                        user_source = self.trace_indice._find_source_trace_from_node(user)[chunk_dim]
                        if input_node_idx in user_source:
                            if get_node_shape(input_node)[user_source[input_node_idx][0]] == 1:
                                input_dict[user_idx] = [None]
                            else:
                                input_dict[user_idx] = user_source[input_node_idx]
                        else:
                            return None, None
            if len(input_dict) == 0:
                remove_inputs.append(input_node)
            else:
                inputs_dim.append(input_dict)
        # remove unchunked inputs
        for i in remove_inputs:
            if i in inputs:
                inputs.remove(i)
        return inputs, inputs_dim

    def _get_prepose_nodes(self, all_node_info: Dict, start_idx: int, end_idx: int, chunk_info) -> List[Node]:
        """
        get all useless nodes in chunk region and prepose them

        Args:
            all_node_info (Dict): describe all node's chunk dim and fix dim
            start_idx (int): chunk start idx
            end_idx (int): chunk end idx

        Returns:
            List[Node]: all nodes to be preposed
        """
        # get all possible prepose nodes
        maybe_prepose_nodes = []
        for node, node_info in all_node_info.items():
            if node_info["chunk_dim"] is None:
                maybe_prepose_nodes.append(node)
        for node in self.node_mgr.get_node_slice_by_idx(start_idx, end_idx):
            if node not in all_node_info and node not in chunk_info["outputs"]:
                maybe_prepose_nodes.append(node)
        maybe_prepose_nodes.sort(
            key=lambda x: self.node_mgr.find_node_idx(x),
            reverse=True,
        )  # from last node to first node
        prepose_nodes = []
        # set every node as root, search its args, if all legal, turn root and args as prepose nodes
        while len(maybe_prepose_nodes) > 0:
            tmp_cur_prepose_nodes = [maybe_prepose_nodes[0]]
            tmp_cur_related_prepose_nodes = []
            prepose_flag = True

            # loop cur node's all arg until out of chunk
            while len(tmp_cur_prepose_nodes) > 0:
                if prepose_flag == False:
                    break
                tmp_next_prepose_nodes = []
                tmp_cur_related_prepose_nodes.extend(tmp_cur_prepose_nodes)
                for cur_prepose_node in tmp_cur_prepose_nodes:
                    if prepose_flag == False:
                        break
                    for cur_prepose_node_arg in cur_prepose_node.all_input_nodes:
                        if type(cur_prepose_node_arg) != type(cur_prepose_node):
                            continue
                        # out of loop
                        if not (start_idx <= self.node_mgr.find_node_idx(cur_prepose_node_arg) < end_idx):
                            continue
                        # compute op in loop
                        elif cur_prepose_node_arg in all_node_info:
                            if all_node_info[cur_prepose_node_arg]["chunk_dim"] is None:
                                tmp_next_prepose_nodes.append(cur_prepose_node_arg)
                            else:
                                prepose_flag = False
                                break
                        # non compute op
                        else:
                            tmp_next_prepose_nodes.append(cur_prepose_node_arg)
                tmp_cur_prepose_nodes = tmp_next_prepose_nodes

            if prepose_flag == False:
                maybe_prepose_nodes.remove(maybe_prepose_nodes[0])
                continue
            else:
                for n in tmp_cur_related_prepose_nodes:
                    if n not in prepose_nodes:
                        prepose_nodes.append(n)
                    if n in maybe_prepose_nodes:
                        maybe_prepose_nodes.remove(n)
        # sort by index
        prepose_nodes.sort(key=lambda x: self.node_mgr.find_node_idx(x))
        chunk_info["args"]["prepose_nodes"] = prepose_nodes

    def _get_non_chunk_inputs(self, chunk_info, start_idx, end_idx):
        # we need to log input nodes to avoid deleting them in the loop
        chunk_node_list = self.node_mgr.get_node_slice_by_idx(start_idx, end_idx + 1)
        # also need to get some prepose node's arg out of non_chunk_inputs
        for n in chunk_info["args"]["prepose_nodes"]:
            chunk_node_list.remove(n)
        non_chunk_inputs = find_chunk_all_input_nodes(chunk_node_list)
        for i in non_chunk_inputs:
            if i not in chunk_info["inputs"]:
                chunk_info["inputs_non_chunk"].append(i)
        return chunk_info

    def flow_search(self, start_idx, start_dim, end_idx, end_dim):
        inputs, outputs = find_chunk_compute_input_and_output_nodes(
            self.node_mgr.get_node_slice_by_idx(start_idx, end_idx + 1)
        )

        # get every node's chunk dim and fix dim
        all_node_info = self._get_all_node_info(end_dim, start_idx, end_idx)
        if all_node_info is None:
            return None

        chunk_info = {
            "region": (start_idx, end_idx),
            "inputs": [],
            "inputs_non_chunk": [],
            "inputs_dim": [],
            "outputs": [self.node_mgr.get_node_by_idx(end_idx)],
            "outputs_non_tensor": {},
            "outputs_dim": [end_dim],
            "node_chunk_dim": all_node_info,
            "args": {},
        }

        # find chunk info for other outputs
        if len(find_tensor_shape_node(outputs)) > 1:
            chunk_info = self._get_other_output_info(outputs, start_idx, start_dim, end_idx, end_dim, chunk_info)
            if chunk_info is None:
                return None

        # get input nodes' chunk dim
        inputs, inputs_dim = self._get_input_nodes_dim(inputs, start_idx, end_idx, all_node_info)
        if inputs is None:
            return None
        chunk_info["inputs"] = inputs
        chunk_info["inputs_dim"] = inputs_dim

        # move useless nodes ahead of loop
        self._get_prepose_nodes(all_node_info, start_idx, end_idx, chunk_info)

        # find non chunk inputs
        chunk_info = self._get_non_chunk_inputs(chunk_info, start_idx, end_idx)

        # reassign reshape size, some size may have changed due to chunk
        chunk_info = self._reassign_reshape_size(chunk_info)

        return chunk_info

    def _get_other_output_info(
        self, outputs: List[Node], start_idx: int, start_dim: int, end_idx: int, end_dim: int, chunk_info: Dict
    ):
        start_node = self.node_mgr.get_node_by_idx(start_idx)
        # loop all outputs
        for output in outputs:
            output_legal = False
            output_idx = self.node_mgr.find_node_idx(output)
            # skip the origin output
            if output_idx == end_idx:
                continue
            # skip non tensor
            if get_node_shape(output) is None:
                # log shape tensor
                if len(output.meta["fwd_out"]) > 0 and isinstance(output.meta["fwd_out"][0], int):
                    chunk_info["outputs_non_tensor"][output] = str(output.meta["fwd_out"])
                continue
            # loop every dim of outputs, try to find a legal one
            for output_dim in range(len(get_node_shape(output))):
                if not self.check_region_start_end(start_node, start_dim, start_idx, output, output_dim, output_idx):
                    continue
                new_all_node_info = self._get_all_node_info(output_dim, start_idx, output_idx)
                if new_all_node_info is None:
                    continue
                # check node info legal
                if self._update_chunk_info(chunk_info, new_all_node_info, output, output_dim) == True:
                    output_legal = True
                    break
            # not legal
            if output_legal == False:
                return None
        return chunk_info

    def _update_chunk_info(self, chunk_info: Dict, new_all_node_info: Dict, output: Node, output_dim: int) -> bool:
        """
        check if there is conflict between new node info and old chunk info. If not, update old chunk info
        """
        # check if conflict
        overlap_flag = False
        for k, v in new_all_node_info.items():
            if k in chunk_info["node_chunk_dim"]:
                overlap_flag = True
                if chunk_info["node_chunk_dim"][k]["chunk_dim"] != v["chunk_dim"]:
                    return False
        # if no overlap, we just consider them as prepose nodes, instead of new output
        if overlap_flag == False:
            return True
        # update chunk info
        for k, v in new_all_node_info.items():
            if k in chunk_info["node_chunk_dim"]:
                chunk_info["node_chunk_dim"][k]["fix_dim"] = list(
                    set(chunk_info["node_chunk_dim"][k]["fix_dim"] + v["fix_dim"])
                )
            else:
                chunk_info["node_chunk_dim"][k] = v
        chunk_info["outputs"].append(output)
        chunk_info["outputs_dim"].append(output_dim)
        return True

    def _reassign_reshape_size(self, chunk_info):
        """
        Some shape args in reshape may have changed due to chunk
        reassign those changed shape
        """
        chunk_region = chunk_info["region"]
        reshape_size = {}
        chunk_shape = get_node_shape(chunk_info["outputs"][0])[chunk_info["outputs_dim"][0]]
        for node in self.node_mgr.get_node_slice_by_idx(chunk_region[0], chunk_region[1] + 1):
            if any(i == get_node_name(node) for i in ["reshape", "view"]):
                if node in chunk_info["args"]["prepose_nodes"]:
                    continue
                if node.args[0] in chunk_info["inputs_non_chunk"]:
                    continue
                reshape_args = flat_list(node.args[1:])
                if (
                    len(reshape_args) == 1
                    and get_node_shape(reshape_args[0]) is None
                    and len(reshape_args[0].meta["fwd_out"]) > 1
                ):
                    continue
                chunk_dim = chunk_info["node_chunk_dim"][node]["chunk_dim"]
                new_shape = ""
                for reshape_arg_dim, reshape_arg in enumerate(reshape_args):
                    if reshape_arg_dim == chunk_dim:
                        new_shape += "min(chunk_size, %d - chunk_idx), " % chunk_shape
                    else:
                        if isinstance(reshape_arg, int):
                            new_shape += "%s, " % str(reshape_arg)
                        else:
                            new_shape += "%s, " % reshape_arg.name
                new_shape = new_shape[:-2]
                origin_shape = str(reshape_args)[1:-1]
                reshape_size[node.name] = [origin_shape, new_shape]
        chunk_info["reshape_size"] = reshape_size
        return chunk_info

    def check_region_start_end(
        self, start_node: Node, start_dim: int, start_idx: int, end_node: Node, end_dim: int, end_idx: int
    ) -> bool:
        """
        check if region start and end is legal
        """
        # dim cannot be None
        if get_node_shape(end_node) is None or get_node_shape(start_node) is None:
            return False
        # dim size cannot be 1
        if get_node_shape(end_node)[end_dim] == 1 or get_node_shape(start_node)[start_dim] == 1:
            return False
        # must have users
        if len(end_node.users) == 0:
            return False
        # check index source align
        if not self.check_index_source(start_dim, start_node, start_idx, end_dim, end_node):
            return False
        # check index compute
        if not self.check_index_compute(start_idx, end_dim, end_node, end_idx):
            return False
        return True
