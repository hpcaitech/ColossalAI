import copy

from .utils import (
    find_chunk_all_input_nodes,
    find_chunk_compute_input_and_output_nodes,
    find_idx_by_name,
    get_node_shape,
    is_non_compute_node,
    is_non_compute_node_except_placeholder,
)


class IndexTracer(object):
    def __init__(self, node_list) -> None:
        self.node_list = node_list
        self.idx_trace_list = self._init_idx_trace_list()
        self.idx_trace_equal = []
        self.idx_view_list = {}
        self.idx_count = -1
        self.all_reorder_map = {i: i for i in range(len(self.idx_trace_list))}

    def _init_idx_trace_list(self):
        idx_trace_list = []
        for n in self.node_list:
            if get_node_shape(n) != None:
                cur_trace = {
                    "idx": [None for _ in range(len(get_node_shape(n)))],
                    "compute": [[] for _ in range(len(get_node_shape(n)))],
                    "source": [{} for _ in range(len(get_node_shape(n)))],
                }
            else:
                cur_trace = {"idx": [], "compute": [], "source": []}
            idx_trace_list.append(cur_trace)
        return idx_trace_list

    def _add_index(self):
        """
        Update the count and return it. To record the idx number.

        Returns:
            idx_count: int
        """
        self.idx_count += 1
        return self.idx_count

    def _del_dim(self, idx, dim_idx):
        self.idx_trace_list[idx]["idx"].pop(dim_idx)
        self.idx_trace_list[idx]["compute"].pop(dim_idx)
        self.idx_trace_list[idx]["source"].pop(dim_idx)

    def _add_dim(self, node_idx, dim_idx):
        self.idx_trace_list[node_idx]["idx"].insert(dim_idx, self._add_index())
        self.idx_trace_list[node_idx]["compute"].insert(dim_idx, [])
        self.idx_trace_list[node_idx]["source"].insert(dim_idx, {})

    def _transform_index(self, node, node_dim):
        node_idx = self._find_idx_trace_from_node(node)
        dims = list(range(len(node_idx)))
        return dims[node_dim]

    def _inherit_index(self, node_from, node_from_dim, node_to, node_to_dim):
        node_from_dim = self._transform_index(node_from, node_from_dim)
        node_to_dim = self._transform_index(node_to, node_to_dim)
        node_from_trace = self._find_trace_from_node(node_from)
        node_to_trace = self._find_trace_from_node(node_to)
        node_to_trace["idx"][node_to_dim] = node_from_trace["idx"][node_from_dim]
        node_to_trace["compute"][node_to_dim] = copy.deepcopy(
            node_from_trace["compute"][node_from_dim]
        )
        self._add_source(node_from, node_from_dim, node_to, node_to_dim, init=True)

    def _inherit_all_computation(self, node_from, node_to):
        node_from_compute = self._find_compute_trace_from_node(node_from)
        node_to_compute = self._find_compute_trace_from_node(node_to)
        assert len(node_from_compute) == len(node_to_compute)
        for i in range(len(node_from_compute)):
            self._add_source(node_from, i, node_to, i)
            node_to_compute[i] = copy.deepcopy(node_from_compute[i])

    def _add_source(self, node_from, node_from_dim, node_to, node_to_dim, init=False):
        node_from_dim = self._transform_index(node_from, node_from_dim)
        node_from_trace_source = self._find_source_trace_from_node(node_from)
        node_to_dim = self._transform_index(node_to, node_to_dim)
        node_to_trace_source = self._find_source_trace_from_node(node_to)
        node_from_idx = find_idx_by_name(node_from.name, self.node_list)
        if init:
            node_to_trace_source[node_to_dim] = {}
        # add dim to cur new source
        if node_from_idx not in node_to_trace_source[node_to_dim]:
            node_to_trace_source[node_to_dim][node_from_idx] = [node_from_dim]
        else:
            if node_from_dim not in node_to_trace_source[node_to_dim][node_from_idx]:
                node_to_trace_source[node_to_dim][node_from_idx].append(node_from_dim)
        # update inputs source
        for node_idx, node_dim in node_from_trace_source[node_from_dim].items():
            if node_idx not in node_to_trace_source[node_to_dim]:
                node_to_trace_source[node_to_dim][node_idx] = copy.deepcopy(node_dim)
            else:
                for d in node_dim:
                    if d not in node_to_trace_source[node_to_dim][node_idx]:
                        node_to_trace_source[node_to_dim][node_idx].append(d)

    def _mark_computation_from_node(self, node_from, node_to, exclude=None):
        if exclude == None:
            exclude = []
        else:
            exclude = [self._transform_index(node_to, i) for i in exclude]
        node_from_compute = self._find_compute_trace_from_node(node_from)
        node_to_compute = self._find_compute_trace_from_node(node_to)
        # assert len(node_from_compute) == len(node_to_compute)
        for i in range(-1, -min(len(node_from_compute), len(node_to_compute)) - 1, -1):
            if self._transform_index(node_to, i) in exclude:
                continue
            self._add_source(node_from, i, node_to, i)
            for j in node_from_compute[i]:
                if j not in node_to_compute[i]:
                    node_to_compute[i].append(j)

    def _mark_idx_equal(self, node1, dim1, node2, dim2):
        """
        Mark 2 index to be equal.

        Args:
            idx1 (int): index count.
            idx2 (int): index count.
        """
        # node1_idx = _find_idx_by_name(node1.name, self.nodes_list)
        # node2_idx = _find_idx_by_name(node2.name, self.nodes_list)
        # if node1_idx > node2_idx:
        #     self._add_source(node2, dim2, node1, dim1)
        # else:
        #     self._add_source(node1, dim1, node2, dim2)

    def _mark_computation(self, node, idx, dim):
        """
        Mark some dims of node as computed.

        Args:
            node (node)
            idx (int): node index
            dim (list or int): dims to be marked as computed
        """
        if isinstance(dim, int):
            dim = [dim]
        dims = list(range(len(get_node_shape(node))))
        for d in dim:
            cur_dim = dims[d]
            if idx not in self.idx_trace_list[idx]["compute"][cur_dim]:
                self.idx_trace_list[idx]["compute"][cur_dim].append(idx)

    def _find_trace_from_node(self, node):
        """
        Find node idx and compute trace by the node.

        Args:
            node (node)
        Returns:
            idx (list): idx of the node
            compute (list): computed idx of the node.
        """
        node_idx = find_idx_by_name(node.name, self.node_list)
        node_dict = self.idx_trace_list[node_idx]
        return node_dict

    def _find_source_trace_from_node(self, node):
        """
        Find node source trace by the node.

        Args:
            node (node)
        Returns:
            idx (list): idx of the node
            compute (list): computed idx of the node.
        """
        node_idx = find_idx_by_name(node.name, self.node_list)
        node_dict = self.idx_trace_list[node_idx]
        return node_dict["source"]

    def _find_idx_trace_from_node(self, node):
        """
        Find node idx trace by the node.

        Args:
            node (node)
        Returns:
            idx (list): idx of the node
        """
        node_idx = find_idx_by_name(node.name, self.node_list)
        return self.idx_trace_list[node_idx]["idx"]

    def _find_compute_trace_from_node(self, node):
        """
        Find node compute trace by the node.

        Args:
            node (node)
        Returns:
            compute (list): computed idx of the node.
        """
        node_idx = find_idx_by_name(node.name, self.node_list)
        return self.idx_trace_list[node_idx]["compute"]

    def _assign_index_as_input(self, node, node_idx, input_node=None):
        """
        Assign node's trace as its input node.

        Args:
            node (node)
            node_idx (int)
        """
        if input_node == None:
            input_node = node.args[0]
        input_node_idx = find_idx_by_name(input_node.name, self.node_list)
        input_node_idx_trace = self.idx_trace_list[input_node_idx]["idx"]

        new_idx_trace = copy.deepcopy(input_node_idx_trace)
        self.idx_trace_list[node_idx]["idx"] = new_idx_trace

        self._inherit_all_computation(input_node, node)

    def _assign_all_index(self, node, node_idx):
        """
        Add new index for all node's dims.

        Args:
            node (node)
            node_idx (int)
        """
        shape = node.meta["tensor_meta"].shape
        new_trace = []
        for _ in shape:
            new_trace.append(self._add_index())
        self.idx_trace_list[node_idx]["idx"] = new_trace

    def _assign_transpose_index(self, node, node_idx):
        """
        Assign index for transpose op.
        1. swap input's dim according to transpose args
        2. inherit input's computation

        Args:
            node (node)
            node_idx (int)
        """
        input_node = node.args[0]
        tranpose_dim = node.args[1:]

        self._assign_index_as_input(node, node_idx, input_node)
        self._inherit_index(input_node, tranpose_dim[1], node, tranpose_dim[0])
        self._inherit_index(input_node, tranpose_dim[0], node, tranpose_dim[1])

    def _assign_permute_index(self, node, node_idx):
        """
        Assign index for permute op.
        1. swap input's dim according to permute args
        2. inherit input's computation

        Args:
            node (node)
            node_idx (int)
        """
        permute_dim = node.args[1:]
        input_node = node.args[0]

        self._assign_index_as_input(node, node_idx, input_node)
        for idx, d in enumerate(permute_dim):
            self._inherit_index(input_node, d, node, idx)

    def _assign_linear_index(self, node, node_idx):
        """
        Assign index for linear op.
        1. copy trace from input node and change last index accroding to weight
        2. mark equal for input node last index, weight first dim and bias dim.
        3. inherit input's computation, mark computation for last dim.

        Args:
            node (node)
            node_idx (int)
        """
        if len(node.args) == 2:
            input_node, weight = node.args
            bias = None
        else:
            input_node, weight, bias = node.args

        self._assign_index_as_input(node, node_idx)
        self._inherit_index(weight, 1, node, -1)

        self._mark_computation(node, node_idx, [-1])
        self._mark_idx_equal(input_node, -1, weight, 0)

        if bias:
            self._mark_idx_equal(input_node, -1, bias, 0)

    def _assign_matmul_index(self, node, node_idx):
        """
        Assign index for matmul op.
        1. copy trace from matmul_left and change last index accroding to matmul_right. (assert they have same length)
        2. mark equal for input matmul_left -1 index and matmul_right -2 dim.
        3. inherit matmul_left and matmul_right computation, mark computation for last dim.

        Args:
            node (node)
            node_idx (int)
        """
        matmul_left, matmul_right = node.args

        assert len(get_node_shape(matmul_left)) == len(get_node_shape(matmul_right))
        self._assign_index_as_input(node, node_idx, matmul_left)
        self._inherit_index(matmul_right, -1, node, -1)

        self._mark_computation_from_node(matmul_right, node, [-1, -2])
        self._mark_computation(node, node_idx, [-1])
        self._mark_idx_equal(matmul_left, -1, matmul_right, -2)

    def _assign_layernorm_index(self, node, idx):
        """
        Assign index for layernorm op.
        1. assign index as input node
        2. inherit computation and mark last 2 dims as computed.

        Args:
            node (node)
            node_idx (int)
        """
        self._assign_index_as_input(node, idx)
        self._mark_computation(node, idx, [-1])

    def _assign_elementwise_index(self, node, idx):
        """
        Assign index for element-wise op (eg. relu sigmoid add mul).
        1. assign index as input node
        2. inherit computation from all input nodes.

        Args:
            node (node)
            node_idx (int)
        """
        self._assign_index_as_input(node, idx)
        nodes_in = []
        for node_in in node.args:
            if type(node_in) == type(node):
                nodes_in.append(node_in)
                self._mark_computation_from_node(node_in, node)
        assert len(nodes_in) <= 2
        if len(nodes_in) == 2:
            node_in0_shape = get_node_shape(nodes_in[0])
            node_in1_shape = get_node_shape(nodes_in[1])
            for i in range(-1, -min(len(node_in0_shape), len(node_in1_shape)) - 1, -1):
                if node_in0_shape[i] == node_in1_shape[i]:
                    self._mark_idx_equal(nodes_in[0], i, nodes_in[1], i)

    def _assgin_no_change_index(self, node, idx):
        self._assign_index_as_input(node, idx)
        for node_in in node.args:
            if type(node_in) == type(node):
                self._mark_computation_from_node(node_in, node)

    def _assign_einsum_index(self, node, idx):
        """
        Assign index for einsum op.

        Args:
            node (node)
            node_idx (int)
        """
        patterns = node.args[0]
        input_nodes = node.args[1:]

        patterns = patterns.replace(" ", "")
        left, right = patterns.split("->")
        left = left.split(",")

        all_index = []
        for i in left:
            for c in i:
                all_index.append(c)
        all_index = set(all_index)
        free_index = set([i for i in right])
        sum_index = all_index - free_index

        for right_idx, right_indice in enumerate(right):
            for left_idx, left_str in enumerate(left):
                if right_indice in left_str:
                    source_idx = left_str.index(right_indice)
                    self._inherit_index(
                        input_nodes[left_idx], source_idx, node, right_idx
                    )

        # for i in sum_index:
        #     for left_idx, left_str in enumerate(left):
        #         if i in left_str:
        #             self._mark_computation(node, idx, left_str.index(i))
        #             break

    def _assign_softmax_index(self, node, idx):
        """
        Assign index for softmax op.
        1. assign index as input node
        2. inherit computation and mark softmax dim as computed.

        Args:
            node (node)
            node_idx (int)
        """
        self._assign_index_as_input(node, idx)
        self._mark_computation(node, idx, [node.kwargs["dim"]])

    def _assign_unsqueeze_index(self, node, node_idx):
        """
        Assign index for unsqueeze op.
        1. assign new index for unsqueeze dim

        Args:
            node (node)
            node_idx (int)
        """
        self._del_dim(node_idx, -1)
        self._assign_index_as_input(node, node_idx)
        self._add_dim(node_idx, node.args[1])

    def _assign_dropout_index(self, node, node_idx):
        """
        Assign index for unsqueeze op.
        1. assign new index for unsqueeze dim

        Args:
            node (node)
            node_idx (int)
        """
        self._assign_index_as_input(node, node_idx)

    def _assign_ones_like_index(self, node, node_idx):
        """
        Assign index for oneslike op.
        1. assign new index for all dim

        Args:
            node (node)
            node_idx (int)
        """
        self._assign_all_index(node, node_idx)

    def _assign_view_reshape_index(self, node, node_idx):
        """
        Assign index for view and reshape op.
        1. get origin shape and target shape by meta info.
        2. compute the real value of -1 in target shape.
        3. determine changed dim, and assgin index for generated dim.
        4. log changed dim and generated dim for restore
        5. inherit computation.
        6. TODO: look into view list to see whether the view is associated with other,
           if so assgin equal dim according to previous view.

        Args:
            node (node)
            node_idx (int)
        """
        # get data, turn into number
        origin_node = node.args[0]
        origin_shape = origin_node.meta["tensor_meta"].shape
        target_shape = []
        for i in range(1, len(node.args)):
            if isinstance(node.args[i], int):
                target_shape.append(node.args[i])
            else:
                target_shape.append(node.args[i].meta["fwd_out"][0])

        # compute the value of -1
        if -1 in target_shape:
            origin_product = 1
            for i in origin_shape:
                origin_product *= i
            target_product = -1
            for i in target_shape:
                target_product *= i
            shape_idx = target_shape.index(-1)
            target_shape[shape_idx] = origin_product // target_product

        # determine changed dim
        len_diff = len(origin_shape) - len(target_shape)
        if len_diff == 1:
            # dim merge
            dim_equal = [i == j for i, j in zip(origin_shape[:-1], target_shape)]
            dim_to = [dim_equal.index(False)]
            dim_from = [dim_equal.index(False), dim_equal.index(False) + 1]
            self._add_dim(node_idx, -1)
        elif len_diff == -1:
            # dim expand
            dim_equal = [i == j for i, j in zip(origin_shape, target_shape[:-1])]
            dim_from = [dim_equal.index(False)]
            dim_to = [dim_equal.index(False), dim_equal.index(False) + 1]
            self._del_dim(node_idx, -1)
        else:
            raise NotImplementedError(
                "shape"
                + str(origin_shape)
                + "and"
                + str(target_shape)
                + "view not implemented"
            )

        # get new index
        origin_trace = self._find_idx_trace_from_node(origin_node)
        self._assign_index_as_input(node, node_idx, origin_node)
        dim_from.reverse()
        for i in dim_from:
            self._del_dim(node_idx, i)
        for i in dim_to:
            self._add_dim(node_idx, i)

        # inherit computation
        compute_log = self._find_compute_trace_from_node(origin_node)
        for i in dim_from:
            if origin_trace[i] in compute_log:
                for j in dim_to:
                    self._mark_computation(node, node_idx, [j])
                break

        # log view, not used now
        view_dict = {
            "idx_from": [origin_trace[i] for i in dim_from],
            "dim_from": dim_from,
            "idx_to": [self.idx_trace_list[node_idx]["idx"][i] for i in dim_to],
            "dim_to": dim_to,
        }
        self.idx_view_list[node] = view_dict

    def _merge_equal_idx(self):
        idx_equal = copy.deepcopy(self.idx_trace_equal)
        idx_equal.reverse()
        for idx in idx_equal:
            merge_to = min(idx)
            merge_from = max(idx)
            for trace in self.idx_trace_list:
                if merge_from in trace["idx"]:
                    trace["idx"] = [
                        merge_to if i == merge_from else i for i in trace["idx"]
                    ]

    def trace_index(self):
        for idx, node in enumerate(self.node_list):
            if node.op == "placeholder":
                self._assign_all_index(node, idx)
            elif node.op == "call_method":
                if "transpose" in node.name:
                    self._assign_transpose_index(node, idx)
                elif "permute" in node.name:
                    self._assign_permute_index(node, idx)
                elif "view" in node.name or "reshape" in node.name:
                    self._assign_view_reshape_index(node, idx)
                elif "unsqueeze" in node.name:
                    self._assign_unsqueeze_index(node, idx)
                elif any(i in node.name for i in ["to", "contiguous"]):
                    self._assgin_no_change_index(node, idx)
                else:
                    raise NotImplementedError(node.name, "method not implemented yet!")
            elif node.op == "call_function":
                if "linear" in node.name:
                    self._assign_linear_index(node, idx)
                elif "matmul" in node.name:
                    self._assign_matmul_index(node, idx)
                elif "softmax" in node.name:
                    self._assign_softmax_index(node, idx)
                elif any(n in node.name for n in ["mul", "add", "sigmoid", "relu"]):
                    self._assign_elementwise_index(node, idx)
                elif "ones_like" in node.name:
                    self._assign_ones_like_index(node, idx)
                elif "dropout" in node.name:
                    self._assign_dropout_index(node, idx)
                elif "einsum" in node.name:
                    self._assign_einsum_index(node, idx)
                elif "getattr" in node.name:
                    continue  # get attr like shape
                elif "getitem" in node.name:
                    continue  # get item in list
                else:
                    raise NotImplementedError(
                        node.name, "function not implemented yet!"
                    )
            elif node.op == "call_module":
                if any(n in node.name for n in ["layernorm", "norm"]):
                    self._assign_layernorm_index(node, idx)
                else:
                    raise NotImplementedError(node.name, "module not implemented yet!")
            elif node.op == "get_attr":
                self._assign_all_index(node, idx)  # get param
            elif node.op == "output":
                continue
            else:
                raise NotImplementedError(node.op, "op not implemented yet!")
        # self._merge_equal_idx()

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
        start_node_idx = find_idx_by_name(start_node.name, self.node_list)
        end_node_trace = self._find_trace_from_node(end_node)
        end_node_trace_source = end_node_trace["source"][end_dim]
        sorted_source = sorted(
            end_node_trace_source.items(), key=lambda d: d[0], reverse=True
        )
        for node_idx, node_dim in sorted_source:
            if node_idx == start_node_idx and start_dim in node_dim:
                return True
            # it means we meet a node outside the loop, and the node is not input node
            if node_idx < start_idx:
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
        end_node_trace = self._find_trace_from_node(end_node)
        end_node_compute = end_node_trace["compute"][end_dim]
        if any(start_idx <= i <= end_idx for i in end_node_compute):
            return False
        return True

    def get_node_chunk_dim(self, node_from, node_from_dim, node_to):
        node_from_source = self._find_source_trace_from_node(node_from)
        dim_source = node_from_source[node_from_dim]
        node_to_idx = find_idx_by_name(node_to.name, self.node_list)
        for k, v in dim_source.items():
            if k == node_to_idx:
                return v
        return None

    def _find_inherit_dim(self, input_node, input_dim, node):
        input_node_idx = find_idx_by_name(input_node.name, self.node_list)
        node_trace_source = self._find_source_trace_from_node(node)
        for node_dim in range(len(get_node_shape(node))):
            if (
                input_node_idx in node_trace_source[node_dim]
                and input_dim[0] in node_trace_source[node_dim][input_node_idx]
            ):
                return node_dim
        return None

    def check_index_duplicate(self, chunk_infos, return_dim=False):
        input_dim_after_node = {}
        for input_node_idx, input_node in enumerate(chunk_infos["inputs"]):
            for k, v in chunk_infos["inputs_dim"][input_node_idx].items():
                inherit_dim = self._find_inherit_dim(input_node, v, self.node_list[k])
                if inherit_dim:
                    input_dim_after_node[k] = inherit_dim

        for node in self.node_list[
            chunk_infos["region"][0] : chunk_infos["region"][1] + 1
        ]:
            if is_non_compute_node_except_placeholder(node):
                continue
            count = 0
            duplicate_dims = []
            node_trace_source = self._find_source_trace_from_node(node)
            for node_dim in range(len(get_node_shape(node))):
                duplicate_dim = []
                duplicate_flag = False
                dim_source = node_trace_source[node_dim]
                for k, v in dim_source.items():
                    if chunk_infos["region"][0] <= k <= chunk_infos["region"][1]:
                        if k in input_dim_after_node and input_dim_after_node[k] in v:
                            duplicate_flag = True
                            duplicate_dim.append((k, v))
                duplicate_dims.append(duplicate_dim)
                if duplicate_flag:
                    count += 1

            if count > 1:
                if return_dim:
                    return False, duplicate_dims
                else:
                    return False
        if return_dim:
            return True, None
        else:
            return True

    def _assgin_single_node_flow(
        self,
        arg_node,
        start_idx,
        end_idx,
        cur_node_dim,
        cur_node_compute,
        cur_node_source,
        cur_node_fix_dim,
        all_node_info,
        next_node_list,
    ):
        arg_idx = find_idx_by_name(arg_node.name, self.node_list)
        # arg in chunk range or be inputs
        if not (start_idx <= arg_idx < end_idx):
            return True

        # find arg dim
        if cur_node_dim is not None:
            # dim is computed
            if arg_idx in cur_node_compute[cur_node_dim]:
                return False
            if arg_idx not in cur_node_source[cur_node_dim]:
                arg_dim = None
            else:
                arg_dim = cur_node_source[cur_node_dim][arg_idx][0]
        else:
            arg_dim = None

        # get fix dim
        arg_fix_dim = []
        if cur_node_dim is not None:
            for i in cur_node_fix_dim:
                fix_dim_source = cur_node_source[i]
                if arg_idx in fix_dim_source:
                    arg_fix_dim.append(fix_dim_source[arg_idx][0])

        # if already in node_info, arg dim must be same
        if arg_node in all_node_info:
            if all_node_info[arg_node]["chunk_dim"] != arg_dim:
                return False
            all_node_info[arg_node]["fix_dim"] = list(
                set(all_node_info[arg_node]["fix_dim"] + arg_fix_dim)
            )
        # else add it to list
        else:
            all_node_info[arg_node] = {"chunk_dim": arg_dim, "fix_dim": arg_fix_dim}

        next_node_list.append(arg_node)
        return True

    def _get_all_node_info(self, end_dim, start_idx, end_idx):
        cur_node_list = [self.node_list[end_idx]]  # start from the last node
        all_node_info = {cur_node_list[0]: {"chunk_dim": end_dim, "fix_dim": []}}

        while len(cur_node_list) > 0:
            next_node_list = []

            for cur_node in cur_node_list:
                # get cur node info
                cur_node_chunk_dim = all_node_info[cur_node]["chunk_dim"]
                cur_node_fix_dim = all_node_info[cur_node]["fix_dim"]
                if cur_node_chunk_dim:
                    cur_node_compute = self._find_compute_trace_from_node(cur_node)
                    cur_node_source = self._find_source_trace_from_node(cur_node)
                else:
                    cur_node_compute = cur_node_source = None

                # get all valid args
                arg_list = []
                for arg in cur_node.args:
                    if type(arg) != type(cur_node):
                        continue
                    if is_non_compute_node(arg):
                        continue
                    arg_list.append(arg)
                    flow_flag = self._assgin_single_node_flow(
                        arg,
                        start_idx,
                        end_idx,
                        cur_node_chunk_dim,
                        cur_node_compute,
                        cur_node_source,
                        cur_node_fix_dim,
                        all_node_info,
                        next_node_list,
                    )
                    if flow_flag == False:
                        return None

                if len(arg_list) == 2:
                    if any(i in cur_node.name for i in ["add", "mul"]):
                        for arg in arg_list:
                            if not (
                                start_idx
                                <= find_idx_by_name(arg.name, self.node_list)
                                < end_idx
                            ):
                                continue
                            arg_chunk_dim = all_node_info[arg]["chunk_dim"]
                            arg_fix_dim = all_node_info[arg]["fix_dim"]
                            arg_shape = get_node_shape(arg)
                            # add all dim as fix dim except chunk dim
                            for i, shape in enumerate(arg_shape):
                                if shape != 1 and i != cur_node_chunk_dim:
                                    if i == arg_chunk_dim:
                                        return None
                                    if i not in arg_fix_dim:
                                        arg_fix_dim.append(i)
                    elif "einsum" in cur_node.name:
                        pass
                    elif "matmul" in cur_node.name:
                        pass
                    else:
                        raise NotImplementedError()
            cur_node_list = next_node_list
        return all_node_info

    def _get_input_nodes_dim(self, inputs, start_idx, end_idx, all_node_info):
        inputs_dim = []
        remove_inputs = []
        for input_node in inputs:
            input_dict = {}
            input_node_idx = find_idx_by_name(input_node.name, self.node_list)
            for user in input_node.users.keys():
                if is_non_compute_node(user):
                    continue
                user_idx = find_idx_by_name(user.name, self.node_list)
                if start_idx <= user_idx <= end_idx:
                    chunk_dim = all_node_info[user]["chunk_dim"]
                    if chunk_dim is not None:
                        user_source = self._find_source_trace_from_node(user)[chunk_dim]
                        if input_node_idx in user_source:
                            input_dict[user_idx] = user_source[input_node_idx]
                        else:
                            return None, None
            if len(input_dict) == 0:
                remove_inputs.append(input_node)
            else:
                inputs_dim.append(input_dict)
        for i in remove_inputs:
            if i in inputs:
                inputs.remove(i)
        return inputs, inputs_dim

    def _set_prepose_nodes(self, all_node_info, start_idx, end_idx):
        # get all possible prepose nodes
        maybe_prepose_nodes = []
        for node, node_info in all_node_info.items():
            if node_info["chunk_dim"] is None:
                maybe_prepose_nodes.append(node)
        maybe_prepose_nodes.sort(
            key=lambda x: find_idx_by_name(x.name, self.node_list),
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
                    for cur_prepose_node_arg in cur_prepose_node.args:
                        if type(cur_prepose_node_arg) != type(cur_prepose_node):
                            continue
                        # out of loop
                        if not (
                            start_idx
                            <= find_idx_by_name(
                                cur_prepose_node_arg.name, self.node_list
                            )
                            < end_idx
                        ):
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
        prepose_nodes.sort(key=lambda x: find_idx_by_name(x.name, self.node_list))

        return prepose_nodes
    
    def flow_search(self, start_idx, start_dim, end_idx, end_dim):
        inputs, outputs = find_chunk_compute_input_and_output_nodes(
            self.node_list[start_idx : end_idx + 1]
        )
        # only single ouput
        if len(outputs) > 1:
            return None

        # get every node's chunk dim and fix dim
        all_node_info = self._get_all_node_info(end_dim, start_idx, end_idx)
        if all_node_info is None:
            return None

        # get input nodes' chunk dim
        inputs, inputs_dim = self._get_input_nodes_dim(inputs, start_idx, end_idx, all_node_info)
        if inputs is None:
            return None

        chunk_info = {
            "region": (start_idx, end_idx),
            "inputs": inputs,
            "inputs_non_chunk": [],
            "inputs_dim": inputs_dim,
            "outputs": outputs,
            "outputs_dim": end_dim,
            "node_chunk_dim": all_node_info,
            "args": {},
        }

        # move useless nodes ahead of loop
        chunk_info["args"]["prepose_nodes"] = self._set_prepose_nodes(all_node_info, start_idx, end_idx)

        # we need to log input nodes to avoid deleteing them in the loop
        chunk_node_list = self.node_list[start_idx : end_idx + 1]
        # also need to get some prepose node's arg out of non_chunk_inputs
        for n in chunk_info["args"]["prepose_nodes"]:
            chunk_node_list.remove(n)
        non_chunk_inputs = find_chunk_all_input_nodes(chunk_node_list)
        for i in non_chunk_inputs:
            if i not in chunk_info["inputs"]:
                chunk_info["inputs_non_chunk"].append(i)

        # reassgin reshape size, some size may have changed due to chunk
        chunk_info = self._reassgin_reshape_size(chunk_info)

        return chunk_info

    def _reassgin_reshape_size(self, chunk_info):
        chunk_region = chunk_info["region"]
        reshape_size = {}
        chunk_shape = get_node_shape(chunk_info["outputs"][0])[
            chunk_info["outputs_dim"]
        ]
        for node in self.node_list[chunk_region[0] : chunk_region[1] + 1]:
            if any(i in node.name for i in ["reshape", "view"]):
                reshape_args = node.args[1:]
                reshape_log = self.idx_view_list[node]
                chunk_dim = chunk_info["node_chunk_dim"][node]["chunk_dim"]
                reshape_size[node.name] = {}
                for reshape_arg_dim, reshape_arg in enumerate(reshape_args):
                    if reshape_arg_dim in reshape_log["dim_to"]:
                        continue
                    if reshape_arg_dim == chunk_dim:
                        reshape_size[node.name][reshape_arg.name] = (
                            "min(chunk_size, %d - chunk_idx)" % chunk_shape
                        )
        chunk_info["reshape_size"] = reshape_size
        return chunk_info

    def _get_reorder_map(self, chunk_info):
        reorder_map = {i: i for i in range(len(self.node_list))}

        chunk_region_start = chunk_info["region"][0]
        chunk_region_end = chunk_info["region"][1]
        chunk_prepose_nodes = chunk_info["args"]["prepose_nodes"]
        chunk_prepose_nodes_idx = [
            find_idx_by_name(i.name, self.node_list) for i in chunk_prepose_nodes
        ]
        # put prepose nodes ahead
        for idx, n in enumerate(chunk_prepose_nodes):
            n_idx = chunk_prepose_nodes_idx[idx]
            reorder_map[n_idx] = chunk_region_start + idx
        # put other nodes after prepose nodes
        for n in self.node_list[chunk_region_start : chunk_region_end + 1]:
            if n in chunk_prepose_nodes:
                continue
            n_idx = find_idx_by_name(n.name, self.node_list)
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
        for idx, input_dim in enumerate(chunk_info["inputs_dim"]):
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
        new_node_list = [None for _ in range(len(self.node_list))]
        for old_idx, new_idx in reorder_map.items():
            new_node_list[new_idx] = self.node_list[old_idx]
        self.node_list = new_node_list

    def _reorder_idx_trace(self, reorder_map):
        # reorder list
        new_idx_trace_list = [None for _ in range(len(self.idx_trace_list))]
        for old_idx, new_idx in reorder_map.items():
            new_idx_trace_list[new_idx] = self.idx_trace_list[old_idx]
        self.idx_trace_list = new_idx_trace_list
        # update compute
        for idx_trace in self.idx_trace_list:
            compute = idx_trace["compute"]
            for dim_compute in compute:
                for idx, i in enumerate(dim_compute):
                    dim_compute[idx] = reorder_map[i]
        # update source
        for idx_trace in self.idx_trace_list:
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
