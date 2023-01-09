import copy

from .utils import (
    find_idx_by_name,
    get_node_shape,
)


class TraceIndice(object):
    def __init__(self, node_list) -> None:
        self.node_list = node_list
        self.indice_trace_list = self._init_indice_trace_list()
        self.indice_trace_equal = []
        self.indice_view_list = {}
        self.indice_count = -1

    def _init_indice_trace_list(self):
        indice_trace_list = []
        for n in self.node_list:
            if get_node_shape(n) != None:
                cur_trace = {
                    "indice": [None for _ in range(len(get_node_shape(n)))],
                    "compute": [[] for _ in range(len(get_node_shape(n)))],
                    "source": [{} for _ in range(len(get_node_shape(n)))],
                }
            else:
                cur_trace = {"indice": [], "compute": [], "source": []}
            indice_trace_list.append(cur_trace)
        return indice_trace_list

    def _add_indice(self):
        """
        Update the count and return it. To record the idx number.

        Returns:
            idx_count: int
        """
        self.indice_count += 1
        return self.indice_count

    def _del_dim(self, idx, dim_idx):
        self.indice_trace_list[idx]["indice"].pop(dim_idx)
        self.indice_trace_list[idx]["compute"].pop(dim_idx)
        self.indice_trace_list[idx]["source"].pop(dim_idx)

    def _add_dim(self, node_idx, dim_idx):
        self.indice_trace_list[node_idx]["indice"].insert(dim_idx, self._add_indice())
        self.indice_trace_list[node_idx]["compute"].insert(dim_idx, [])
        self.indice_trace_list[node_idx]["source"].insert(dim_idx, {})

    def _transform_indice(self, node, node_dim):
        node_idx = self._find_indice_trace_from_node(node)
        dims = list(range(len(node_idx)))
        return dims[node_dim]

    def _inherit_indice(self, node_from, node_from_dim, node_to, node_to_dim):
        node_from_dim = self._transform_indice(node_from, node_from_dim)
        node_to_dim = self._transform_indice(node_to, node_to_dim)
        node_from_trace = self._find_trace_from_node(node_from)
        node_to_trace = self._find_trace_from_node(node_to)
        node_to_trace["indice"][node_to_dim] = node_from_trace["indice"][node_from_dim]
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
        node_from_dim = self._transform_indice(node_from, node_from_dim)
        node_from_trace_source = self._find_source_trace_from_node(node_from)
        node_to_dim = self._transform_indice(node_to, node_to_dim)
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
            exclude = [self._transform_indice(node_to, i) for i in exclude]
        node_from_compute = self._find_compute_trace_from_node(node_from)
        node_to_compute = self._find_compute_trace_from_node(node_to)
        # assert len(node_from_compute) == len(node_to_compute)
        for i in range(-1, -min(len(node_from_compute), len(node_to_compute)) - 1, -1):
            if self._transform_indice(node_to, i) in exclude:
                continue
            self._add_source(node_from, i, node_to, i)
            for j in node_from_compute[i]:
                if j not in node_to_compute[i]:
                    node_to_compute[i].append(j)

    def _mark_indice_equal(self, node1, dim1, node2, dim2):
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
            if idx not in self.indice_trace_list[idx]["compute"][cur_dim]:
                self.indice_trace_list[idx]["compute"][cur_dim].append(idx)

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
        node_dict = self.indice_trace_list[node_idx]
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
        node_dict = self.indice_trace_list[node_idx]
        return node_dict["source"]

    def _find_indice_trace_from_node(self, node):
        """
        Find node idx trace by the node.

        Args:
            node (node)
        Returns:
            idx (list): idx of the node
        """
        node_idx = find_idx_by_name(node.name, self.node_list)
        return self.indice_trace_list[node_idx]["indice"]

    def _find_compute_trace_from_node(self, node):
        """
        Find node compute trace by the node.

        Args:
            node (node)
        Returns:
            compute (list): computed idx of the node.
        """
        node_idx = find_idx_by_name(node.name, self.node_list)
        return self.indice_trace_list[node_idx]["compute"]

    def _assign_indice_as_input(self, node, node_idx, input_node=None):
        """
        Assign node's trace as its input node.

        Args:
            node (node)
            node_idx (int)
        """
        if input_node == None:
            input_node = node.args[0]
        input_node_idx = find_idx_by_name(input_node.name, self.node_list)
        input_node_idx_trace = self.indice_trace_list[input_node_idx]["indice"]

        new_idx_trace = copy.deepcopy(input_node_idx_trace)
        self.indice_trace_list[node_idx]["indice"] = new_idx_trace

        self._inherit_all_computation(input_node, node)

    def _assign_all_indice(self, node, node_idx):
        """
        Add new index for all node's dims.

        Args:
            node (node)
            node_idx (int)
        """
        shape = node.meta["tensor_meta"].shape
        new_trace = []
        for _ in shape:
            new_trace.append(self._add_indice())
        self.indice_trace_list[node_idx]["indice"] = new_trace

    def _assign_transpose_indice(self, node, node_idx):
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

        self._assign_indice_as_input(node, node_idx, input_node)
        self._inherit_indice(input_node, tranpose_dim[1], node, tranpose_dim[0])
        self._inherit_indice(input_node, tranpose_dim[0], node, tranpose_dim[1])

    def _assign_permute_indice(self, node, node_idx):
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

        self._assign_indice_as_input(node, node_idx, input_node)
        for idx, d in enumerate(permute_dim):
            self._inherit_indice(input_node, d, node, idx)

    def _assign_linear_indice(self, node, node_idx):
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

        self._assign_indice_as_input(node, node_idx)
        self._inherit_indice(weight, 1, node, -1)

        self._mark_computation(node, node_idx, [-1])
        self._mark_indice_equal(input_node, -1, weight, 0)

        if bias:
            self._mark_indice_equal(input_node, -1, bias, 0)

    def _assign_matmul_indice(self, node, node_idx):
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
        self._assign_indice_as_input(node, node_idx, matmul_left)
        self._inherit_indice(matmul_right, -1, node, -1)

        self._mark_computation_from_node(matmul_right, node, [-1, -2])
        self._mark_computation(node, node_idx, [-1])
        self._mark_indice_equal(matmul_left, -1, matmul_right, -2)

    def _assign_layernorm_indice(self, node, idx):
        """
        Assign index for layernorm op.
        1. assign index as input node
        2. inherit computation and mark last 2 dims as computed.

        Args:
            node (node)
            node_idx (int)
        """
        self._assign_indice_as_input(node, idx)
        self._mark_computation(node, idx, [-1])

    def _assign_elementwise_indice(self, node, idx):
        """
        Assign index for element-wise op (eg. relu sigmoid add mul).
        1. assign index as input node
        2. inherit computation from all input nodes.

        Args:
            node (node)
            node_idx (int)
        """
        self._assign_indice_as_input(node, idx)
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
                    self._mark_indice_equal(nodes_in[0], i, nodes_in[1], i)

    def _assgin_no_change_indice(self, node, idx):
        self._assign_indice_as_input(node, idx)
        for node_in in node.args:
            if type(node_in) == type(node):
                self._mark_computation_from_node(node_in, node)

    def _assign_einsum_indice(self, node, idx):
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
                    self._inherit_indice(
                        input_nodes[left_idx], source_idx, node, right_idx
                    )

        # for i in sum_index:
        #     for left_idx, left_str in enumerate(left):
        #         if i in left_str:
        #             self._mark_computation(node, idx, left_str.index(i))
        #             break

    def _assign_softmax_indice(self, node, idx):
        """
        Assign index for softmax op.
        1. assign index as input node
        2. inherit computation and mark softmax dim as computed.

        Args:
            node (node)
            node_idx (int)
        """
        self._assign_indice_as_input(node, idx)
        self._mark_computation(node, idx, [node.kwargs["dim"]])

    def _assign_unsqueeze_indice(self, node, node_idx):
        """
        Assign index for unsqueeze op.
        1. assign new index for unsqueeze dim

        Args:
            node (node)
            node_idx (int)
        """
        self._del_dim(node_idx, -1)
        self._assign_indice_as_input(node, node_idx)
        self._add_dim(node_idx, node.args[1])

    def _assign_dropout_indice(self, node, node_idx):
        """
        Assign index for unsqueeze op.
        1. assign new index for unsqueeze dim

        Args:
            node (node)
            node_idx (int)
        """
        self._assign_indice_as_input(node, node_idx)

    def _assign_ones_like_indice(self, node, node_idx):
        """
        Assign index for oneslike op.
        1. assign new index for all dim

        Args:
            node (node)
            node_idx (int)
        """
        self._assign_all_indice(node, node_idx)

    def _assign_view_reshape_indice(self, node, node_idx):
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
        origin_trace = self._find_indice_trace_from_node(origin_node)
        self._assign_indice_as_input(node, node_idx, origin_node)
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
            "idx_to": [self.indice_trace_list[node_idx]["indice"][i] for i in dim_to],
            "dim_to": dim_to,
        }
        self.indice_view_list[node] = view_dict

    def _merge_equal_idx(self):
        idx_equal = copy.deepcopy(self.indice_trace_equal)
        idx_equal.reverse()
        for idx in idx_equal:
            merge_to = min(idx)
            merge_from = max(idx)
            for trace in self.indice_trace_list:
                if merge_from in trace["indice"]:
                    trace["indice"] = [
                        merge_to if i == merge_from else i for i in trace["indice"]
                    ]

    def trace_index(self):
        for idx, node in enumerate(self.node_list):
            if node.op == "placeholder":
                self._assign_all_indice(node, idx)
            elif node.op == "call_method":
                if "transpose" in node.name:
                    self._assign_transpose_indice(node, idx)
                elif "permute" in node.name:
                    self._assign_permute_indice(node, idx)
                elif "view" in node.name or "reshape" in node.name:
                    self._assign_view_reshape_indice(node, idx)
                elif "unsqueeze" in node.name:
                    self._assign_unsqueeze_indice(node, idx)
                elif any(i in node.name for i in ["to", "contiguous"]):
                    self._assgin_no_change_indice(node, idx)
                else:
                    raise NotImplementedError(node.name, "method not implemented yet!")
            elif node.op == "call_function":
                if "linear" in node.name:
                    self._assign_linear_indice(node, idx)
                elif "matmul" in node.name:
                    self._assign_matmul_indice(node, idx)
                elif "softmax" in node.name:
                    self._assign_softmax_indice(node, idx)
                elif any(n in node.name for n in ["mul", "add", "sigmoid", "relu"]):
                    self._assign_elementwise_indice(node, idx)
                elif "ones_like" in node.name:
                    self._assign_ones_like_indice(node, idx)
                elif "dropout" in node.name:
                    self._assign_dropout_indice(node, idx)
                elif "einsum" in node.name:
                    self._assign_einsum_indice(node, idx)
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
                    self._assign_layernorm_indice(node, idx)
                else:
                    raise NotImplementedError(node.name, "module not implemented yet!")
            elif node.op == "get_attr":
                self._assign_all_indice(node, idx)  # get param
            elif node.op == "output":
                continue
            else:
                raise NotImplementedError(node.op, "op not implemented yet!")
