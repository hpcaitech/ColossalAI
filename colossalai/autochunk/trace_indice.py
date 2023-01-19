import copy
from typing import Dict, List, Tuple

from torch.fx.node import Node

from .utils import find_first_tensor_arg, find_idx_by_name, flat_list, get_node_shape


class TraceIndice(object):
    """
    Trace all indice infomation for every node.

    Indice is a logical concept. Equal dims can been treated as one indice.
    eg. dim(x1) = [a, b, c]
        dim(x2) = [d, e, f]
        and we have x3 = x1 * x2.
        then a=d, b=e, c=f, due to the broadcast property,
        dim(x1)=dim(x2)=dim(x3)=[a, b, c]
    This class will record every node's dims' indice, compute and source.

    Attibutes:
        node_list (List)
        indice_trace_list (List): [{"indice": [...], "compute": [...], "source": [...]}, {...}]
        indice_view_list (Dict): not used for now
        indice_count (int): record indice number

    Args:
        node_list (List)
    """

    def __init__(self, node_list: List[Node]) -> None:
        self.node_list = node_list
        self.indice_trace_list = self._init_indice_trace_list()
        self.indice_view_list = {}
        self.indice_count = -1
        self.trace_range = []
        self.active_node_list = []

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

    def set_trace_range(self, trace_range: List, active_node_list: List) -> None:
        self.trace_range = trace_range
        self.active_node_list = active_node_list

    def _add_indice(self):
        """
        Update the count and return it. To record the idx number.

        Returns:
            indice_count: int
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
        node_to_trace["compute"][node_to_dim] = copy.deepcopy(node_from_trace["compute"][node_from_dim])
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

    def _assign_indice_as_input(self, node: Node, node_idx: int, input_node=None):
        """
        Assign node's trace as its input node.

        Args:
            node (node)
            node_idx (int)
        """
        if input_node == None:
            input_node = find_first_tensor_arg(node)
        input_node_idx = find_idx_by_name(input_node.name, self.node_list)
        input_node_idx_trace = self.indice_trace_list[input_node_idx]["indice"]

        new_idx_trace = copy.deepcopy(input_node_idx_trace)
        self.indice_trace_list[node_idx]["indice"] = new_idx_trace

        self._inherit_all_computation(input_node, node)

    def _assign_all_indice(self, node: Node, node_idx: int):
        """
        Add new indice for all node's dims.

        Args:
            node (node)
            node_idx (int)
        """
        shape = node.meta["tensor_meta"].shape
        if shape is None:
            return
        new_trace = []
        for _ in shape:
            new_trace.append(self._add_indice())
        self.indice_trace_list[node_idx]["indice"] = new_trace

    def _assign_transpose_indice(self, node: Node, node_idx: int):
        """
        Assign indice for transpose op.
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

    def _assign_permute_indice(self, node: Node, node_idx: int):
        """
        Assign indice for permute op.
        1. swap input's dim according to permute args
        2. inherit input's computation

        Args:
            node (node)
            node_idx (int)
        """
        permute_dim = flat_list(node.args[1:])
        input_node = node.args[0]

        self._assign_indice_as_input(node, node_idx, input_node)
        for idx, d in enumerate(permute_dim):
            self._inherit_indice(input_node, d, node, idx)

    def _assign_linear_indice(self, node: Node, node_idx: int):
        """
        Assign indice for linear op.
        1. copy trace from input node and change last indice accroding to weight
        2. mark equal for input node last indice, weight first dim and bias dim.
        3. inherit input's computation, mark computation for last dim.

        Args:
            node (node)
            node_idx (int)
        """
        if len(node.args) == 2:
            _, weight = node.args
        else:
            _, weight, _ = node.args

        self._assign_indice_as_input(node, node_idx)
        self._inherit_indice(weight, 1, node, -1)

        self._mark_computation(node, node_idx, [-1])

    def _assign_matmul_indice(self, node: Node, node_idx: int):
        """
        Assign indice for matmul op.
        1. copy trace from matmul_left and change last indice accroding to matmul_right. (assert they have same length)
        2. mark equal for input matmul_left -1 indice and matmul_right -2 dim.
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

    def _assign_layernorm_indice(self, node, idx):
        """
        Assign indice for layernorm op.
        1. assign indice as input node
        2. inherit computation and mark last 2 dims as computed.

        Args:
            node (node)
            node_idx (int)
        """
        self._assign_indice_as_input(node, idx)
        self._mark_computation(node, idx, [-1])

    def _assign_elementwise_indice(self, node, idx):
        """
        Assign indice for element-wise op (eg. relu sigmoid add mul).
        1. assign indice as input node
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

    def _assgin_no_change_indice(self, node, idx):
        self._assign_indice_as_input(node, idx)
        for node_in in node.args:
            if type(node_in) == type(node):
                self._mark_computation_from_node(node_in, node)

    def _assign_einsum_indice(self, node, idx):
        """
        Assign indice for einsum op.

        Args:
            node (node)
            node_idx (int)
        """
        patterns = node.args[0]
        input_nodes = node.args[1:]

        patterns = patterns.replace(" ", "")
        left, right = patterns.split("->")
        left = left.split(",")

        if '...' in right:
            replace_list = "!@#$%^&*"
            target_len = len(get_node_shape(node))
            add_len = target_len - len(right) + 3
            replace_str = replace_list[:add_len]
            right = right.replace("...", replace_str)
            for ll in range(len(left)):
                left[ll] = left[ll].replace("...", replace_str)

        all_index = []
        for i in left:
            for c in i:
                all_index.append(c)
        all_index = set(all_index)

        for right_idx, right_indice in enumerate(right):
            for left_idx, left_str in enumerate(left):
                if right_indice in left_str:
                    source_idx = left_str.index(right_indice)
                    self._inherit_indice(input_nodes[left_idx], source_idx, node, right_idx)

    def _assign_softmax_indice(self, node, idx):
        """
        Assign indice for softmax op.
        1. assign indice as input node
        2. inherit computation and mark softmax dim as computed.

        Args:
            node (node)
            node_idx (int)
        """
        self._assign_indice_as_input(node, idx)
        self._mark_computation(node, idx, [node.kwargs["dim"]])

    def _assign_unsqueeze_indice(self, node: Node, node_idx: int):
        """
        Assign indice for unsqueeze op.
        1. assign new indice for unsqueeze dim

        Args:
            node (node)
            node_idx (int)
        """
        self._del_dim(node_idx, -1)
        self._assign_indice_as_input(node, node_idx)
        dim_idx = node.args[1]
        # unsqueeze(-1) = unsqueeze(shape_num + 1)
        if dim_idx < 0:
            dim_idx = list(range(len(get_node_shape(node))))[dim_idx]
        self._add_dim(node_idx, dim_idx)

    def _assign_dropout_indice(self, node: Node, node_idx: int):
        """
        Assign indice for unsqueeze op.
        1. assign new indice for unsqueeze dim

        Args:
            node (node)
            node_idx (int)
        """
        self._assign_indice_as_input(node, node_idx)

    def _assign_ones_like_indice(self, node: Node, node_idx: int):
        """
        Assign indice for oneslike op.
        1. assign new indice for all dim

        Args:
            node (node)
            node_idx (int)
        """
        self._assign_all_indice(node, node_idx)

    def _assign_cat_indice(self, node: Node, node_idx: int):
        """
        Assign indice for cat op.

        Args:
            node (node)
            node_idx (int)
        """
        nodes_in = flat_list(node.args[0])
        self._assign_indice_as_input(node, node_idx, input_node=nodes_in[0])
        for n in nodes_in[1:]:
            self._mark_computation_from_node(n, node)
        cat_dim = node.kwargs["dim"]
        self._del_dim(node_idx, cat_dim)
        self._add_dim(node_idx, cat_dim)

    def _assign_sum_indice(self, node: Node, node_idx: int):
        """
        Assign indice for sum op.

        Args:
            node (node)
            node_idx (int)
        """
        nodes_in = flat_list(node.args[0])
        self._add_dim(node_idx, 0)
        self._assign_indice_as_input(node, node_idx, input_node=nodes_in[0])
        for n in nodes_in[1:]:
            self._mark_computation_from_node(n, node)
        cat_dim = node.kwargs["dim"]
        self._del_dim(node_idx, cat_dim)

    def _assign_getitem_indice(self, node: Node, node_idx: int):
        """
        Assign indice for getitem.
        getitem can act like slice sometimes

        Args:
            node (node)
            node_idx (int)
        """
        node_args = flat_list(node.args[1:])
        flag = False
        for node_arg in node_args:
            node_arg_str = str(node_arg)
            if any(i == node_arg_str for i in ["None", "Ellipsis"]):
                flag = True
                break
            if "slice" in node_arg_str:
                flag = True
                break
        if flag == False:
            return

        # node args should be like [Ellipsis, slice(start, step, end), None]
        node_shape = get_node_shape(node)
        origin_idx_count = 0
        new_idx_count = 0
        new_dim_num = sum([1 if str(i) == "None" else 0 for i in node_args])
        for _ in range(new_dim_num):
            self._del_dim(node_idx, 0)
        delete_dim_num = sum([1 if str(i) == "0" else 0 for i in node_args])
        for _ in range(delete_dim_num):
            self._add_dim(node_idx, 0)
        self._assign_indice_as_input(node, node_idx)

        for _, node_arg in enumerate(node_args):
            node_arg_str = str(node_arg)
            # Ellipsis means [..., ]
            if "Ellipsis" == node_arg_str:
                shape_gap = len(node_shape) - len(node_args) + 1
                origin_idx_count += shape_gap
                new_idx_count += shape_gap
            # slice(None, None, None) means all indexes
            elif "slice" in node_arg_str:
                if "slice(None, None, None)" != node_arg_str:
                    self._del_dim(node_idx, new_idx_count)
                    self._add_dim(node_idx, new_idx_count)
                origin_idx_count += 1
                new_idx_count += 1
            # None means a new dim
            elif "None" == node_arg_str:
                self._add_dim(node_idx, new_idx_count)
                new_idx_count += 1
            elif "0" == node_arg_str:
                self._del_dim(node_idx, new_idx_count)
                origin_idx_count += 1
            else:
                raise NotImplementedError()

    def _assign_view_reshape_indice(self, node: Node, node_idx: int):
        """
        Assign indice for view and reshape op.
        1. get origin shape and target shape by meta info.
        2. compute the real value of -1 in target shape.
        3. determine changed dim, and assgin indice for generated dim.
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
        unflated_args = flat_list(node.args)
        for i in range(1, len(unflated_args)):
            if isinstance(unflated_args[i], int):
                target_shape.append(unflated_args[i])
            else:
                target_shape.append(unflated_args[i].meta["fwd_out"][0])

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
            raise NotImplementedError("shape" + str(origin_shape) + "and" + str(target_shape) + "view not implemented")

        # get new indice
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

    def _clear_trace(self, node_idx: int) -> None:
        """
        clear too far trace to speed up computation
        """
        trace_range = None
        for i in range(len(self.trace_range)):
            if self.trace_range[i][1] == node_idx:
                trace_range = (self.trace_range[i][0], self.trace_range[i][1])
                break
            if self.trace_range[i][1] > node_idx:
                break
        if trace_range is None:
            return

        active_nodes = self.active_node_list[trace_range[0]:trace_range[1] + 1]
        active_nodes = set(flat_list(active_nodes))
        active_nodes = [find_idx_by_name(i, self.node_list) for i in active_nodes]
        for i in range(trace_range[0], trace_range[1] + 1):
            trace = self.indice_trace_list[i]
            # clear compute
            for dim_compute in trace["compute"]:
                for i in range(len(dim_compute) - 1, -1, -1):
                    if dim_compute[i] < trace_range[0] and dim_compute[i] not in active_nodes:
                        dim_compute.pop(i)
                continue
            # clear source
            for dim_source in trace["source"]:
                for k in list(dim_source.keys()):
                    if k < trace_range[0] and k not in active_nodes:
                        dim_source.pop(k)

    def trace_indice(self):
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
                elif any(i in node.name for i in ["to", "contiguous", "clone"]):
                    self._assgin_no_change_indice(node, idx)
                elif "new_ones" in node.name:
                    self._assign_ones_like_indice(node, idx)
                else:
                    raise NotImplementedError(node.name, "method not implemented yet!")
            elif node.op == "call_function":
                if "linear" in node.name:
                    self._assign_linear_indice(node, idx)
                elif "cat" in node.name:
                    self._assign_cat_indice(node, idx)
                elif "matmul" in node.name:
                    self._assign_matmul_indice(node, idx)
                elif "softmax" in node.name:
                    self._assign_softmax_indice(node, idx)
                elif any(n in node.name for n in ["mul", "add", "sigmoid", "relu", "sub", "truediv"]):
                    self._assign_elementwise_indice(node, idx)
                elif "ones_like" in node.name:
                    self._assign_ones_like_indice(node, idx)
                elif "dropout" in node.name:
                    self._assign_dropout_indice(node, idx)
                elif "einsum" in node.name:
                    self._assign_einsum_indice(node, idx)
                elif "sum" in node.name:
                    self._assign_sum_indice(node, idx)
                elif "layer_norm" in node.name:
                    self._assign_layernorm_indice(node, idx)
                elif "getitem" in node.name:
                    self._assign_getitem_indice(node, idx)
                elif any(i in node.name for i in ["getattr", "getitem", "eq", "_assert"]):
                    continue
                else:
                    raise NotImplementedError(node.name, "function not implemented yet!")
            elif node.op == "call_module":
                if any(n in node.name for n in ["layernorm", "norm"]):
                    self._assign_layernorm_indice(node, idx)
                elif any(n in node.name for n in ["sigmoid", "dropout", "relu"]):
                    self._assign_elementwise_indice(node, idx)
                else:
                    raise NotImplementedError(node.name, "module not implemented yet!")
            elif node.op == "get_attr":
                self._assign_all_indice(node, idx)    # get param
            elif node.op == "output":
                continue
            else:
                raise NotImplementedError(node.op, "op not implemented yet!")

            # limit trace range
            self._clear_trace(idx)
