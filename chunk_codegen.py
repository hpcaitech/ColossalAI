import colossalai
import torch
import copy
from typing import List, Callable, Any, Tuple, Dict, Iterable

from torch.fx.node import Node, Argument, map_arg, _type_repr, _get_qualified_name
from torch.fx.graph import _Namespace, PythonCode, _custom_builtins, _is_from_torch, _format_target, magic_methods, CodeGen, _origin_type_map, inplace_methods, _CustomBuiltin
from colossalai.fx.profiler import calculate_fwd_out, calculate_fwd_tmp, parameter_size, activation_size
CODEGEN_AVAILABLE = True
__all__ = ['ChunkCodeGen']


def _delete_free_var_from_last_use(user_to_last_uses):
    for key, value in user_to_last_uses.items():
        for n in value:
            if n.op == 'placeholder':
                user_to_last_uses[key].remove(n)


class NodeIndexTracer(object):
    def __init__(self, gm) -> None:
        self.gm = gm
        self.nodes_list = list(gm.graph.nodes)
        self.idx_trace_list = [{'idx': [], 'compute': {}} for _ in range(len(self.nodes_list))] 
        self.idx_trace_equal = []
        self.idx_view_list = []
        self.idx_count = -1

    def _add_index(self):
        """
        Update the count and return it. To record the idx number.
        
        Returns:
            idx_count: int
        """        
        self.idx_count += 1
        return self.idx_count

    def _inherit_computation(self, node_from, node_to):
        """
        Inherit computed dim from node_from to node_to.
        If a dim in node_from is marked as computed and exists in node_to,
        still mark it as computed in node_to.

        Args:
            node_from (node): node to be inherited
            node_to (node): new node to inherit
        """        
        _, compute_from = self._find_trace_from_node(node_from)
        idx_to, compute_to = self._find_trace_from_node(node_to)
        for k, v in compute_from.items():
            if k in idx_to:
                if k in compute_to:
                    compute_to[k].extend(v)
                else:
                    compute_to[k] = copy.deepcopy(v)
    
    def _mark_idx_equal(self, idx1, idx2):
        """
        Mark 2 index to be equal.

        Args:
            idx1 (int): index count.
            idx2 (int): index count.
        """        
        self.idx_trace_equal.append((idx1, idx2))
        
    def _mark_computation(self, node, idx, dim):
        """
        Mark some dims of node as computed.

        Args:
            node (node)
            idx (int): node index
            dim (list or int): dims to be marked as computed
        """        
        input_node_idx_trace = self._find_idx_trace_from_node(node)
        if isinstance(dim, int):
            dim = [dim]
        for d in dim:
            cur_idx = input_node_idx_trace[d]
            if cur_idx not in self.idx_trace_list[idx]['compute']:
                self.idx_trace_list[idx]['compute'][cur_idx] = [idx]
            else:
                self.idx_trace_list[idx]['compute'][cur_idx].append(idx)
    
    def _find_trace_from_node(self, node):
        """
        Find node idx and compute trace by the node.

        Args:
            node (node)
        Returns:
            idx (list): idx of the node
            compute (list): computed idx of the node.
        """        
        node_idx = _find_idx_by_name(node.name, self.nodes_list)
        node_dict = self.idx_trace_list[node_idx]
        return node_dict['idx'], node_dict['compute']
    
    def _find_idx_trace_from_node(self, node):
        """
        Find node idx trace by the node.

        Args:
            node (node)
        Returns:
            idx (list): idx of the node
        """ 
        node_idx = _find_idx_by_name(node.name, self.nodes_list)
        return self.idx_trace_list[node_idx]['idx']
    
    def _find_compute_trace_from_node(self, node):
        """
        Find node compute trace by the node.

        Args:
            node (node)
        Returns:
            compute (list): computed idx of the node.
        """ 
        node_idx = _find_idx_by_name(node.name, self.nodes_list)
        return self.idx_trace_list[node_idx]['compute']
    
    def _assign_index_as_input(self, node, node_idx):
        """
        Assign node's trace as its input node.

        Args:
            node (node)
            node_idx (int)
        """        
        input_node_idx = _find_idx_by_name(node.args[0].name, self.nodes_list)
        input_node_idx_trace = self.idx_trace_list[input_node_idx]['idx']
        
        new_idx_trace = copy.deepcopy(input_node_idx_trace)
        self.idx_trace_list[node_idx]['idx'] = new_idx_trace
    
    def _assign_all_index(self, node, node_idx):
        """
        Add new index for all node's dims.

        Args:
            node (node)
            node_idx (int)
        """  
        shape = node.meta['tensor_meta'].shape
        new_trace = []
        for _ in shape:
            new_trace.append(self._add_index())
        self.idx_trace_list[node_idx]['idx'] = new_trace   

    def _assign_transpose_index(self, node, node_idx):
        """
        Assign index for transpose op.
        1. swap input's dim according to transpose args
        2. inherit input's computation

        Args:
            node (node)
            node_idx (int)
        """  
        tranpose_dim = node.args[1:]
        input_node_idx_trace = self._find_idx_trace_from_node(node.args[0])
        
        new_idx_trace = copy.deepcopy(input_node_idx_trace)
        new_idx_trace[tranpose_dim[0]] = input_node_idx_trace[tranpose_dim[1]]
        new_idx_trace[tranpose_dim[1]] = input_node_idx_trace[tranpose_dim[0]]

        self.idx_trace_list[node_idx]['idx'] = new_idx_trace
        self._inherit_computation(node.args[0], node)
        
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
        input_node_idx_trace = self._find_idx_trace_from_node(node.args[0])
        
        new_idx_trace = copy.deepcopy(input_node_idx_trace)
        for idx, d in enumerate(permute_dim):
            new_idx_trace[idx] = input_node_idx_trace[d]

        self.idx_trace_list[node_idx]['idx'] = new_idx_trace
        self._inherit_computation(node.args[0], node)
        
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
        input_node_idx_trace = self._find_idx_trace_from_node(input_node)
        weight_idx_trace = self._find_idx_trace_from_node(weight)
        
        new_idx_trace = copy.deepcopy(input_node_idx_trace)
        new_idx_trace[-1] = weight_idx_trace[1]
        self.idx_trace_list[node_idx]['idx'] = new_idx_trace

        self._inherit_computation(input_node, node)
        self._mark_computation(node, node_idx, [-1])
        self._mark_idx_equal(input_node_idx_trace[-1], weight_idx_trace[0])
        
        if bias:
            bias_idx_trace = self._find_idx_trace_from_node(bias)
            self._mark_idx_equal(input_node_idx_trace[-1], bias_idx_trace[0])

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
        matmul_left_idx_trace = self._find_idx_trace_from_node(matmul_left)
        matmul_right_idx_trace = self._find_idx_trace_from_node(matmul_right)
        
        assert(len(matmul_left_idx_trace) == len(matmul_right_idx_trace))
        new_idx_trace = copy.deepcopy(matmul_left_idx_trace)
        new_idx_trace[-1] = matmul_right_idx_trace[-1]
        self.idx_trace_list[node_idx]['idx'] = new_idx_trace

        self._inherit_computation(matmul_left, node)
        self._inherit_computation(matmul_right, node)
        self._mark_computation(node, node_idx, [-1])
        self._mark_idx_equal(matmul_left_idx_trace[-1], matmul_right_idx_trace[-2])

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
        self._inherit_computation(node.args[0], node)
        self._mark_computation(node, idx, [-1, -2])
    
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
        for node_in in node.args:
            if type(node_in) not in (int, float):
                self._inherit_computation(node_in, node)
                
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
        self._inherit_computation(node.args[0], node)
        self._mark_computation(node, idx, [node.kwargs['dim']])
        
    def _assign_unsqueeze_index(self, node, node_idx):
        """
        Assign index for unsqueeze op.
        1. assign new index for unsqueeze dim

        Args:
            node (node)
            node_idx (int)
        """ 
        self._assign_index_as_input(node, node_idx)
        self._inherit_computation(node.args[0], node)
        self.idx_trace_list[node_idx]['idx'].insert(node.args[1], self._add_index())
        
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
        
    def _assign_to_index(self, node, node_idx):
        """
        Assign index for to op.
        1. assign new index for all dim

        Args:
            node (node)
            node_idx (int)
        """ 
        self._assign_index_as_input(node, node_idx)

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
        origin_shape = origin_node.meta['tensor_meta'].shape
        target_shape = []
        for i in range(1, len(node.args)):
            if isinstance(node.args[i], int):
                target_shape.append(node.args[i])
            else:
                target_shape.append(node.args[i].meta['fwd_out'][0])

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
        elif len_diff == -1:
            # dim expand
            dim_equal = [i == j for i, j in zip(origin_shape, target_shape[:-1])]
            dim_from = [dim_equal.index(False)]
            dim_to = [dim_equal.index(False), dim_equal.index(False) + 1]
        else:
            raise NotImplementedError("shape" + str(origin_shape) + 'and' + str(target_shape) + "view not implemented")

        # get new index
        origin_trace = self._find_idx_trace_from_node(origin_node)
        new_trace = copy.deepcopy(origin_trace)
        dim_from.reverse()
        for i in dim_from:
            new_trace.pop(i)
        for i in dim_to:
            new_trace.insert(i, self._add_index())
        self.idx_trace_list[node_idx]['idx'] = new_trace
        
        # inherit computation
        self._inherit_computation(origin_node, node)
        compute_log = self._find_compute_trace_from_node(origin_node)
        for i in dim_from:
            if origin_trace[i] in compute_log:
                for j in dim_to:
                    self._mark_computation(node, node_idx, [j])
                break
        
        # log view, not used now
        view_dict = {"idx_from": [origin_trace[i] for i in dim_from],
                     "dim_from": dim_from,
                     "idx_to": [new_trace[i] for i in dim_to],
                     "dim_to": dim_to}
        self.idx_view_list.append(view_dict) 
    
    def _remove_duplicate_compute(self):
        for i in self.idx_trace_list:
            for k, v in i['compute'].items():
                i['compute'][k] = list(set(v))
    
    def _merge_equal_idx(self):
        idx_equal = copy.deepcopy(self.idx_trace_equal)
        idx_equal.reverse()
        for idx in idx_equal:
            merge_to = min(idx)
            merge_from = max(idx)
            for trace in self.idx_trace_list:
                if merge_from in trace['idx']:
                    trace['idx'] = [merge_to if i == merge_from else i for i in trace['idx']]
    
    def trace_node_idx(self):
        for idx, node in enumerate(self.nodes_list):
            if node.op == 'placeholder':
                self._assign_all_index(node, idx)
            elif node.op == 'call_method':
                if 'transpose' in node.name:
                    self._assign_transpose_index(node, idx)
                elif 'permute' in node.name:
                    self._assign_permute_index(node, idx)
                elif 'view' in node.name or 'reshape' in node.name:
                    self._assign_view_reshape_index(node, idx)
                elif 'unsqueeze' in node.name:
                    self._assign_unsqueeze_index(node, idx)
                elif 'to' in node.name:
                    self._assign_to_index(node, idx)
                else:
                    raise NotImplementedError(node.name, "method not implemented yet!")
            elif node.op == 'call_function':
                if 'linear' in node.name:
                    self._assign_linear_index(node, idx)
                elif 'matmul' in node.name:
                    self._assign_matmul_index(node, idx)
                elif 'softmax' in node.name:
                    self._assign_softmax_index(node, idx)
                elif any(n in node.name for n in ['mul', 'add', 'sigmoid', 'relu']):
                    self._assign_elementwise_index(node, idx)
                elif 'ones_like' in node.name:
                    self._assign_ones_like_index(node, idx)
                elif 'dropout' in node.name:
                    self._assign_dropout_index(node, idx)
                elif 'getattr' in node.name:
                    continue # get attr like shape
                elif 'getitem' in node.name:
                    continue # get item in list
                else:
                    raise NotImplementedError(node.name, "function not implemented yet!")
            elif node.op == 'call_module':
                if any(n in node.name for n in ['layernorm', 'norm']):
                    self._assign_layernorm_index(node, idx)
                else:
                    raise NotImplementedError(node.name, "module not implemented yet!")
            elif node.op == 'get_attr':
                self._assign_all_index(node, idx) # get param
            elif node.op == 'output':
                continue
            else:
                raise NotImplementedError(node.op, "op not implemented yet!")
            
        self._remove_duplicate_compute()
        self._merge_equal_idx()


class MemoryEstimator(object):
    def __init__(self) -> None:
        pass

    def _get_meta_node_size(self, x):
        x = x.meta['tensor_meta']
        x = x.numel * torch.tensor([], dtype=x.dtype).element_size()
        return x

    def _get_output_node(self, n):
        fwd_out = {x.uuid: x for x in n.meta["fwd_out"] if isinstance(x, torch.Tensor) and hasattr(x, 'uuid')}
        out_size = activation_size(fwd_out)
        out_node = [n.name] if out_size > 0 else []
        return out_size, out_node
    
    def _get_output_node_size(self, n):
        return self._get_output_node(n)[0]
    
    def _add_active_node(self, n, active_list):
        new_active = self._get_output_node(n)[1]
        for i in new_active:
            if i not in active_list:
                active_list.append(i)

    def _get_delete_node(self, user, user_to_last_uses):
        delete_size = 0
        delete_node = []
        if user.op not in ('placeholder', 'output'):
            nodes_to_delete = user_to_last_uses.get(user, [])
            if len(nodes_to_delete):
                out_node = [self._get_output_node(i) for i in nodes_to_delete]
                delete_size = sum([i[0] for i in out_node])
                for i in range(len(out_node)):
                    if out_node[i][0] > 0:
                        delete_node.append(out_node[i][1][0])
                    elif nodes_to_delete[i].op == 'placeholder':
                        delete_node.append(nodes_to_delete[i].name)
        return delete_size, delete_node
    
    def _get_delete_node_size(self, user, user_to_last_uses):
        return self._get_delete_node(user, user_to_last_uses)[0]
    
    def _remove_deactive_node(self, user, user_to_last_uses, active_list):
        delete_node = self._get_delete_node(user, user_to_last_uses)[1]
        for i in delete_node:
            active_list.remove(i)

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
        not_contiguous_ops = ['transpose', 'permute']

        if node.op == 'call_function' and any(n in node.name for n in ['matmul', 'reshape']):
            for n in node.args:
                if n in not_contiguous_list:
                    # matmul won't change origin tensor, but create a tmp copy
                    mem += self._get_output_node_size(n)
        elif node.op == 'call_module':
            for n in node.args:
                if n in not_contiguous_list:
                    # module will just make origin tensor to contiguous
                    if delete:
                        not_contiguous_list.remove(n)
        elif node.op == 'call_method' and any(i in node.name for i in not_contiguous_ops):
            if node not in not_contiguous_list:
                not_contiguous_list.append(node)
        elif any(i in node.args for i in not_contiguous_list):
            if node not in not_contiguous_list:
                not_contiguous_list.append(node)

        return mem

    def _get_chunk_ratio(self, node, chunk_dim, chunk_size):
        shape = node.meta['tensor_meta'].shape
        chunk_ratio = float(chunk_size) / shape[chunk_dim]
        return chunk_ratio


    def _get_chunk_delete_node_size(self, user, user_to_last_uses, chunk_ratio, node_list, start_node, end_node):
        if user.op in ('placeholder', 'output'):
            return 0
        nodes_to_delete = user_to_last_uses.get(user, [])
        delete_size = 0
        for n in nodes_to_delete:
            node_idx = _find_idx_by_name(n.name, node_list)
            if start_node <= node_idx < end_node:
                delete_size += self._get_output_node_size(n) * chunk_ratio
        return delete_size


    def _print_mem_log(self, log, nodes, title=None):
        if title:
            print(title)
        for idx, (l, n) in enumerate(zip(log, nodes)):
            print("%s:%.2f \t" % (n.name, l), end='')
            if (idx + 1) % 3 == 0:
                print("")
        print("\n")

    def _print_compute_op_mem_log(self, log, nodes, title=None):
        if title:
            print(title)
        for idx, (l, n) in enumerate(zip(log, nodes)):
            if n.op in ['placeholder', 'get_attr', 'output']:
                continue
            if any(i in n.name for i in ['getitem', 'getattr']):
                continue
            print("%s:%.2f \t" % (n.name, l), end='')
            if (idx + 1) % 3 == 0:
                print("")
        print("\n")
    
    def estimate_chunk_inference_mem(self, gm: torch.fx.GraphModule, start_nodes=None, end_nodes=None, chunk_dims=None, chunk_sizes=None):
        act_memory = 0.0
        act_memory_peak_log = []
        act_memory_after_node_log = []
        active_node_list = []
        active_node_list_log = []
        not_contiguous_list = []
        node_list = list(gm.graph.nodes)
        user_to_last_uses = self._get_last_usr(node_list)
        user_to_last_uses_no_free_var = self._get_last_usr(node_list)
        _delete_free_var_from_last_use(user_to_last_uses_no_free_var)
        
        use_chunk = all(i is not None for i in [start_nodes, end_nodes, chunk_dims, chunk_sizes])
        chunk_within = False
        chunk_region_idx = 0
        chunk_ratio = 1 # use it to estimate chunk mem

        for idx, node in enumerate(node_list):
            # if node in chunk start nodes, change chunk ratio and add chunk_tensor
            if use_chunk and idx in start_nodes:
                chunk_within = True
                chunk_ratio = self._get_chunk_ratio(node, chunk_dims[chunk_region_idx], chunk_sizes[chunk_region_idx])
                act_memory += self._get_output_node_size(node_list[end_nodes[chunk_region_idx]]) / (1024 ** 2)
                
            # if node is placeholder, just add the size of the node
            if node.op == 'placeholder':
                act_memory += self._get_meta_node_size(node) * chunk_ratio / (1024 ** 2)
                act_memory_peak_log.append(act_memory)
                active_node_list.append(node.name)
            # skip output
            elif node.op == 'output':
                continue
            # node is an operation, calculate tmp, output node and delete node memory
            else:
                # forward memory
                act_memory += self._get_contiguous_memory(node, not_contiguous_list) * chunk_ratio / (1024 ** 2)
                act_memory += self._get_output_node_size(node) * chunk_ratio / (1024 ** 2)
                # record max act memory
                act_memory_peak_log.append(act_memory)
                # delete useless memory
                act_memory -= self._get_contiguous_memory(node, not_contiguous_list, delete=True) * chunk_ratio / (1024 ** 2)
                if chunk_within:
                    act_memory -= self._get_chunk_delete_node_size(
                        node, user_to_last_uses_no_free_var, chunk_ratio, node_list, 
                        start_nodes[chunk_region_idx], end_nodes[chunk_region_idx]) / (1024 ** 2)
                else:
                    act_memory -= self._get_delete_node_size(node, user_to_last_uses_no_free_var) / (1024 ** 2)

            # log active node
            self._add_active_node(node, active_node_list)
            self._remove_deactive_node(node, user_to_last_uses, active_node_list)

            # if node in chunk end nodes, restore chunk settings
            if use_chunk and idx in end_nodes:
                act_memory -= self._get_output_node_size(node) * chunk_ratio / (1024 ** 2)
                chunk_within = False
                chunk_ratio = 1
                chunk_region_idx += 1
            
            act_memory_after_node_log.append(act_memory)
            active_node_list_log.append(copy.deepcopy(active_node_list))

        print("with chunk" if use_chunk else "without chunk")
        # self._print_mem_log(act_memory_peak_log, node_list, "peak")
        # self._print_mem_log(act_memory_after_node_log, node_list, "after")
        self._print_compute_op_mem_log(act_memory_peak_log, node_list, "peak")
        self._print_compute_op_mem_log(act_memory_after_node_log, node_list, "after")

        # param_memory = parameter_size(gm)
        # all_memory = act_memory + param_memory
        return act_memory_peak_log, act_memory_after_node_log, active_node_list_log


class ChunkRegionSearch(object):
    def __init__(self, gm) -> None:
        self.gm = gm
        self.node_list = list(gm.graph.nodes)
        self.memory_estimator = MemoryEstimator()
        self.index_tracer = NodeIndexTracer(gm)
        self.index_tracer.trace_node_idx()

    def _find_peak_node(self, mem_peak):
        max_value = max(mem_peak)
        max_idx = mem_peak.index(max_value)
        return max_idx
    
    def _get_free_var(self):
        free_var_idx = []
        for idx, n in enumerate(self.node_list):
            if n.op == 'placeholder':
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
    
    def _search_max_chunk_region(self, active_node, peak_node):
        free_vars = self._get_free_var()
        min_var = self._get_min_free_var(active_node, free_vars)
        
        # from peak_node to free_var
        chunk_region_start = None
        for i in range(peak_node, -1, -1):
            if len(active_node[i]) == min_var:
                chunk_region_start = i + 1
                break
            if i in free_vars or i == 0:
                raise RuntimeError()
        # from peak_node to len-2
        chunk_region_end = None
        for i in range(peak_node, len(active_node)):
            if len(active_node[i]) == min_var:
                chunk_region_end = i
                break
            if i in free_vars or i == 0:
                raise RuntimeError()
        return chunk_region_start, chunk_region_end
    
    def _not_compute(self, trace, chunk_range, dim_idx):
        if trace['idx'][dim_idx] not in trace['compute']:
            return True
        if trace['idx'][dim_idx] in trace['compute'] and \
            all(i < chunk_range[0] or i > chunk_range[1] for i in trace['compute'][trace['idx'][dim_idx]]):
            return True
        return False
    
    def _search_possible_chunk_regions(self, max_chunk_region, peak_node):
        possible_chunk_region = []
        output_trace = copy.deepcopy(self.index_tracer.idx_trace_list)
        input_trace = []
        for i, n in enumerate(self.node_list):
            if len(n.args) > 0 and n.op != 'output':
                input_idx = _find_idx_by_name(n.args[0].name, self.node_list)
                input_trace.append(output_trace[input_idx])
            else:
                input_trace.append(None)

        for before_idx in range(max_chunk_region[0], peak_node):
            for after_idx in range(peak_node, max_chunk_region[1] + 1):
                # skip non compute nodes
                if any(op in ['placeholder', 'get_attr', 'output'] for op in 
                       [self.node_list[before_idx].op, self.node_list[after_idx].op]):
                    continue
                if any(any(i in name for i in ['getitem', 'getattr']) for name in 
                       [self.node_list[before_idx].name, self.node_list[after_idx].name]):
                    continue
                
                # select free dim
                before_trace = input_trace[before_idx]
                after_trace = output_trace[after_idx]
                free_dim = []
                for i in range(min(len(before_trace['idx']), len(after_trace['idx']))):
                   if (before_trace['idx'][i] == after_trace['idx'][i] and 
                       self._not_compute(before_trace, (before_idx, after_idx), i) and
                       self._not_compute(after_trace, (before_idx, after_idx), i) and
                       self.node_list[after_idx].meta['tensor_meta'].shape[i] != 1):
                       free_dim.append(i)
                possible_chunk_region.append({'region': (before_idx, after_idx), 'dim': free_dim})
        return possible_chunk_region
    
    def _search_best_chunk_region(self, possible_chunk_regions):
        max_region_range = 0
        best_regions = None
        for i in possible_chunk_regions:
            if i['region'][1] - i['region'][0] > max_region_range:
                best_regions = i
                max_region_range = i['region'][1] - i['region'][0]
        return best_regions
    
    def _step_search(self, peak_node, active_node):
        max_chunk_region = self._search_max_chunk_region(active_node, peak_node)
        possible_chunk_regions = self._search_possible_chunk_regions(max_chunk_region, peak_node)
        best_chunk_region = self._search_best_chunk_region(possible_chunk_regions)
        return best_chunk_region
    
    def _stop_search(self, init_mem_peak, mem_peak):
        sorted_init_mem_peak = sorted(init_mem_peak)
        if max(mem_peak) < sorted_init_mem_peak[int(len(sorted_init_mem_peak) * 0.5)]:
            return True
        return False
    
    def search_region(self):
        chunk_regions = []
        init_mem_peak, _, active_node = self.memory_estimator.estimate_chunk_inference_mem(self.gm)
        mem_peak = init_mem_peak
        
        while True:
            peak_node = self._find_peak_node(mem_peak)
            chunk_region = self._step_search(peak_node, active_node)
            if chunk_region is None or len(chunk_region['dim']) == 0:
                break
            
            chunk_regions.append(chunk_region)
            mem_peak, _, active_node = self.memory_estimator.estimate_chunk_inference_mem(
                self.gm, [i['region'][0] for i in chunk_regions], 
                [i['region'][1] for i in chunk_regions], [i['dim'][0] for i in chunk_regions], [1] * len(chunk_regions))
            
            if self._stop_search(init_mem_peak, mem_peak):
                break

        return chunk_regions


def _gen_chunk_slice_dim(chunk_dim, chunk_idx_name, shape):
    new_shape = "["
    for idx, i in enumerate(shape):
        if idx == chunk_dim:
            new_shape += "%s:%s + chunk_size" % (chunk_idx_name, chunk_idx_name)
        else:
            new_shape += ":"
        new_shape += ", "
    new_shape = new_shape[:-2] + "]"
    return new_shape


def _get_first_non_single_dim(shape):
    for idx, i in enumerate(shape):
        if i == 1:
            continue
        else:
            return idx
    raise RuntimeError("can not get first non single dim for shape", shape)


def _gen_loop_start(chunk_input_meta, chunk_output, chunk_dim, chunk_size=2):
    if len(chunk_input_meta) == 1:
        node = chunk_input_meta[0]
        node_shape = node.meta['tensor_meta'].shape
        free_shape = [node_shape[i] if i in chunk_dim else 1 for i in range(len(node_shape))]
        chunk_dim = _get_first_non_single_dim(free_shape)
        chunk_slice = _gen_chunk_slice_dim(chunk_dim, "gen_chunk_idx", node_shape)
        out_shape = str(list(chunk_output.meta['tensor_meta'].shape))
        
        context = "chunk_result = torch.empty(%s, dtype=%s.dtype, device=%s.device); chunk_size = %d\nfor gen_chunk_idx in range" % (
            out_shape, node.name, node.name, chunk_size)
        context += "(0, %s.shape[%d], chunk_size):\n" % (node.name, chunk_dim)
        context += "    chunk_tensor = %s%s\n" % (node.name, chunk_slice)
    else:
        raise NotImplementedError("input with size %d not implemented" % len(chunk_input_meta))
    return context


def _gen_loop_end(chunk_outputs, chunk_inputs, node_list, chunk_dim):
    chunk_inputs_name = chunk_inputs[0].name
    chunk_outputs_name = chunk_outputs.name
    chunk_outputs_idx = _find_idx_by_name(chunk_outputs_name, node_list)
    chunk_output_shape = chunk_outputs.meta['tensor_meta'].shape
    free_shape = [chunk_output_shape[i] if i in chunk_dim else 1 for i in range(len(chunk_output_shape))]
    chunk_dim = _get_first_non_single_dim(free_shape)
    chunk_slice = _gen_chunk_slice_dim(chunk_dim, "gen_chunk_idx", chunk_output_shape)
    context = "    chunk_result%s = %s\n" % (chunk_slice, chunk_outputs_name)

    context += chunk_outputs_name + " = chunk_result;  chunk_result = None;  chunk_size = None"
    
    # determine if its the last use for chunk input
    users_name = list(chunk_inputs[0].users.keys())
    if all([_find_idx_by_name(user.name, node_list) <= chunk_outputs_idx for user in users_name]):
        context += ";  %s = None" % chunk_inputs_name

    context += "\n"
    return context


def _find_input_and_output_nodes(nodes: List[Node]):
    """
    Find the input and output node names which are not found in the given list of nodes.
    """
    input_nodes = []
    output_nodes = []

    # if a node has an input node which is not in the node list
    # we treat that input node as the input of the checkpoint function
    for node in nodes:
        for input_node in node._input_nodes.keys():
            node_repr = repr(input_node)
            if input_node not in nodes and input_node not in input_nodes:
                input_nodes.append(input_node)

    # if a node has a user node which is not in the node list
    # we treat that user node as the node receiving the current node output
    for node in nodes:
        for output_node in node.users.keys():
            node_repr = repr(node)
            if output_node not in nodes and output_node not in output_nodes:
                output_nodes.append(output_node)

    return input_nodes, output_nodes


def _find_idx_by_name(name, nodes_list):
    for idx, node in enumerate(nodes_list):
        if node.name == name:
            return idx
    raise RuntimeError("name %s not found in node list" % name)


def _replace_name(context, name_from, name_to):
    patterns = [(" ", " "), (" ", "."), (" ", ","), ("(", ")"), ("(", ",")]
    for p in patterns:
        source = p[0] + name_from + p[1]
        target = p[0] + name_to + p[1]
        if source in context:
            context = context.replace(source, target)
    return context


def emit_code_with_chunk(body, ckpt_func, nodes, emit_node_func, delete_unused_value_func, meta_nodes, meta_graph):
    """Emit code with nested activation checkpoint
    When we detect some of the node.activation_checkpoint is a List, we will use
    this function to emit the activation checkpoint codes.

    Args:
        body: forward code
        ckpt_func: checkpoint functions code
        nodes: graph.nodes
        emit_node_func: function to emit node
        delete_unused_value_func: function to remove the unused value
    """

    # find the offload regions
    chunk_region_search = ChunkRegionSearch(meta_graph)
    chunk_search = chunk_region_search.search_region()
    chunk_regions = [i['region'] for i in chunk_search]
    chunk_dims = [i['dim'] for i in chunk_search]
    
    chunk_starts = [item[0] for item in chunk_regions]
    chunk_ends = [item[1] for item in chunk_regions]
    chunk_inputs = []
    chunk_outputs = []
    within_chunk_region = False

    node_list = list(nodes)

    # find the input and output var names for each offload region
    for idx, (start, end) in enumerate(chunk_regions):
        offload_node_list = node_list[start:end + 1]
        inputs, outputs = _find_input_and_output_nodes(offload_node_list)
        chunk_inputs.append(inputs)
        chunk_outputs.append(outputs)
    chunk_inputs_idx = [[_find_idx_by_name(j.name, node_list) for j in i] for i in chunk_inputs]
    chunk_outputs_idx = [[_find_idx_by_name(j.name, node_list) for j in i] for i in chunk_outputs]
    chunk_inputs_names = []
    for i in chunk_inputs:
        for j in i:
            chunk_inputs_names.append(j.name)
    
    # this flag is to prevent repeated insert of save tensors
    # hooks definition in ckpt_func
    node_idx = 0
    region_idx = 0
    while node_idx < len(node_list):
        node = node_list[node_idx]

        if node_idx in chunk_starts:
            within_chunk_region = True
                
            # add for loop
            chunk_input_meta = [meta_nodes[i] for i in chunk_inputs_idx[region_idx]]
            body.append(_gen_loop_start(chunk_input_meta, node_list[chunk_ends[region_idx]], chunk_dims[region_idx]))

        if within_chunk_region:
            emit_node_func(node, body)
            # replace input var with chunk var
            body[-1] = _replace_name(body[-1], chunk_inputs[region_idx][0].name, 'chunk_tensor')
            body[-1] = '    ' + body[-1]
            delete_unused_value_func(node, body, chunk_inputs_names)

        else:
            emit_node_func(node, body)
            if node_idx not in chunk_inputs:
                delete_unused_value_func(node, body, chunk_inputs_names)

        if node_idx in chunk_ends:
            body.append(_gen_loop_end(node, chunk_inputs[region_idx], node_list, chunk_dims[region_idx]))
            within_chunk_region = False
            region_idx += 1

        node_idx += 1


if CODEGEN_AVAILABLE:

    class ChunkCodeGen(CodeGen):
        def __init__(self, meta_graph):
            super().__init__()
            self.meta_graph = meta_graph
            self.meta_node = list(meta_graph.graph.nodes)

        def _gen_python_code(self, nodes, root_module: str, namespace: _Namespace) -> PythonCode:
            free_vars: List[str] = []
            body: List[str] = []
            globals_: Dict[str, Any] = {}
            wrapped_fns: Dict[str, None] = {}

            # Wrap string in list to pass by reference
            maybe_return_annotation: List[str] = ['']

            def add_global(name_hint: str, obj: Any):
                """Add an obj to be tracked as a global.

                We call this for names that reference objects external to the
                Graph, like functions or types.

                Returns: the global name that should be used to reference 'obj' in generated source.
                """
                if _is_from_torch(obj) and obj != torch.device:    # to support registering torch.device
                    # HACK: workaround for how torch custom ops are registered. We
                    # can't import them like normal modules so they must retain their
                    # fully qualified name.
                    return _get_qualified_name(obj)

                # normalize the name hint to get a proper identifier
                global_name = namespace.create_name(name_hint, obj)

                if global_name in globals_:
                    assert globals_[global_name] is obj
                    return global_name
                globals_[global_name] = obj
                return global_name

            # set _custom_builtins here so that we needn't import colossalai in forward
            _custom_builtins["colossalai"] = _CustomBuiltin("import colossalai", colossalai)

            # Pre-fill the globals table with registered builtins.
            for name, (_, obj) in _custom_builtins.items():
                add_global(name, obj)

            def type_repr(o: Any):
                if o == ():
                    # Empty tuple is used for empty tuple type annotation Tuple[()]
                    return '()'

                typename = _type_repr(o)

                if hasattr(o, '__origin__'):
                    # This is a generic type, e.g. typing.List[torch.Tensor]
                    origin_type = _origin_type_map.get(o.__origin__, o.__origin__)
                    origin_typename = add_global(_type_repr(origin_type), origin_type)

                    if hasattr(o, '__args__'):
                        # Assign global names for each of the inner type variables.
                        args = [type_repr(arg) for arg in o.__args__]

                        if len(args) == 0:
                            # Bare type, such as `typing.Tuple` with no subscript
                            # This code-path used in Python < 3.9
                            return origin_typename

                        return f'{origin_typename}[{",".join(args)}]'
                    else:
                        # Bare type, such as `typing.Tuple` with no subscript
                        # This code-path used in Python 3.9+
                        return origin_typename

                # Common case: this is a regular module name like 'foo.bar.baz'
                return add_global(typename, o)

            def _format_args(args: Tuple[Argument, ...], kwargs: Dict[str, Argument]) -> str:

                def _get_repr(arg):
                    # Handle NamedTuples (if it has `_fields`) via add_global.
                    if isinstance(arg, tuple) and hasattr(arg, '_fields'):
                        qualified_name = _get_qualified_name(type(arg))
                        global_name = add_global(qualified_name, type(arg))
                        return f"{global_name}{repr(tuple(arg))}"
                    return repr(arg)

                args_s = ', '.join(_get_repr(a) for a in args)
                kwargs_s = ', '.join(f'{k} = {_get_repr(v)}' for k, v in kwargs.items())
                if args_s and kwargs_s:
                    return f'{args_s}, {kwargs_s}'
                return args_s or kwargs_s

            # Run through reverse nodes and record the first instance of a use
            # of a given node. This represents the *last* use of the node in the
            # execution order of the program, which we will use to free unused
            # values
            node_to_last_use: Dict[Node, Node] = {}
            user_to_last_uses: Dict[Node, List[Node]] = {}

            def register_last_uses(n: Node, user: Node):
                if n not in node_to_last_use:
                    node_to_last_use[n] = user
                    user_to_last_uses.setdefault(user, []).append(n)

            for node in reversed(nodes):
                map_arg(node.args, lambda n: register_last_uses(n, node))
                map_arg(node.kwargs, lambda n: register_last_uses(n, node))
            
            _delete_free_var_from_last_use(user_to_last_uses)
            
            # NOTE: we add a variable to distinguish body and ckpt_func
            def delete_unused_values(user: Node, body, to_keep=[]):
                """
                Delete values after their last use. This ensures that values that are
                not used in the remainder of the code are freed and the memory usage
                of the code is optimal.
                """
                if user.op == 'placeholder':
                    return
                if user.op == 'output':
                    body.append('\n')
                    return
                nodes_to_delete = user_to_last_uses.get(user, [])
                nodes_to_delete = [i for i in nodes_to_delete if i.name not in to_keep]
                if len(nodes_to_delete):
                    to_delete_str = ' = '.join([repr(n) for n in nodes_to_delete] + ['None'])
                    body.append(f';  {to_delete_str}\n')
                else:
                    body.append('\n')

            # NOTE: we add a variable to distinguish body and ckpt_func
            def emit_node(node: Node, body):
                maybe_type_annotation = '' if node.type is None else f' : {type_repr(node.type)}'
                if node.op == 'placeholder':
                    assert isinstance(node.target, str)
                    maybe_default_arg = '' if not node.args else f' = {repr(node.args[0])}'
                    free_vars.append(f'{node.target}{maybe_type_annotation}{maybe_default_arg}')
                    raw_name = node.target.replace('*', '')
                    if raw_name != repr(node):
                        body.append(f'{repr(node)} = {raw_name}\n')
                    return
                elif node.op == 'call_method':
                    assert isinstance(node.target, str)
                    body.append(
                        f'{repr(node)}{maybe_type_annotation} = {_format_target(repr(node.args[0]), node.target)}'
                        f'({_format_args(node.args[1:], node.kwargs)})')
                    return
                elif node.op == 'call_function':
                    assert callable(node.target)
                    # pretty print operators
                    if node.target.__module__ == '_operator' and node.target.__name__ in magic_methods:
                        assert isinstance(node.args, tuple)
                        body.append(f'{repr(node)}{maybe_type_annotation} = '
                                    f'{magic_methods[node.target.__name__].format(*(repr(a) for a in node.args))}')
                        return

                    # pretty print inplace operators; required for jit.script to work properly
                    # not currently supported in normal FX graphs, but generated by torchdynamo
                    if node.target.__module__ == '_operator' and node.target.__name__ in inplace_methods:
                        body.append(f'{inplace_methods[node.target.__name__].format(*(repr(a) for a in node.args))};  '
                                    f'{repr(node)}{maybe_type_annotation} = {repr(node.args[0])}')
                        return

                    qualified_name = _get_qualified_name(node.target)
                    global_name = add_global(qualified_name, node.target)
                    # special case for getattr: node.args could be 2-argument or 3-argument
                    # 2-argument: attribute access; 3-argument: fall through to attrib function call with default value
                    if global_name == 'getattr' and \
                    isinstance(node.args, tuple) and \
                    isinstance(node.args[1], str) and \
                    node.args[1].isidentifier() and \
                    len(node.args) == 2:
                        body.append(
                            f'{repr(node)}{maybe_type_annotation} = {_format_target(repr(node.args[0]), node.args[1])}')
                        return
                    body.append(
                        f'{repr(node)}{maybe_type_annotation} = {global_name}({_format_args(node.args, node.kwargs)})')
                    if node.meta.get('is_wrapped', False):
                        wrapped_fns.setdefault(global_name)
                    return
                elif node.op == 'call_module':
                    assert isinstance(node.target, str)
                    body.append(f'{repr(node)}{maybe_type_annotation} = '
                                f'{_format_target(root_module, node.target)}({_format_args(node.args, node.kwargs)})')
                    return
                elif node.op == 'get_attr':
                    assert isinstance(node.target, str)
                    body.append(f'{repr(node)}{maybe_type_annotation} = {_format_target(root_module, node.target)}')
                    return
                elif node.op == 'output':
                    if node.type is not None:
                        maybe_return_annotation[0] = f" -> {type_repr(node.type)}"
                    body.append(self.generate_output(node.args[0]))
                    return
                raise NotImplementedError(f'node: {node.op} {node.target}')

            # Modified for activation checkpointing
            ckpt_func = []

            # if any node has a list of labels for activation_checkpoint, we
            # will use nested type of activation checkpoint codegen
            emit_code_with_chunk(body, ckpt_func, nodes, emit_node, delete_unused_values, self.meta_node, self.meta_graph)

            if len(body) == 0:
                # If the Graph has no non-placeholder nodes, no lines for the body
                # have been emitted. To continue to have valid Python code, emit a
                # single pass statement
                body.append('pass\n')

            if len(wrapped_fns) > 0:
                wrap_name = add_global('wrap', torch.fx.wrap)
                wrap_stmts = '\n'.join([f'{wrap_name}("{name}")' for name in wrapped_fns])
            else:
                wrap_stmts = ''

            if self._body_transformer:
                body = self._body_transformer(body)

            for name, value in self.additional_globals():
                add_global(name, value)

            # as we need colossalai.utils.checkpoint, we need to import colossalai
            # in forward function
            prologue = self.gen_fn_def(free_vars, maybe_return_annotation[0])
            prologue = ''.join(ckpt_func) + prologue
            prologue = prologue

            code = ''.join(body)
            code = '\n'.join('    ' + line for line in code.split('\n'))
            fn_code = f"""
{wrap_stmts}

{prologue}
{code}"""   
            print(fn_code)
            return PythonCode(fn_code, globals_)
