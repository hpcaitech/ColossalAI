import colossalai
import torch
import copy
from typing import List, Callable, Any, Tuple, Dict, Iterable

try:
    from torch.fx.node import Node, Argument, map_arg, _type_repr, _get_qualified_name
    from torch.fx.graph import _Namespace, PythonCode, _custom_builtins, _is_from_torch, _format_target, magic_methods, CodeGen, _origin_type_map, inplace_methods, _CustomBuiltin
    from colossalai.fx.profiler import calculate_fwd_out, calculate_fwd_tmp, parameter_size, activation_size
    CODEGEN_AVAILABLE = True
except:
    from torch.fx.graph import _Namespace, PythonCode, _custom_builtins, _is_from_torch, _format_target, magic_methods, _origin_type_map, _format_args, _CustomBuiltin
    from torch.fx.node import Node, Argument, map_arg, _type_repr, _get_qualified_name
    CODEGEN_AVAILABLE = False

if CODEGEN_AVAILABLE:
    __all__ = ['ChunkCodeGen']
else:
    __all__ = ['python_code_with_activation_checkpoint']


class NodeIndexTracer(object):
    def __init__(self, gm) -> None:
        self.gm = gm
        self.nodes_list = list(gm.graph.nodes)
        self.idx_trace_list = [{'idx': [], 'compute': []} for _ in range(len(self.nodes_list))] 
        self.idx_trace_equal = []
        self.idx_view_list = []
        self.idx_count = 1

    def add_index(self):
        self.idx_count += 1
        return self.idx_count - 1

    def inherit_computation(self, node_from, node_to):
        _, compute_from = self.find_trace_from_node(node_from)
        idx_to, compute_to = self.find_trace_from_node(node_to)
        for i in compute_from:
            if i in idx_to and i not in compute_to:
                compute_to.append(i)
    
    def mark_idx_equal(self, idx1, idx2):
        self.idx_trace_equal.append((idx1, idx2))
        
    def mark_computation(self, node, idx, dim):
        input_node_idx_trace = self.find_idx_trace_from_node(node)
        if isinstance(dim, int):
            dim = [dim]
        for d in dim:
            cur_idx = input_node_idx_trace[d]
            if cur_idx not in self.idx_trace_list[idx]['compute']:
                self.idx_trace_list[idx]['compute'].append(cur_idx)
    
    def find_trace_from_node(self, node):
        node_idx = _find_idx_by_name(node.name, self.nodes_list)
        node_dict = self.idx_trace_list[node_idx]
        return node_dict['idx'], node_dict['compute']
    
    def find_idx_trace_from_node(self, node):
        node_idx = _find_idx_by_name(node.name, self.nodes_list)
        return self.idx_trace_list[node_idx]['idx']
    
    def find_compute_trace_from_node(self, node):
        node_idx = _find_idx_by_name(node.name, self.nodes_list)
        return self.idx_trace_list[node_idx]['compute']
    
    def assign_index_as_input(self, node, node_idx):
        input_node_idx = _find_idx_by_name(node.args[0].name, self.nodes_list)
        input_node_idx_trace = self.idx_trace_list[input_node_idx]['idx']
        
        new_idx_trace = copy.deepcopy(input_node_idx_trace)
        self.idx_trace_list[node_idx]['idx'] = new_idx_trace
    
    def assign_all_index(self, node, node_idx):
        shape = node.meta['tensor_meta'].shape
        new_trace = []
        for _ in shape:
            new_trace.append(self.add_index())
        self.idx_trace_list[node_idx]['idx'] = new_trace   

    def assign_transpose_index(self, node, node_idx):
        tranpose_dim = node.args[1:]
        input_node_idx_trace = self.find_idx_trace_from_node(node.args[0])
        
        new_idx_trace = copy.deepcopy(input_node_idx_trace)
        new_idx_trace[tranpose_dim[0]] = input_node_idx_trace[tranpose_dim[1]]
        new_idx_trace[tranpose_dim[1]] = input_node_idx_trace[tranpose_dim[0]]

        self.idx_trace_list[node_idx]['idx'] = new_idx_trace
        self.inherit_computation(node.args[0], node)
        
    def assign_permute_index(self, node, node_idx):
        permute_dim = node.args[1:]
        input_node_idx_trace = self.find_idx_trace_from_node(node.args[0])
        
        new_idx_trace = copy.deepcopy(input_node_idx_trace)
        for idx, d in enumerate(permute_dim):
            new_idx_trace[idx] = input_node_idx_trace[d]

        self.idx_trace_list[node_idx]['idx'] = new_idx_trace
        self.inherit_computation(node.args[0], node)
        
    def assign_linear_index(self, node, node_idx):
        input_node, weight, bias = node.args
        input_node_idx_trace = self.find_idx_trace_from_node(input_node)
        weight_idx_trace = self.find_idx_trace_from_node(weight)
        
        new_idx_trace = copy.deepcopy(input_node_idx_trace)
        new_idx_trace[-1] = weight_idx_trace[1]
        self.idx_trace_list[node_idx]['idx'] = new_idx_trace

        self.inherit_computation(input_node, node)
        self.mark_computation(node, node_idx, [-1])
        self.mark_idx_equal(input_node_idx_trace[-1], weight_idx_trace[0])
        
        if bias:
            bias_idx_trace = self.find_idx_trace_from_node(bias)
            self.mark_idx_equal(input_node_idx_trace[-1], bias_idx_trace[0])

    def assign_matmul_index(self, node, node_idx):
        matmul_left, matmul_right = node.args
        matmul_left_idx_trace = self.find_idx_trace_from_node(matmul_left)
        matmul_right_idx_trace = self.find_idx_trace_from_node(matmul_right)
        
        assert(len(matmul_left_idx_trace) == len(matmul_right_idx_trace))
        new_idx_trace = copy.deepcopy(matmul_left_idx_trace)
        new_idx_trace[-1] = matmul_right_idx_trace[-1]
        self.idx_trace_list[node_idx]['idx'] = new_idx_trace

        self.inherit_computation(matmul_left, node)
        self.inherit_computation(matmul_right, node)
        self.mark_computation(node, node_idx, [-1])
        self.mark_idx_equal(matmul_left_idx_trace[-1], matmul_right_idx_trace[-2])

    def assign_layernorm_index(self, node, idx):
        self.assign_index_as_input(node, idx)
        self.inherit_computation(node.args[0], node)
        self.mark_computation(node, idx, [-1, -2])
    
    def assign_elementwise_index(self, node, idx):
        self.assign_index_as_input(node, idx)
        for node_in in node.args:
            if type(node_in) not in (int, float):
                self.inherit_computation(node_in, node)
                
    def assign_softmax_index(self, node, idx):
        self.assign_index_as_input(node, idx)
        self.mark_computation(node, idx, [node.kwargs['dim']])

    def assign_view_reshape_index(self, node, node_idx):
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
        origin_trace = self.find_idx_trace_from_node(origin_node)
        new_trace = copy.deepcopy(origin_trace)
        dim_from.reverse()
        for i in dim_from:
            new_trace.pop(i)
        for i in dim_to:
            new_trace.insert(i, self.add_index())
        self.idx_trace_list[node_idx]['idx'] = new_trace
        
        # inherit computation
        self.inherit_computation(origin_node, node)
        compute_log = self.find_compute_trace_from_node(origin_node)
        for i in dim_from:
            if origin_trace[i] in compute_log:
                for j in dim_to:
                    self.mark_computation(node, node_idx, [j])
                break
        
        # log view
        view_dict = {"idx_from": [origin_trace[i] for i in dim_from],
                     "dim_from": dim_from,
                     "idx_to": [new_trace[i] for i in dim_to],
                     "dim_to": dim_to}
        self.idx_view_list.append(view_dict) 
        
    def trace_node_idx(self):
        for idx, node in enumerate(self.nodes_list):
            if node.op == 'placeholder':
                self.assign_all_index(node, idx)
            elif node.op == 'call_method':
                if 'transpose' in node.name:
                    self.assign_transpose_index(node, idx)
                elif 'permute' in node.name:
                    self.assign_permute_index(node, idx)
                elif 'view' in node.name or 'reshape' in node.name:
                    self.assign_view_reshape_index(node, idx)
                else:
                    raise NotImplementedError(node.name, "method not implemented yet!")
            elif node.op == 'call_function':
                if 'linear' in node.name:
                    self.assign_linear_index(node, idx)
                elif 'matmul' in node.name:
                    self.assign_matmul_index(node, idx)
                elif 'softmax' in node.name:
                    self.assign_softmax_index(node, idx)
                elif any(n in node.name for n in ['mul', 'add', 'sigmoid', 'relu']):
                    self.assign_elementwise_index(node, idx)
                elif 'getattr' in node.name:
                    continue # get attr like shape
                elif 'getitem' in node.name:
                    continue # get item in list
                else:
                    raise NotImplementedError(node.name, "function not implemented yet!")
            elif node.op == 'call_module':
                if any(n in node.name for n in ['layernorm', 'norm']):
                    self.assign_layernorm_index(node, idx)
                else:
                    raise NotImplementedError(node.name, "module not implemented yet!")
            elif node.op == 'get_attr':
                self.assign_all_index(node, idx) # get param
            elif node.op == 'output':
                continue
            else:
                raise NotImplementedError(node.op, "op not implemented yet!")

def _get_meta_node_size(x):
    x = x.meta['tensor_meta']
    x = x.numel * torch.tensor([], dtype=x.dtype).element_size()
    return x


def _get_output_node_size(n):
    fwd_out = {x.uuid: x for x in n.meta["fwd_out"] if isinstance(x, torch.Tensor) and hasattr(x, 'uuid')}
    return activation_size(fwd_out)


def _get_delete_node_size(user, user_to_last_uses):
    if user.op in ('placeholder', 'output'):
        return 0
    nodes_to_delete = user_to_last_uses.get(user, [])
    if len(nodes_to_delete):
        delete_size = sum([_get_output_node_size(i) for i in nodes_to_delete])
        return delete_size
    return 0


def _get_last_usr(nodes):
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


def _delete_free_var_from_last_use(user_to_last_uses):
    for key, value in user_to_last_uses.items():
        for n in value:
            if n.op == 'placeholder':
                user_to_last_uses[key].remove(n)


def _get_contiguous_memory(node, not_contiguous_list, delete=False):
    mem = 0
    not_contiguous_ops = ['transpose', 'permute']

    if node.op == 'call_function' and 'matmul' in node.name:
        for n in node.args:
            if n in not_contiguous_list:
                # matmul won't change origin tensor, but create a tmp copy
                mem += _get_output_node_size(n)
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


def _estimate_inference_mem(gm: torch.fx.GraphModule):
    act_memory = 0.0
    act_memory_peak_log = []
    act_memory_after_node_log = []
    not_contiguous_list = []
    user_to_last_uses = _get_last_usr(list(gm.graph.nodes))
    _delete_free_var_from_last_use(user_to_last_uses)
    for node in gm.graph.nodes:
        # if node is placeholder, just add the size of the node
        if node.op == 'placeholder':
            act_memory += _get_meta_node_size(node) / (1024 ** 2)
            act_memory_peak_log.append(act_memory)
            act_memory_after_node_log.append(act_memory)
        # skip output
        elif node.op == 'output':
            continue
        # node is an operation, calculate tmp, output node and delete node memory
        else:
            # forward memory
            act_memory += _get_contiguous_memory(node, not_contiguous_list) / (1024 ** 2)
            act_memory += _get_output_node_size(node) / (1024 ** 2)
            # record max act memory
            act_memory_peak_log.append(act_memory)
            # delete useless memory
            act_memory -= _get_delete_node_size(node, user_to_last_uses) / (1024 ** 2)
            act_memory -= _get_contiguous_memory(node, not_contiguous_list, delete=True) / (1024 ** 2)
            act_memory_after_node_log.append(act_memory)

    print("no chunk")
    _print_mem_log(act_memory_peak_log, list(gm.graph.nodes), "peak")
    _print_mem_log(act_memory_after_node_log, list(gm.graph.nodes), "after")
    
    param_memory = parameter_size(gm)
    return act_memory + param_memory, param_memory


def _get_chunk_ratio(node, chunk_dim, chunk_size):
    shape = node.meta['tensor_meta'].shape
    chunk_ratio = float(chunk_size) / shape[chunk_dim]
    return chunk_ratio


def _get_chunk_delete_node_size(user, user_to_last_uses, chunk_ratio, node_list, start_node, end_node):
    if user.op in ('placeholder', 'output'):
        return 0
    nodes_to_delete = user_to_last_uses.get(user, [])
    delete_size = 0
    for n in nodes_to_delete:
        node_idx = _find_idx_by_name(n.name, node_list)
        if start_node <= node_idx < end_node:
            delete_size += _get_output_node_size(n) * chunk_ratio
    return delete_size


def _print_mem_log(log, nodes, title=None):
    if title:
        print(title)
    for idx, (l, n) in enumerate(zip(log, nodes)):
        print("%s:%.2f \t" % (n.name, l), end='')
        if (idx + 1) % 3 == 0:
            print("")
    print("\n")


def _estimate_chunk_inference_mem(gm: torch.fx.GraphModule, start_nodes, end_nodes, chunk_dims, chunk_sizes):
    act_memory = 0.0
    act_memory_peak_log = []
    act_memory_after_node_log = []
    not_contiguous_list = []
    user_to_last_uses = _get_last_usr(list(gm.graph.nodes))
    _delete_free_var_from_last_use(user_to_last_uses)
    within_chunk = False
    region_idx = 0
    chunk_ratio = 1 # use it to estimate chunk mem
    node_list = list(gm.graph.nodes)

    for idx, node in enumerate(node_list):
        # if node in chunk start nodes, change chunk ratio and add chunk_tensor
        if idx in start_nodes:
            within_chunk = True
            chunk_ratio = _get_chunk_ratio(node, chunk_dims[region_idx], chunk_sizes[region_idx])
            act_memory += _get_output_node_size(node_list[end_nodes[region_idx]]) / (1024 ** 2)
            
        # if node is placeholder, just add the size of the node
        if node.op == 'placeholder':
            act_memory += _get_meta_node_size(node) * chunk_ratio / (1024 ** 2)
            act_memory_peak_log.append(act_memory)
        # skip output
        elif node.op == 'output':
            continue
        # node is an operation, calculate tmp, output node and delete node memory
        else:
            # forward memory
            # TODO: permute will create a tmp copy if not contiguous
            act_memory += _get_contiguous_memory(node, not_contiguous_list) * chunk_ratio / (1024 ** 2)
            act_memory += _get_output_node_size(node) * chunk_ratio / (1024 ** 2)
            # record max act memory
            act_memory_peak_log.append(act_memory)
            # delete useless memory
            act_memory -= _get_contiguous_memory(node, not_contiguous_list, delete=True) * chunk_ratio / (1024 ** 2)
            if within_chunk:
                act_memory -= _get_chunk_delete_node_size(
                    node, user_to_last_uses, chunk_ratio, node_list, 
                    start_nodes[region_idx], end_nodes[region_idx]) / (1024 ** 2)
            else:
                act_memory -= _get_delete_node_size(node, user_to_last_uses) / (1024 ** 2)
            
        if idx in end_nodes:
            act_memory -= _get_output_node_size(node) * chunk_ratio / (1024 ** 2)
            within_chunk = False
            chunk_ratio = 1
            region_idx += 1
        
        act_memory_after_node_log.append(act_memory)

    print("chunk")
    _print_mem_log(act_memory_peak_log, node_list, "peak")
    _print_mem_log(act_memory_after_node_log, node_list, "after")

    param_memory = parameter_size(gm)
    return act_memory + param_memory, param_memory


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


def _gen_loop_start(chunk_input_meta, chunk_output, chunk_size=2):
    if len(chunk_input_meta) == 1:
        node = chunk_input_meta[0]
        node_shape = node.meta['tensor_meta'].shape
        chunk_dim = _get_first_non_single_dim(node_shape)
        chunk_slice = _gen_chunk_slice_dim(chunk_dim, "gen_chunk_idx", node_shape)
        out_shape = str(list(chunk_output.meta['tensor_meta'].shape))
        
        context = "chunk_result = torch.empty(%s, dtype=%s.dtype, device=%s.device); chunk_size = %d\nfor gen_chunk_idx in range" % (
            out_shape, node.name, node.name, chunk_size)
        context += "(0, %s.shape[%d], chunk_size):\n" % (node.name, chunk_dim)
        context += "    chunk_tensor = %s%s\n" % (node.name, chunk_slice)
    else:
        raise NotImplementedError("input with size %d not implemented" % len(chunk_input_meta))
    return context


def _gen_loop_end(chunk_outputs, chunk_inputs, node_list):
    chunk_inputs_name = chunk_inputs[0].name
    chunk_outputs_name = chunk_outputs.name
    chunk_outputs_idx = _find_idx_by_name(chunk_outputs_name, node_list)
    chunk_output_shape = chunk_outputs.meta['tensor_meta'].shape
    chunk_dim = _get_first_non_single_dim(chunk_output_shape)
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
            if input_node not in nodes and node_repr not in input_nodes:
                input_nodes.append(input_node)

    # if a node has a user node which is not in the node list
    # we treat that user node as the node receiving the current node output
    for node in nodes:
        for output_node in node.users.keys():
            node_repr = repr(node)
            if output_node not in nodes and node_repr not in output_nodes:
                output_nodes.append(output_node)

    return input_nodes, output_nodes


def _find_idx_by_name(name, nodes_list):
    for idx, node in enumerate(nodes_list):
        if node.name == name:
            return idx
    raise RuntimeError("name %s not found in node list" % name)
        

def _find_offload_regions(nodes: List[Node]):
    """This function is to find the offload regions
    In pofo algorithm, during annotation, we will annotate the offload region with the 
    list in the form of [idx, offload_input, offload_bar]. idx indicates the offload
    region's index, offload_input is a bool type indicates whether we need to offload
    the input, offload_bar is a bool type indicates whether we need to offload all the
    intermediate x_bars of this region.
    """
    offload_regions = []
    offload_labels = []
    start = -1
    end = -1
    current_region = None

    for idx, node in enumerate(nodes):
        if hasattr(node, 'activation_offload') and isinstance(getattr(node, 'activation_offload', None), Iterable):
            act_offload_label = node.activation_offload

            if current_region == None:
                current_region = act_offload_label
                start = idx
                offload_labels.append(act_offload_label)

            if act_offload_label != current_region:
                assert start != -1
                offload_regions.append((start, idx - 1))
                offload_labels.append(act_offload_label)
                current_region = act_offload_label
                start = idx
                end = -1

        else:
            if current_region is not None:
                end = idx - 1
                assert start != -1 and end != -1
                offload_regions.append((start, end))
                start = end = -1
                current_region = None

            else:
                pass

    return offload_regions, offload_labels


def _gen_ckpt_fn_def(label, free_vars: List[str]) -> str:
    """
    Generate the checkpoint function definition
    """
    return f"def checkpoint_{label}({', '.join(['self'] + free_vars)}):"


def _gen_ckpt_output(output_vars: List[str]) -> str:
    """
    Generate the return statement for checkpoint region
    """
    return f"return {', '.join(output_vars)}"


def _gen_ckpt_usage(label, activation_offload, input_vars, output_vars, use_reentrant=True):
    """
    Generate the checkpoint function call code text
    """
    outputs = ', '.join(output_vars)
    inputs = ', '.join(input_vars)
    return f'{outputs} = colossalai.utils.activation_checkpoint.checkpoint(self.checkpoint_{label}, {activation_offload}, {inputs}, use_reentrant={use_reentrant})'


def _end_of_ckpt(node: Node, check_idx: int) -> bool:
    """Check if the node could end the ckpt region

    Args:
        node (Node): torch.fx.Node
        check_idx (int): the index of checkpoint level for 
        nested checkpoint

    Returns:
        bool
    """
    if hasattr(node, "activation_checkpoint"):
        if isinstance(node.activation_checkpoint, list):
            return node.activation_checkpoint[check_idx] == None
        else:
            return False
    else:
        return True


def _find_nested_ckpt_regions(nodes, check_idx=0):
    """
    Find the nested checkpoint regions given a list of consecutive nodes. The outputs 
    will be list of tuples, each tuple is in the form of (start_index, end_index).
    """
    ckpt_regions = []
    start = -1
    end = -1
    current_region = None

    for idx, node in enumerate(nodes):
        if hasattr(node, 'activation_checkpoint'):
            if isinstance(getattr(node, 'activation_checkpoint'), int):
                act_ckpt_label = node.activation_checkpoint
            else:
                act_ckpt_label = node.activation_checkpoint[check_idx]

            # this activation checkpoint label is not set yet
            # meaning this is the first node of the activation ckpt region
            if current_region is None:
                current_region = act_ckpt_label
                start = idx

            # if activation checkpoint has changed
            # we restart the tracking
            # e.g. node ckpt states = [ckpt1, ckpt2, ckpt2, ckpt2]
            if act_ckpt_label != current_region:
                assert start != -1
                ckpt_regions.append((start, idx - 1))
                current_region = act_ckpt_label
                start = idx
                end = -1
        elif current_region is not None and _end_of_ckpt(node, check_idx):
            # used to check the case below
            # node ckpt states = [ckpt, ckpt, non-ckpt]
            end = idx - 1
            assert start != -1 and end != -1
            ckpt_regions.append((start, end))
            start = end = -1
            current_region = None
        else:
            pass

    if current_region is not None:
        end = len(nodes) - 1
        ckpt_regions.append((start, end))
    return ckpt_regions


def emit_ckpt_func(body,
                   ckpt_func,
                   node_list: List[Node],
                   emit_node_func,
                   delete_unused_value_func,
                   level=0,
                   in_ckpt=False):
    """Emit ckpt fuction in nested way

    Args:
        body: forward code, in recursive calls, this part will be checkpoint
        functions code
        ckpt_func: checkpoint functions code, in recursive calls, this part
        will be a buffer
        node_list (List[Node]): list of torch.fx.Node
        emit_node_func: function to emit a node
        delete_unused_value_func: function to delete unused value
        level (int, optional): checkpoint level. Defaults to 0.
        in_ckpt (bool, optional): indicates wether the func is in recursive
        call. Defaults to False.
    """
    inputs, outputs = _find_input_and_output_nodes(node_list)

    # if the current checkpoint function use int as label, using old generation method
    if isinstance(node_list[0].activation_checkpoint, int):
        label = node_list[0].activation_checkpoint
        ckpt_fn_def = _gen_ckpt_fn_def(label, inputs)
        ckpt_func.append(f'{ckpt_fn_def}\n')
        for node in node_list:
            emit_node_func(node, ckpt_func)
            ckpt_func[-1] = '    ' + ckpt_func[-1]
            delete_unused_value_func(node, ckpt_func)

        ckpt_func.append('    ' + _gen_ckpt_output(outputs) + '\n\n')
        activation_offload = getattr(node_list[0], "activation_offload", False)
        usage = _gen_ckpt_usage(label, activation_offload, inputs, outputs, False)
        usage += "\n"
        body.append(usage)

    # use nested ckpt function codegen
    else:
        # label given by each layer, e.g. if you are currently at level [0, 1, 1]
        # the label will be '0_1_1'
        label = "_".join([str(idx) for idx in node_list[0].activation_checkpoint[:level + 1]])
        ckpt_fn_def = _gen_ckpt_fn_def(label, inputs)
        ckpt_func.append(f'{ckpt_fn_def}\n')

        # if there is more level to fetch
        if level + 1 < len(node_list[0].activation_checkpoint):
            ckpt_regions = _find_nested_ckpt_regions(node_list, level + 1)
            start_idx = [item[0] for item in ckpt_regions]
            end_idx = [item[1] for item in ckpt_regions]

            # use ckpt_func_buffer to store nested checkpoint functions
            ckpt_func_buffer = []
            node_idx = 0
            while 1:
                if node_idx >= len(node_list):
                    break

                if node_idx in start_idx:
                    ckpt_node_list = node_list[node_idx:end_idx[start_idx.index(node_idx)] + 1]
                    emit_ckpt_func(ckpt_func, ckpt_func_buffer, ckpt_node_list, emit_node_func,
                                   delete_unused_value_func, level + 1, True)
                    node_idx += len(ckpt_node_list)

                else:
                    node = node_list[node_idx]
                    emit_node_func(node, ckpt_func)
                    ckpt_func[-1] = '    ' + ckpt_func[-1]
                    delete_unused_value_func(node, ckpt_func)
                    node_idx += 1

            ckpt_func.append('    ' + _gen_ckpt_output(outputs) + '\n\n')
            ckpt_func += ckpt_func_buffer
            activation_offload = getattr(node_list[0], "activation_offload", False)
            usage = _gen_ckpt_usage(label, activation_offload, inputs, outputs, False) + '\n'
            if in_ckpt:
                usage = '    ' + usage
            body.append(usage)

        # last level
        else:
            for node in node_list:
                emit_node_func(node, ckpt_func)
                ckpt_func[-1] = '    ' + ckpt_func[-1]
                delete_unused_value_func(node, ckpt_func)

            ckpt_func.append('    ' + _gen_ckpt_output(outputs) + '\n\n')
            activation_offload = getattr(node_list[0], "activation_offload", False)
            usage = _gen_ckpt_usage(label, activation_offload, inputs, outputs, False) + '\n'
            if in_ckpt:
                usage = '    ' + usage
            body.append(usage)


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
    chunk_regions = [(58, 62)]
    chunk_starts = [item[0] for item in chunk_regions]
    chunk_ends = [item[1] for item in chunk_regions]
    chunk_inputs = []
    chunk_outputs = []
    within_chunk_region = False

    node_list = list(nodes)
    _estimate_chunk_inference_mem(meta_graph, chunk_starts, chunk_ends, [1], [2])
    _estimate_inference_mem(meta_graph)
    node_index_tracer = NodeIndexTracer(meta_graph)
    node_index_tracer.trace_node_idx()

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
            body.append(_gen_loop_start(chunk_input_meta, node_list[chunk_ends[region_idx]]))

        if within_chunk_region:
            emit_node_func(node, body)
            # replace input var with chunk var
            if node_idx in chunk_starts:
                body[-1] = body[-1].replace("("+ chunk_inputs[region_idx][0].name +")", '(chunk_tensor)')
            body[-1] = '    ' + body[-1]
            delete_unused_value_func(node, body, chunk_inputs_names)

        else:
            emit_node_func(node, body)
            if node_idx not in chunk_inputs:
                delete_unused_value_func(node, body, chunk_inputs_names)

        if node_idx in chunk_ends:
            body.append(_gen_loop_end(node, chunk_inputs[region_idx], node_list))
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
