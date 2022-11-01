import heapq
import inspect
import torch

from colossalai.logging import get_dist_logger
from colossalai.nn.layer.utils import CheckpointModule
from typing import List

from collections import OrderedDict

def _binary_partition(weights: List, start: int, end: int):
    """Returns the binary partition position of `weights`, given the start
    position `st` and the end position `ed`.

    Args:
        weights (list): A python list to be binary partitioned
        start (int): the start position of the binary partition
        end (int): the end position of the binary partition

    Returns:
        int: the binary partition position of `weights`
    """
    w_sum = weights[end - 1]
    prefix = 0
    if start > 0:
        w_sum -= weights[start - 1]
        prefix = weights[start - 1]
    minimum = float("inf")
    for idx in range(start + 1, end):
        front = weights[idx - 1] - prefix
        diff = abs(w_sum - 2 * front)
        if diff < minimum:
            pos = idx
            minimum = diff

    return start, pos, end


def _heap_addition(weights: List, intervals: int, add_cnt: int):
    """
    """

    def _heap_push(heap, st, ed):
        value = weights[ed - 1]
        if st > 0:
            value -= weights[st - 1]
        heapq.heappush(heap, (-value, st, ed))

    ret_intervals = []
    heap = []

    for st, ed in intervals:
        _heap_push(heap, st, ed)

    while add_cnt > 0:
        _, st, ed = heapq.heappop(heap)
        if ed - st == 1:
            ret_intervals.append((st, ed))
        else:
            l, m, r = _binary_partition(weights, st, ed)
            _heap_push(heap, l, m)
            _heap_push(heap, m, r)
            add_cnt -= 1

    while heap:
        _, st, ed = heapq.heappop(heap)
        ret_intervals.append((st, ed))

    ret_intervals.sort()
    return ret_intervals


def _calc_partitions(weights, value):
    prev = 0
    prefix = 0
    num_block = 0
    intervals = []

    for idx, w in enumerate(weights):
        if weights[idx] - prefix > value:
            intervals.append((prev, idx))
            prev = idx
            prefix = weights[idx - 1]
            num_block += 1

    intervals.append((prev, len(weights)))
    return num_block + 1, intervals


def _binary_search(weights, num):
    length = len(weights)
    prefix = [1 if w == 0 else w for w in weights]
    for i in range(1, length):
        prefix[i] += prefix[i - 1]

    lower_bound = max(weights)
    upper_bound = prefix[length - 1]

    while upper_bound > lower_bound:
        mid = (upper_bound + lower_bound) // 2
        number, _ = _calc_partitions(prefix, mid)
        if number <= num:
            upper_bound = mid
        else:
            lower_bound = mid + 1

    num_block, intervals = _calc_partitions(prefix, upper_bound)
    if num_block < num:
        intervals = _heap_addition(prefix, intervals, num - num_block)

    return intervals


def partition_uniform(num_items, pipeline_parallel_size, num_chunks):
    assert num_items % num_chunks == 0, \
        "Layer length should be divided by the number of chunks, otherwise parameter method is recomended"

    logger = get_dist_logger()
    parts = [[] for _ in range(pipeline_parallel_size)]
    partition_items = num_items // num_chunks
    for idx in range(num_chunks):
        base_idx = idx * partition_items
        chunk_size = partition_items // pipeline_parallel_size
        left = pipeline_parallel_size - partition_items % pipeline_parallel_size
        if chunk_size == 0:
            logger.warning("Some nodes in Pipeline have no requests")

        for p in range(pipeline_parallel_size):
            st = base_idx
            base_idx += chunk_size + (p >= left)
            parts[p].append((st, base_idx))

    return parts


def partition_balanced(weights, pipeline_parallel_size, num_chunks):
    num_total = pipeline_parallel_size * num_chunks
    num_items = len(weights)
    if num_items <= num_total:
        return partition_uniform(num_items, pipeline_parallel_size, num_chunks)

    intervals = _binary_search(weights, num_total)

    current = 0
    parts = [[] for _ in range(pipeline_parallel_size)]
    for inter in intervals:
        parts[current].append(inter)
        current = (current + 1) % pipeline_parallel_size

    return parts


def build_kwargs_for_module(function, input_tensor, kw_dict):
    """
    Generally, the first argument of module.forward is an input tensor come from the previous layer.
    Therefore, we just filter the kwargs from second element of the dictionary.
    """
    sig = inspect.signature(function)
    if input_tensor is None:
        kwargs_offset = 0
    elif isinstance(input_tensor, torch.Tensor):
        kwargs_offset = 1
    elif isinstance(input_tensor, (tuple, OrderedDict)):
        #assert isinstance(input_tensor, tuple), f'input_tensor should be a torch.Tensor or a tuple object.'
        # Huggingface will take their own structures based on OrderedDict as the output 
        # between layers so we've to close this check.
        kwargs_offset = len(input_tensor)
    args_name_list = list(sig.parameters.keys())
    kw_dict = {k: v for k, v in kw_dict.items() if k in args_name_list[kwargs_offset:]}
    if len(kw_dict) == 0:
        return None
    return kw_dict


def build_kwargs_for_function(function, kw_dict):
    sig = inspect.signature(function)
    kw_dict = {k: v for k, v in kw_dict.items() if k in sig.parameters}
    if len(kw_dict) == 0:
        return None
    return kw_dict


def exec_func_with_kwargs(func, kw_dict, input_tensor, kwargs):
    """
    We suppose the callable object passed to to_layer_list method in two purpose:
        a. use the callable object to modify input tensor, such as \
            lambda x: torch.flatten(x, 1)
        b. use the callable object to modify kwargs value, such as \
            def foo(attention_mask=None):
                if attention_mask is not None:
                    batch_size = input_ids.shape[0]
                    attention_mask = attention_mask.view(batch_size, -1)
                return attention_mask
    """

    if kw_dict is not None:
        rst = func(**kw_dict)
        if isinstance(rst, tuple):
            for i, k in enumerate(kw_dict.keys()):
                kwargs[k] = rst[i]
        else:
            for k in kw_dict.keys():
                kwargs[k] = rst
        return input_tensor
    if isinstance(input_tensor, tuple):
        assert len(input_tensor) > 0, f'input_tensor should not be empty, when kw_dict is None.'
        sig = inspect.signature(func)
        func_args_num = len(sig.parameters)
        assert func_args_num <= len(
            input_tensor), f'func requires {func_args_num} arguments, but input_tensors only have {len(input_tensor)}.'
        if func_args_num < len(input_tensor):
            return func(*input_tensor[:func_args_num])
        else:
            return func(*input_tensor)
    assert isinstance(input_tensor, torch.Tensor), 'input_tensor should be a type of torch.Tensor or tuple.'
    return func(input_tensor)


def exec_funcs_with_kwargs(func_dict, func_key, input_tensor, kwargs):

    assert func_key in func_dict, f"{func_key} is not in the function_dict."
    funcs_to_exec = func_dict[func_key]
    if isinstance(funcs_to_exec, list):
        for f in funcs_to_exec:
            f_kwargs = build_kwargs_for_function(f, kwargs)
            input_tensor = exec_func_with_kwargs(f, f_kwargs, input_tensor, kwargs)
    else:
        f_kwargs = build_kwargs_for_function(funcs_to_exec, kwargs)
        input_tensor = exec_func_with_kwargs(funcs_to_exec, f_kwargs, input_tensor, kwargs)

    return input_tensor


def call_module(module, args=None, kwargs=None):
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if isinstance(module, CheckpointModule):
        forward_func = module._forward
    else:
        forward_func = module.forward
    sig = inspect.signature(forward_func)
    param_nums = len(sig.parameters)
    feed_nums = len(args) + len(kwargs)
    args_needed_nums = param_nums - len(kwargs)
    args_needed = args[:args_needed_nums]
    if isinstance(module, CheckpointModule):
        convert_kwargs_to_args = []
        for v in kwargs.values():
            convert_kwargs_to_args.append(v)
        return module(*args_needed, *convert_kwargs_to_args)
    else:
        return module(*args_needed, **kwargs)


def customized_partition(exec_seq):
    '''
    This function will analyze the exec_seq. In the exec_seq, users will use 'SPLIT_NODE' as an 
    annotation to note the partition point.
    '''
    customized_parts = {}
    start = 0
    stop = 0
    rank = 0
    for element in exec_seq:
        if isinstance(element, str):
            if element == 'SPLIT_NODE':
                customized_parts[rank] = [(start, stop)]
                start = stop
                rank += 1
            else:
                stop += 1
    customized_parts[rank] = [(start, stop)]
    return customized_parts
