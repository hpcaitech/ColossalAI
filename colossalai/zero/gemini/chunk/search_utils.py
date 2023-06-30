import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch.distributed as dist
import torch.nn as nn

from colossalai.tensor import ColoParameter
from colossalai.utils import is_ddp_ignored
from colossalai.zero.gemini.memory_tracer import MemStats, OrderedParamGenerator


def _filter_exlarge_params(model: nn.Module, size_dict: Dict[int, List[int]]) -> None:
    """_filter_exlarge_params

    Filter those parameters whose size is too large (more than 3x standard deviations) from others.

    Args:
        model (nn.Module): the model.
        size_dict (Dict[int, List[int]]): the size dict of parameters.
    """
    agg_size_list = []
    for key in size_dict:
        agg_size_list.extend(size_dict[key])

    if len(agg_size_list) == 0:
        return

    params_size_arr = np.array(agg_size_list)

    std = np.std(params_size_arr)
    mean = np.mean(params_size_arr)
    upper_limit = mean + 3 * std

    for key in size_dict:
        org_list = size_dict[key]
        size_dict[key] = list(filter(lambda x: x <= upper_limit, org_list))


def _get_unused_byte(size_list: List[int], chunk_size: int) -> int:
    """_get_unused_byte

    Get unused byte for a certain chunk size.

    Args:
        size_list (List[int]): the size list of parameters.
        chunk_size (int): the chunk size.

    Returns:
        int: the unused byte.
    """
    acc = 0
    left = 0
    for s in size_list:
        if s > left:
            acc += left
            left = chunk_size
        left -= s
    return left + acc


def _tensor_numel(local_param: ColoParameter, strict_ddp_flag: bool) -> int:
    """_tensor_numel

    Get the number of elements of a tensor.

    Args:
        local_param (ColoParameter): The local parameter.
        strict_ddp_flag (bool): whether to enable the strict ddp mode.

    Returns:
        int: the number of elements.
    """
    if strict_ddp_flag and type(local_param) is ColoParameter:
        return local_param.numel_global()
    else:
        # if local_param is not ColoParameter, we assume it's replicated
        return local_param.numel()


def classify_params_by_dp_degree(param_order: OrderedParamGenerator,
                                 strict_ddp_flag: bool = False) -> Dict[int, List[ColoParameter]]:
    """classify_params_by_dp_degree

    Classify the parameters by their dp degree

    Args:
        param_order (OrderedParamGenerator): the order of param be vised
        strict_ddp_flag (bool, optional): whether to enable the strict ddp mode. Defaults to False.

    Returns:
        Dict[int, List[ColoParameter]]: a dict contains the classification results.
        The keys are dp_degrees and the values are parameters.
    """
    params_dict: Dict[int, List[ColoParameter]] = dict()
    for param in param_order.generate():
        # assert isinstance(param, ColoParameter), "please init model in the ColoInitContext"
        if is_ddp_ignored(param):
            continue

        if strict_ddp_flag or type(param) is not ColoParameter:
            # if model is not initialized with ColoInitContext, we assume it's replicated
            # TODO(ver217): integrate DTensor
            param_key = dist.get_world_size()
        else:
            param_key = param.process_group.dp_world_size()

        if param_key not in params_dict:
            params_dict[param_key] = []
        params_dict[param_key].append(param)

    return params_dict


def search_chunk_configuration(
        model: nn.Module,
        search_range_m: float,
        search_interval: int,    # hidden size is the best value for the interval
        min_chunk_size_m: float = 32,
        filter_exlarge_params: bool = True,
        strict_ddp_flag: bool = False,
        memstas: Optional[MemStats] = None) -> Tuple[Dict, int, int]:
    """search_chunk_configuration

    Search the chunk configuration for a model.

    Args:
        model (nn.Module): torch module
        search_range_m (float): searching range divided by 2^20.
        search_interval (int): searching interval.
        min_chunk_size_m (float, optional): the minimum size of a distributed chunk, divided by 2^20..
        filter_exlarge_params (bool, optional): filter extreme large parameters. Defaults to True.
        strict_ddp_flag (bool, optional): whether to enable the strict ddp mode.
            all parameters keep replicated in this mode.

    Returns:
        Tuple[Dict, int]: chunk config (a dict of dp_degree -> chunk init args) and its memory chunk waste in byte.
    """

    if memstas is not None:
        param_order = memstas.param_order()
    else:
        # build the param visited order right now
        param_order = OrderedParamGenerator()
        for p in model.parameters():
            param_order.append(p)

    search_range = round(search_range_m * 1024**2)
    min_chunk_size = round(min_chunk_size_m * 1024**2)
    assert search_range >= 0

    params_dict = classify_params_by_dp_degree(param_order, strict_ddp_flag)
    size_lcm = np.lcm.reduce(list(params_dict.keys()))
    config_dict: Dict[int, Dict] = dict()
    total_param_size = 0

    size_dict: Dict[int, List[int]] = dict()
    for dp_degree in params_dict:
        params_list = params_dict[dp_degree]
        size_list = [_tensor_numel(p, strict_ddp_flag) for p in params_list]
        group_acc_size = sum(size_list)
        total_param_size += group_acc_size

        # let small parameters keep gathered in CUDA all the time
        if group_acc_size < min_chunk_size:
            config_dict[dp_degree] = dict(chunk_size=group_acc_size, keep_gathered=True)
        else:
            size_dict[dp_degree] = size_list

    if filter_exlarge_params:
        _filter_exlarge_params(model, size_dict)

    max_size = min_chunk_size
    for key in size_dict:
        max_size = max(max_size, max(size_dict[key]))
    start_size = int(math.ceil(max_size / search_interval) * search_interval)

    min_chunk_waste = float('+inf')
    best_chunk_size = start_size

    for chunk_size in range(start_size, start_size + search_range + 1, search_interval):
        temp_waste = 0
        for key in size_dict:
            temp_waste += _get_unused_byte(size_dict[key], chunk_size)
        if temp_waste < min_chunk_waste:
            min_chunk_waste = temp_waste
            best_chunk_size = chunk_size

    # the chunk size needs to be divided by each groups sizes
    best_chunk_size = best_chunk_size + (-best_chunk_size % size_lcm)
    for dp_degree in params_dict:
        if dp_degree in config_dict:
            continue
        config_dict[dp_degree] = dict(chunk_size=best_chunk_size, keep_gathered=False)

    return config_dict, total_param_size, min_chunk_waste
