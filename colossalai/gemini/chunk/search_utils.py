import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch.nn as nn

from colossalai.gemini.memory_tracer import MemStats, OrderedParamGenerator
from colossalai.tensor import ColoParameter
from colossalai.utils import is_ddp_ignored


def _filter_exlarge_params(model: nn.Module, size_dict: Dict[int, List[int]]) -> None:
    """
    Filter those parameters whose size is too large (more than 3x standard deviations) from others.
    """
    params_size = [p.numel() for p in model.parameters() if not is_ddp_ignored(p)]
    params_size_arr = np.array(params_size)

    std = np.std(params_size_arr)
    mean = np.mean(params_size_arr)
    upper_limit = mean + 3 * std

    for key in size_dict:
        org_list = size_dict[key]
        size_dict[key] = list(filter(lambda x: x <= upper_limit, org_list))


def _get_unused_byte(size_list: List[int], chunk_size: int) -> int:
    """Get unused byte for a certain chunk size.
    """
    acc = 0
    left = 0
    for s in size_list:
        if s > left:
            acc += left
            left = chunk_size
        left -= s
    return left + acc


def classify_params_by_dp_degree(param_order: OrderedParamGenerator) -> Dict[int, List[ColoParameter]]:
    """classify_params_by_dp_degree

    Classify the parameters by their dp degree

    Args:
        param_order (OrderedParamGenerator): the order of param be visied

    Returns:
        Dict[int, List[ColoParameter]]: a dict contains the classification results.
        The keys are dp_degrees and the values are parameters.
    """
    params_dict: Dict[int, List[ColoParameter]] = dict()
    for param in param_order.generate():
        assert isinstance(param, ColoParameter), "please init model in the ColoInitContext"
        if is_ddp_ignored(param):
            continue

        param_key = param.process_group.dp_world_size()

        if param_key not in params_dict:
            params_dict[param_key] = []
        params_dict[param_key].append(param)

    return params_dict


def search_chunk_configuration(
        model: nn.Module,
        search_range_mb: float,
        search_interval_byte: int,    # hidden size is the best value for the interval
        min_chunk_size_mb: float = 32,
        filter_exlarge_params: bool = True,
        memstas: Optional[MemStats] = None) -> Tuple[Dict, int]:
    """search_chunk_configuration

    Args:
        model (nn.Module): torch module
        search_range_mb (float): searching range in mega byte.
        search_interval_byte (int): searching interval in byte.
        filter_exlarge_params (bool, optional): filter extreme large parameters. Defaults to True.

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

    search_range_byte = round(search_range_mb * 1024**2)
    min_chunk_size_byte = round(min_chunk_size_mb * 1024**2)
    assert search_range_byte >= 0

    params_dict = classify_params_by_dp_degree(param_order)
    config_dict: Dict[int, Dict] = dict()

    size_dict: Dict[int, List[int]] = dict()
    for dp_degree in params_dict:
        params_list = params_dict[dp_degree]
        size_list = [p.numel() for p in params_list]
        # let small parameters keep gathered in CUDA all the time
        total_size = sum(size_list)
        if total_size < min_chunk_size_byte:
            config_dict[dp_degree] = dict(chunk_size=total_size, keep_gathered=True)
        else:
            size_dict[dp_degree] = size_list

    if filter_exlarge_params:
        _filter_exlarge_params(model, size_dict)

    max_size = min_chunk_size_byte
    for key in size_dict:
        max_size = max(max_size, max(size_dict[key]))
    start_size = int(math.ceil(max_size / search_interval_byte) * search_interval_byte)

    min_chunk_waste = float('+inf')
    best_chunk_size = start_size

    for chunk_size in range(start_size, start_size + search_range_byte + 1, search_interval_byte):
        temp_waste = 0
        for key in size_dict:
            temp_waste += _get_unused_byte(size_dict[key], chunk_size)
        if temp_waste < min_chunk_waste:
            min_chunk_waste = temp_waste
            best_chunk_size = chunk_size

    for dp_degree in params_dict:
        if dp_degree in config_dict:
            continue
        config_dict[dp_degree] = dict(chunk_size=best_chunk_size, keep_gathered=False)

    return config_dict, min_chunk_waste
