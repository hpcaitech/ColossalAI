import math
from typing import Dict, List, Tuple

import numpy as np
import torch.nn as nn

from colossalai.tensor import ColoParameter


def in_ddp(param: nn.Parameter) -> bool:
    return not getattr(param, '_ddp_to_ignore', False)


def _filter_exlarge_params(model: nn.Module, size_dict: Dict[int, List[int]]) -> None:
    """Filter those parameters whose size is too large from others.
    """
    params_size = [p.numel() for p in model.parameters() if in_ddp(p)]
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


def clasify_params(model: nn.Module) -> Dict[int, List[ColoParameter]]:
    """Clasify each parameter by its size of DP group.
    """
    params_dict: Dict[int, List[ColoParameter]] = dict()
    for param in model.parameters():
        assert isinstance(param, ColoParameter), "please init model in the ColoInitContext"
        if not in_ddp(param):
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
        filter_exlarge_params: bool = True) -> Tuple[Dict, int]:
    search_range_byte = round(search_range_mb * 1024**2)
    min_chunk_size_byte = round(min_chunk_size_mb * 1024**2)
    assert search_range_byte >= 0

    params_dict = clasify_params(model)
    config_dict: Dict[int, Dict] = dict()

    size_dict: Dict[int, List[int]] = dict()
    for key in params_dict:
        params_list = params_dict[key]
        size_list = [p.numel() for p in params_list]
        # let small parameters keep gathered in CUDA all the time
        total_size = sum(size_list)
        if total_size < min_chunk_size_byte:
            config_dict[key] = dict(chunk_size=total_size, keep_gathered=True)
        else:
            size_dict[key] = size_list

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

    for key in params_dict:
        if key in config_dict:
            continue
        config_dict[key] = dict(chunk_size=best_chunk_size, keep_gathered=False)

    return config_dict, min_chunk_waste
