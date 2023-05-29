import math
from typing import Tuple

import torch
import torch.nn as nn

from colossalai.elixir.cuda import gpu_device
from colossalai.elixir.utils import print_rank_0

from .base import SearchBase
from .result import SearchResult
from .utils import find_minimum_waste_size, find_search_range, get_multi_used_params, to_divide

dtype_to_es = {torch.float16: 2, torch.float32: 4, torch.float64: 8}


class SearchMiniWaste(SearchBase):
    """Search the best chunk size to minimize the waste of memory.

    args:
        module: the module to be searched
        default_group_size: the default group size of communications
        dtype: the data type of the parameters
        prefetch: whether to prefetch the parameters
        verbose: whether to print the search details
        inp: a dictionary, the example input of the model
        step_fn: the example step function of training
    """

    def __init__(self,
                 module: nn.Module,
                 default_group_size: int,
                 dtype: torch.dtype = torch.float,
                 prefetch: bool = False,
                 verbose: bool = False,
                 inp=None,
                 step_fn=None) -> None:

        super().__init__(module, dtype, prefetch, verbose, inp, step_fn)
        self.default_group_size = default_group_size

    def private_truncate(self, param: nn.Parameter) -> int:
        return to_divide(param.numel(), self.default_group_size)

    def public_trucate(self, length: int) -> int:
        return to_divide(length, self.default_group_size)

    def search(self) -> Tuple:
        min_chunk_size, max_chunk_size, search_interval = find_search_range(self.meta_module)
        # get multi-used parameters
        private_params = get_multi_used_params(self.meta_module)
        # get parameters used only one time
        public_params = [p for p in self.meta_module.parameters() if p not in private_params]
        # collect the number of elements of each parameter
        public_numels = [p.numel() for p in public_params]
        # calculate the sumary of all parameters
        total_size = sum(public_numels)

        if total_size <= min_chunk_size:
            public_block_size = total_size
            waste_size = 0
        else:
            public_block_size, waste_size = find_minimum_waste_size(
            # pre-commit: do not rearrange
                numel_group_list=[public_numels],
                min_range=min_chunk_size,
                max_range=max_chunk_size,
                interval=search_interval)

        if self.verbose:
            if total_size == 0:
                waste_percentage = 0
            else:
                waste_percentage = 100 * waste_size / total_size
            print_rank_0(
                f'Minimum waste search result: chunk size = {public_block_size}, waste percentage = {waste_percentage: .1f} %'
            )

        # initialize the mapping from parameters to chunks
        param_to_chunk_id = dict()
        chunk_id = 0
        # deal with private parameters
        for p in private_params:
            param_to_chunk_id[p] = chunk_id
            chunk_id += 1
        # record the upper bound
        private_id_upperbound = chunk_id
        # deal with public parameters
        last_left = 0
        for p in public_params:
            p_size = p.numel()

            if last_left < p_size:
                last_left = public_block_size
                chunk_id += 1

            assert last_left >= p_size

            last_left -= p_size
            param_to_chunk_id[p] = chunk_id

        # initailize public groups
        public_number_chunks = chunk_id - private_id_upperbound
        public_groups = [[] for _ in range(public_number_chunks)]
        for p in public_params:
            public_chunk_id = param_to_chunk_id[p] - private_id_upperbound - 1
            public_groups[public_chunk_id].append(p)

        # calculate the number of minimum chunks allocated in R cache
        max_lived_chunks = 0
        for module in self.meta_module.modules():
            param_set = set()
            for param in module.parameters(recurse=False):
                param_set.add(param_to_chunk_id[param])
            max_lived_chunks = max(max_lived_chunks, len(param_set))
        # allocate more chunks for prefetch
        if self.prefetch_flag:
            max_lived_chunks = min(max_lived_chunks + 4, public_number_chunks)

        if total_size == 0:
            max_lived_chunks = 0

        self.public_block_size = public_block_size
        self.public_block_number = max_lived_chunks

        return (private_params, public_groups)


def minimum_waste_search(m: nn.Module,
                         group_size: int,
                         unified_dtype: torch.dtype = torch.float,
                         cpu_offload: bool = False,
                         prefetch: bool = False,
                         verbose: bool = False,
                         pin_memory: bool = True,
                         inp=None,
                         step_fn=None) -> SearchResult:

    search_class = SearchMiniWaste(
    # pre-commit: do not rearrange
        module=m,
        default_group_size=group_size,
        dtype=unified_dtype,
        prefetch=prefetch,
        verbose=verbose,
        inp=inp,
        step_fn=step_fn)

    private_group, public_groups = search_class.search()
    chunk_plans = search_class.generate_chunk_plans(private_group, public_groups)

    # assign shard device
    if cpu_offload:
        shard_device = torch.device('cpu')
    else:
        shard_device = gpu_device()

    for plan in chunk_plans:
        plan.kwargs['shard_device'] = shard_device
        if cpu_offload:
            plan.kwargs['cpu_pin_memory'] = pin_memory

    chunk_group = search_class.allocate_chunk_group(chunk_plans)

    return SearchResult(chunk_group=chunk_group,
                        chunk_plans=chunk_plans,
                        param_called_per_step=search_class.param_per_step)
