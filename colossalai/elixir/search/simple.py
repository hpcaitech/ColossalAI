import math
from typing import Tuple

import torch
import torch.nn as nn

from .base import SearchBase
from .result import SearchResult
from .utils import get_multi_used_params, to_divide


class SearchSimple(SearchBase):

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

    def search(self, split_number: int, allocate_factor: float) -> Tuple:
        # get multi-used parameters
        private_params = get_multi_used_params(self.meta_module)
        # get parameters used only one time
        public_params = [p for p in self.meta_module.parameters() if p not in private_params]

        # calculate the size of each group
        len_public = len(public_params)
        split_number = min(len_public, split_number)
        # allocate a list for groups
        public_groups = list()
        if split_number > 0:
            average_size = len_public // split_number
            left_size = len_public % split_number

            # set the size of each segment
            pack_size_list = [average_size] * split_number
            for i in range(split_number):
                if left_size > 0:
                    pack_size_list[i] += 1
                left_size -= 1

            # split public parameters
            for i in range(split_number):
                p_list = list()
                for _ in range(pack_size_list[i]):
                    p = public_params.pop(0)
                    p_list.append(p)
                public_groups.append(p_list)
            assert len(public_params) == 0

            # calculate the maximum summarized size
            max_sum_size = 0
            for p_list in public_groups:
                sum_size = sum([p.numel() for p in p_list])
                max_sum_size = max(max_sum_size, sum_size)
        else:
            max_sum_size = 0

        self.public_block_size = max_sum_size
        self.public_block_number = math.ceil(split_number * allocate_factor)

        return (private_params, public_groups)


def simple_search(m: nn.Module,
                  group_size: int,
                  split_number: int = 10,
                  allocate_factor: float = 0.6,
                  unified_dtype: torch.dtype = torch.float,
                  shard_device: torch.device = torch.device('cpu'),
                  prefetch: bool = False,
                  verbose: bool = False,
                  inp=None,
                  step_fn=None) -> SearchResult:

    search_class = SearchSimple(
    # pre-commit: do not rearrange
        module=m,
        default_group_size=group_size,
        dtype=unified_dtype,
        prefetch=prefetch,
        verbose=verbose,
        inp=inp,
        step_fn=step_fn)

    private_group, public_groups = search_class.search(split_number, allocate_factor)
    chunk_plans = search_class.generate_chunk_plans(private_group, public_groups)
    # assign shard device
    for plan in chunk_plans:
        plan.kwargs['shard_device'] = shard_device

    chunk_group = search_class.allocate_chunk_group(chunk_plans)

    return SearchResult(chunk_group=chunk_group,
                        chunk_plans=chunk_plans,
                        param_called_per_step=search_class.param_per_step)
