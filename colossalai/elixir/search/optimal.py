import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.autograd.profiler_util import _format_memory

from colossalai.elixir.cuda import get_allowed_memory, gpu_device
from colossalai.elixir.tracer.memory_tracer import cuda_memory_profiling
from colossalai.elixir.utils import calc_buffer_size, print_rank_0

from .base import SearchBase
from .result import SearchResult
from .simulator import find_optimal_chunk_size, rcache_prioirity_check
from .utils import find_search_range, get_multi_used_params, to_divide

dtype_to_es = {torch.float16: 2, torch.float32: 4, torch.float64: 8}


class SearchOptimal(SearchBase):
    """Search the best chunk size to maximize the training throughput.
    Users should provide the example input data and step function of training.

    args:
        module: the module to be searched
        default_group_size: the default group size of communications
        activation_fragment_factor: the factor to estimate the total activation memory usage
        allocation_fragment_factor: the factor to estimate the effective ratio of the memory usage can be used by Elixir
        driver_usage: the memory usage of the cuda driver
        dtype: the data type of the parameters
        verbose: whether to print the search details
        overlap: whether to overlap the communication and computation
        inp: a dictionary, the example input of the model
        step_fn: the example step function of training
    """

    def __init__(self,
                 module: nn.Module,
                 default_group_size: int,
                 activation_fragment_factor: float = 1.25,
                 allocation_fragment_factor: float = 0.95,
                 driver_usage: float = 2 * 1024**3,
                 dtype: torch.dtype = torch.float,
                 verbose: bool = False,
                 overlap: bool = False,
                 inp=None,
                 step_fn=None) -> None:
        # as for optimal search, we must profile the model first
        super().__init__(module, dtype, True, verbose, inp, step_fn)
        # profile cuda memory usage
        memo_usage = cuda_memory_profiling(model=self.meta_module, inp=inp, step_fn=step_fn, dtype=dtype)
        torch.cuda.empty_cache()
        buffer_occ = memo_usage['buffer_occ']
        # get the maximum memory usage of activation
        predict_activation = memo_usage['activation_occ']
        # calculate the total capacity of the current device
        gpu_memory = get_allowed_memory()
        # allowed capacity = allocation_fragment_factor * (total capacity - activation_fragment_factor * activation)
        self.cuda_capacity = int(
            allocation_fragment_factor *
            (gpu_memory - driver_usage - buffer_occ - activation_fragment_factor * predict_activation))
        hook_buffer_store_size = calc_buffer_size(m=self.meta_module, test_dtype=self.unified_dtype)
        self.cuda_elements = self.cuda_capacity // dtype_to_es.get(dtype) - hook_buffer_store_size

        if self.cuda_elements < 0:
            raise RuntimeError('optimal search: activation is too large, please reduce batch size')

        if self.verbose:
            print_rank_0('Predict memory usage:')
            for k, v in memo_usage.items():
                print_rank_0(f'{k}: {_format_memory(v)}')
            print_rank_0(f'allowed allocation space: {_format_memory(self.cuda_capacity)}')
            print_rank_0(f'hook buffer store size: {hook_buffer_store_size}')
            print_rank_0(f'allowed {dtype} elements: {self.cuda_elements}')

        self.default_group_size = default_group_size
        self.comm_overlap = overlap

    def private_truncate(self, param: nn.Parameter) -> int:
        return to_divide(param.numel(), self.default_group_size)

    def public_trucate(self, length: int) -> int:
        return to_divide(length, self.default_group_size)

    def search(self) -> Tuple:
        min_chunk_size, max_chunk_size, search_interval = find_search_range(self.meta_module)
        # get multi-used parameters
        private_params = get_multi_used_params(self.meta_module)
        # subtract the footprint of fused parameters
        for param in private_params:
            self.cuda_elements -= param.numel()
        if self.cuda_elements < 0:
            raise RuntimeError('optimal search: no enough space for fused parameters')

        # initialize public params in the called order
        public_params = list()
        public_param_set = set()
        name_to_param = {name: param for name, param in self.meta_module.named_parameters()}
        for name_set in self.param_per_step:
            for name in name_set:
                param = name_to_param.get(name)
                if param in private_params or param in public_param_set:
                    continue
                public_params.append(param)
                public_param_set.add(param)
        del name_to_param
        del public_param_set

        # collect the number of elements of each parameter
        public_numels = [p.numel() for p in public_params]
        # calculate the sumary of all parameters
        total_size = sum(public_numels)
        # collect the name for each public parameters
        public_param_names = [self.param_to_name[p] for p in public_params]

        if total_size <= min_chunk_size:
            public_block_size = total_size
            n_blocks = 1
            waste_size = 0
        else:
            public_block_size, n_blocks, waste_size = find_optimal_chunk_size(
            # pre-commit: do not rearrange
                param_per_step=self.param_per_step,
                param_names=public_param_names,
                param_numels=public_numels,
                cuda_elements=self.cuda_elements,
                overlap=self.comm_overlap,
                min_range=min_chunk_size,
                max_range=max_chunk_size,
                interval=search_interval)
        # truncate the size of public blocks
        public_block_size = self.public_trucate(public_block_size)
        if self.cuda_elements < n_blocks * public_block_size:
            raise RuntimeError('no enough space for unfused parameters')

        if self.verbose:
            if total_size == 0:
                waste_percentage = 0
            else:
                waste_percentage = 100 * waste_size / total_size
            print_rank_0(
                f'Optimal search result: chunk size = {public_block_size}, waste percentage = {waste_percentage: .1f} %'
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

        if total_size == 0:
            n_blocks = 0

        self.public_block_size = public_block_size
        self.public_block_number = n_blocks

        return (private_params, public_groups)

    def configure_rcache_size(self, chunk_plans: list, os_factor: int):
        element_os = 4
        if self.unified_dtype == torch.float16:
            element_pa = 2
        elif self.unified_dtype == torch.float:
            element_pa = 4
        else:
            raise NotImplementedError

        priority = rcache_prioirity_check(n=self.default_group_size, r_os=os_factor, e_p=element_pa, e_o=element_os)
        if self.verbose:
            print_rank_0(f'rCache Priority Check: {priority}')

        if not priority:
            n_cache_blocks = max(4, math.ceil(self.max_checkpoint_size / self.public_block_size) + 1)
            if self.comm_overlap:
                n_cache_blocks += 2
            n_cache_blocks = min(n_cache_blocks, self.public_block_number)
            self.cuda_elements -= n_cache_blocks * self.public_block_size
            if self.verbose:
                print_rank_0(f'n_cache_block is set to {n_cache_blocks}')
        else:
            self.cuda_elements -= self.public_block_number * self.public_block_size

        def try_move_chunk_to_cuda(fused: bool):
            for (i, plan) in enumerate(chunk_plans):
                rcache_fused = plan.kwargs.get('rcache_fused', False)
                if not fused and rcache_fused:
                    continue
                elif fused and not rcache_fused:
                    break
                param_os_size = os_factor * plan.chunk_size // self.default_group_size
                if self.cuda_elements >= param_os_size:
                    plan.kwargs['shard_device'] = gpu_device()
                    self.cuda_elements -= param_os_size
                else:
                    plan.kwargs['shard_device'] = torch.device('cpu')
                    plan.kwargs['cpu_pin_memory'] = True
                if self.verbose:
                    print_rank_0(f"chunk {i}: shard device -> {plan.kwargs['shard_device']}")

        # check chunks that are not fused on rCache
        try_move_chunk_to_cuda(False)
        # check chunks that are fused on rCache
        try_move_chunk_to_cuda(True)

        if not priority:
            extra_blocks = math.floor(self.cuda_elements / self.public_block_size)
            extra_blocks = min(extra_blocks, self.public_block_number - n_cache_blocks)
            self.cuda_elements -= extra_blocks * self.public_block_size
            self.public_block_number = n_cache_blocks + extra_blocks
            if self.verbose:
                print_rank_0(f'n_extra_blocks is set to {extra_blocks}')

        return chunk_plans


def optimal_search(
    # pre-commit: do not rearrange
        m: nn.Module,
        group_size: int,
        unified_dtype: torch.dtype = torch.float,
        optimizer_type: str = 'Adam',
        overlap: bool = False,
        verbose: bool = False,
        inp=None,
        step_fn=None) -> SearchResult:

    search_class = SearchOptimal(
    # pre-commit: do not rearrange
        module=m,
        default_group_size=group_size,
        dtype=unified_dtype,
        verbose=verbose,
        overlap=overlap,
        inp=inp,
        step_fn=step_fn)

    private_group, public_groups = search_class.search()
    chunk_plans = search_class.generate_chunk_plans(private_group, public_groups)

    if unified_dtype == torch.float16:
        master_weight_factor = 2
    elif unified_dtype == torch.float:
        master_weight_factor = 1
    else:
        raise NotImplementedError

    if optimizer_type == 'SGD':
        extra_sotre_factor = 1
    elif optimizer_type == 'Adam':
        extra_sotre_factor = 2
    else:
        raise NotImplementedError

    os_factor = 1 + (1 + extra_sotre_factor) * master_weight_factor
    chunk_plans = search_class.configure_rcache_size(chunk_plans, os_factor)
    chunk_group = search_class.allocate_chunk_group(chunk_plans)

    return SearchResult(chunk_group=chunk_group,
                        chunk_plans=chunk_plans,
                        param_called_per_step=search_class.param_per_step)
