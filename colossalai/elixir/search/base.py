from abc import ABC, abstractmethod
from functools import partial
from typing import List, Tuple

import torch
import torch.nn as nn

from colossalai.elixir.chunk import BlockSpec, ChunkGroup, MemoryPool
from colossalai.elixir.tracer.param_tracer import generate_tf_order
from colossalai.elixir.tracer.utils import meta_copy
from colossalai.elixir.utils import print_rank_0

from .result import ChunkPlan
from .utils import to_meta_tensor


class SearchBase(ABC):
    """A basic class for search algorithms.

    args:
        module: the model to be searched
        dtype: the unified dtype of all parameters
        prefetch: whether to prefetch chunks during training
        verbose: whether to print search details
        inp: a dictionary, the example input of the model
        step_fn: the example step function of the model
    """

    def __init__(self,
                 module: nn.Module,
                 dtype: torch.dtype = torch.float,
                 prefetch: bool = False,
                 verbose: bool = False,
                 inp=None,
                 step_fn=None) -> None:

        self.unified_dtype = dtype
        self.meta_module = meta_copy(module, partial(to_meta_tensor, dtype=self.unified_dtype))
        self.prefetch_flag = prefetch
        self.verbose = verbose
        self.param_to_name = {param: name for name, param in self.meta_module.named_parameters()}

        self.public_block_size = 1024
        self.public_block_number = 0

        self.param_per_step = None
        self.max_checkpoint_size = 0
        if self.prefetch_flag:
            assert inp is not None and step_fn is not None
            tf_running_info = generate_tf_order(self.meta_module, inp, step_fn, dtype)
            self.param_per_step = tf_running_info.get('params_per_step')
            if self.verbose:
                print_rank_0('Prefetch enabled: the called order of parameters')
                for i, step in enumerate(self.param_per_step):
                    print_rank_0(f'step {i}: {step}')

            name_to_param = {name: param for name, param in self.meta_module.named_parameters()}
            for checkpoint in tf_running_info.get('checkpoint_info'):
                sum_numel = 0
                for i in range(*checkpoint):
                    for name in self.param_per_step[i]:
                        param = name_to_param[name]
                        sum_numel += param.numel()
                self.max_checkpoint_size = max(self.max_checkpoint_size, sum_numel)
                if self.verbose:
                    print_rank_0(f'checkpoint infomation: from-to -> {checkpoint}, numel -> {sum_numel}')

    @abstractmethod
    def private_truncate(self, param: nn.Parameter) -> int:
        """A function used to truncate the length of a private chunk,
        which only contains one parameter.
        """
        pass

    @abstractmethod
    def public_trucate(self, length: int) -> int:
        """A function used to trucate the length of all publick chunks
        """
        pass

    @abstractmethod
    def search(self, *args, **kwargs) -> Tuple:
        """The core search function. It returns a tuple of a private group and public groups.
        """
        pass

    def generate_chunk_plans(self, private_group, publick_groups) -> List[ChunkPlan]:
        plans = list()
        for param in private_group:
            chunk_size = self.private_truncate(param)
            chunk_dtype = param.dtype
            chunk_kwargs = dict(rcache_fused=True)
            chunk_plan = ChunkPlan(name_list=[self.param_to_name[param]],
                                   chunk_size=chunk_size,
                                   chunk_dtype=chunk_dtype,
                                   kwargs=chunk_kwargs)
            plans.append(chunk_plan)

        self.public_block_size = self.public_trucate(self.public_block_size)
        public_chunk_size = self.public_block_size
        public_chunk_dtype = self.unified_dtype
        for group in publick_groups:
            chunk_kwargs = {}
            chunk_plan = ChunkPlan(name_list=[self.param_to_name[p] for p in group],
                                   chunk_size=public_chunk_size,
                                   chunk_dtype=public_chunk_dtype,
                                   kwargs=chunk_kwargs)
            plans.append(chunk_plan)

        if self.verbose:
            print_rank_0(f'Chunk plans: total {len(plans)} chunks')
            for i, plan in enumerate(plans):
                print_rank_0(f'plan {i}: {plan}')

        return plans

    def allocate_chunk_group(self, chunk_plans: List[ChunkPlan]) -> ChunkGroup:
        block_require_list = list()
        for plan in chunk_plans:
            kwargs = plan.kwargs
            if kwargs.get('rcache_fused', False):
                block_require_list.append(BlockSpec(plan.chunk_size, plan.chunk_dtype))

        mp = MemoryPool('cuda')
        mp.allocate(public_dtype=self.unified_dtype,
                    public_block_size=self.public_block_size,
                    public_block_number=self.public_block_number,
                    private_block_list=block_require_list)

        if self.verbose:
            print_rank_0(
                f'Memory pool (rcache): {mp}\n\tblock size -> {mp.public_block_size}, block number -> {mp.public_free_cnt}'
            )

        return ChunkGroup(mp)
