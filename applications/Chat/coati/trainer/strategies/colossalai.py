import warnings
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import colossalai
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin
from colossalai.booster.plugin.gemini_plugin import GeminiModel
from colossalai.booster.plugin.low_level_zero_plugin import LowLevelZeroModel
from colossalai.tensor import ProcessGroup, ShardSpec
from colossalai.utils import get_current_device
from colossalai.zero import ColoInitContext
from colossalai.zero.gemini.gemini_ddp import GeminiDDP

from .ddp import DDPStrategy


class LowLevelZeroStrategy(DDPStrategy):
    """
        The strategy for training with ColossalAI.

    Args:
        stage(int): The stage to use in ZeRO. Choose in (1, 2)
        precision(str): The precision to use. Choose in ('fp32', 'fp16').
        seed(int): The seed for the random number generator.
        placement_policy(str): The placement policy for gemini. Choose in ('cpu', 'cuda')
                          If it is “cpu”, parameters, gradients and optimizer states will be offloaded to CPU,
                          If it is “cuda”, they will not be offloaded, which means max CUDA memory will be used. It is the fastest.
        reduce_bucket_size(int): The reduce bucket size in bytes. Only for ZeRO-1 and ZeRO-2.
        overlap_communication(bool): Whether to overlap communication and computation. Only for ZeRO-1 and ZeRO-2.
        initial_scale(float): The initial scale for the optimizer.
        growth_factor(float): The growth factor for the optimizer.
        backoff_factor(float): The backoff factor for the optimizer.
        growth_interval(int): The growth interval for the optimizer.
        hysteresis(int): The hysteresis for the optimizer.
        min_scale(float): The minimum scale for the optimizer.
        max_scale(float): The maximum scale for the optimizer.
        max_norm(float): The maximum norm for the optimizer.
        norm_type(float): The norm type for the optimizer.

    """

    def __init__(self,
                 stage: int = 3,
                 precision: str = 'fp16',
                 seed: int = 42,
                 placement_policy: str = 'cuda',
                 reduce_bucket_size: int = 12 * 1024**2,    # only for stage 1&2
                 overlap_communication: bool = True,    # only for stage 1&2
                 initial_scale: float = 2**16,
                 growth_factor: float = 2,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 1000,
                 hysteresis: int = 2,
                 min_scale: float = 1,
                 max_scale: float = 2**32,
                 max_norm: float = 0.0,
                 norm_type: float = 2.0
                 ) -> None:

        assert stage in (1, 2), f'Unsupported stage "{stage}"'
        assert placement_policy in ('cpu', 'cuda'), f'Unsupported placement policy "{placement_policy}"'
        assert precision in ('fp32', 'fp16'), f'Unsupported precision "{precision}"'

        plugin_initializer = lambda: LowLevelZeroPlugin(
            # zero_config
            stage=stage,
            precision=precision,
            # zero_optim_config
            reduce_bucket_size_in_m=reduce_bucket_size,
            overlap_communication=overlap_communication,
            cpu_offload=(placement_policy == 'cpu'),
            # optim_config
            initial_scale=initial_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            hysteresis=hysteresis,
            min_scale=min_scale,
            max_scale=max_scale,
            max_norm=max_norm,
            norm_type=norm_type
        )

        super().__init__(seed, plugin_initializer)

    def _post_init(self) -> None:
        assert isinstance(self.plugin, LowLevelZeroPlugin), \
            f'{type(self).__name__}\'s plugin is not initialized properly.'

    def setup_distributed(self) -> None:
        colossalai.launch_from_torch({}, seed=self.seed)

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        assert isinstance(model, LowLevelZeroModel)
        return model.module

    def get_model_state_dict_shard(self, model: nn.Module, **config):
        assert isinstance(model, LowLevelZeroModel)
        yield from model.state_dict_shard(max_shard_size=1024, only_rank_0=False)


class GeminiStrategy(DDPStrategy):
    """
        The strategy for training with ColossalAI.

    Args:
        seed(int): The seed for the random number generator.
        shard_init(bool): Whether to shard the model parameters during initialization. Only for ZeRO-3.
            This is not compatible with `from_pretrained()`. We temporarily disable this and will support it in the future.
        placement_policy(str): The placement policy for gemini. Choose in ('cpu', 'cuda')
                          If it is “cpu”, parameters, gradients and optimizer states will be offloaded to CPU,
                          If it is “cuda”, they will not be offloaded, which means max CUDA memory will be used. It is the fastest.
        pin_memory(bool): Whether to pin the memory for the data loader. Only for ZeRO-3.
        force_outputs_fp32(bool): Whether to force the outputs to be fp32. Only for ZeRO-3.
        search_range_m(int): The number of search range for the chunk size, divided by 2^20. Only for ZeRO-3.
        hidden_dim(optional, int): The hidden dimension for the gemini. Only for ZeRO-3.
        min_chunk_size_m(float): The minimum chunk size divided by 2^20. Only for ZeRO-3.
        gpu_margin_mem_ratio(float): The margin memory ratio for the GPU. Only for ZeRO-3.
        initial_scale(float): The initial scale for the optimizer.
        growth_factor(float): The growth factor for the optimizer.
        backoff_factor(float): The backoff factor for the optimizer.
        growth_interval(int): The growth interval for the optimizer.
        hysteresis(int): The hysteresis for the optimizer.
        min_scale(float): The minimum scale for the optimizer.
        max_scale(float): The maximum scale for the optimizer.
        max_norm(float): The maximum norm for the optimizer.
        norm_type(float): The norm type for the optimizer.

    """

    def __init__(self,
                 seed: int = 42,
                 shard_init: bool = False,    # only for stage 3
                 placement_policy: str = 'cuda',
                 pin_memory: bool = True,    # only for stage 3
                 force_outputs_fp32: bool = False,    # only for stage 3
                 search_range_m: int = 32,    # only for stage 3
                 hidden_dim: Optional[int] = None,    # only for stage 3
                 min_chunk_size_m: float = 32,    # only for stage 3
                 gpu_margin_mem_ratio: float = 0.0,    # only for stage 3
                 initial_scale: float = 2**16,
                 growth_factor: float = 2,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 1000,
                 hysteresis: int = 2,
                 min_scale: float = 1,
                 max_scale: float = 2**32,
                 max_norm: float = 0.0,
                 norm_type: float = 2.0
                 ) -> None:

        assert placement_policy in ('cpu', 'cuda'), f'Unsupported placement policy "{placement_policy}"'

        # TODO(ver217): support shard_init when using from_pretrained()
        if shard_init:
            warnings.warn(
                f'Shard init is not supported model.from_pretrained() yet. '
                'Please load weights after strategy.prepare()'
            )
        self.shard_init = shard_init

        warnings.warn(f'Stage 3 only supports fp16. Precision is set to fp16.')

        # NOTE: dist should be initialized before calling get_current_device()
        plugin_initializer = lambda: GeminiPlugin(
            # gemini_config
            device=get_current_device(),
            placement_policy=placement_policy,
            precision='fp16',
            pin_memory=pin_memory,
            force_outputs_fp32=force_outputs_fp32,
            strict_ddp_mode=shard_init,
            search_range_m=search_range_m,
            hidden_dim=hidden_dim,
            min_chunk_size_m=min_chunk_size_m,
            # zero_optim_config
            gpu_margin_mem_ratio=gpu_margin_mem_ratio,
            # optim_config
            initial_scale=initial_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            hysteresis=hysteresis,
            min_scale=min_scale,
            max_scale=max_scale,
            max_norm=max_norm,
            norm_type=norm_type
        )

        super().__init__(seed, plugin_initializer)

    def _post_init(self) -> None:
        assert isinstance(self.plugin, GeminiPlugin), \
            f'{type(self).__name__}\'s plugin is not initialized properly.'

    def setup_distributed(self) -> None:
        colossalai.launch_from_torch({}, seed=self.seed)

    def model_init_context(self):
        world_size = dist.get_world_size()
        shard_pg = ProcessGroup(tp_degree=world_size) if self.shard_init else None
        default_dist_spec = ShardSpec([-1], [world_size]) if self.shard_init else None
        return ColoInitContext(device=get_current_device(),
                               dtype=torch.half,
                               default_pg=shard_pg,
                               default_dist_spec=default_dist_spec)

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        assert isinstance(model, GeminiModel)
        ddp_model = model.unwrap()
        assert isinstance(ddp_model, GeminiDDP)
        return ddp_model.module

    def save_pretrained(self,
                        model: nn.Module,
                        path: str,
                        only_rank0: bool = True,
                        tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        raise RuntimeError('ColossalAI strategy with stage-3 does not support save_pretrained() now')

    def get_model_state_dict_shard(self, model: nn.Module, **config):
        assert isinstance(self.plugin, GeminiPlugin)
        yield from super().get_model_state_dict_shard(model, **config)
