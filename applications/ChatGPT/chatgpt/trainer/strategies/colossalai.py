import warnings
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

import colossalai
from colossalai.nn.optimizer import CPUAdam, HybridAdam
from colossalai.nn.parallel import zero_model_wrapper, zero_optim_wrapper
from colossalai.tensor import ProcessGroup, ShardSpec
from colossalai.utils import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext

from .ddp import DDPStrategy


class ColossalAIStrategy(DDPStrategy):
    """
        The strategy for training with ColossalAI.

    Args:
        stage(int): The stage to use in ZeRO. Choose in (1, 2, 3)
        seed(int): The seed for the random number generator.
        shard_init(bool): Whether to shard the model parameters during initialization. Only for ZeRO-3.
            This is not compativle with `from_pretrained()`. We temporarily disable this and will support it in the future.
        placement_policy(str): The placement policy for gemini. Choose in ('cpu', 'cuda')
                          If it is “cpu”, parameters, gradients and optimizer states will be offloaded to CPU,
                          If it is “cuda”, they will not be offloaded, which means max CUDA memory will be used. It is the fastest.
        pin_memory(bool): Whether to pin the memory for the data loader. Only for ZeRO-3.
        force_outputs_fp32(bool): Whether to force the outputs to be fp32. Only for ZeRO-3.
        search_range_mb(int): The search range in MB for the chunk size. Only for ZeRO-3.
        hidden_dim(optional, int): The hidden dimension for the gemini. Only for ZeRO-3.
        min_chunk_size_mb(float): The minimum chunk size in MB. Only for ZeRO-3.
        gpu_margin_mem_ratio(float): The margin memory ratio for the GPU. Only for ZeRO-3.
        reduce_bugket_size(int): The reduce bucket size in bytes. Only for ZeRO-1 and ZeRO-2.
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

    def __init__(
            self,
            stage: int = 3,
            seed: int = 42,
            shard_init: bool = False,    # only for stage 3
            placement_policy: str = 'cuda',
            pin_memory: bool = True,    # only for stage 3
            force_outputs_fp32: bool = False,    # only for stage 3
            search_range_mb: int = 32,    # only for stage 3
            hidden_dim: Optional[int] = None,    # only for stage 3
            min_chunk_size_mb: float = 32,    # only for stage 3
            gpu_margin_mem_ratio: float = 0.0,    # only for stage 3
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
            norm_type: float = 2.0) -> None:
        super().__init__(seed)
        assert placement_policy in ('cpu', 'cuda'), f'Unsupported placement policy "{placement_policy}"'
        self.stage = stage
        # TODO(ver217): support shard_init when using from_pretrained()
        if shard_init:
            warnings.warn(f'Shard init is not supported yet. Ignore.')
            shard_init = False
        self.shard_init = shard_init
        self.gemini_config = dict(device=get_current_device(),
                                  placement_policy=placement_policy,
                                  pin_memory=pin_memory,
                                  force_outputs_fp32=force_outputs_fp32,
                                  strict_ddp_mode=shard_init,
                                  search_range_mb=search_range_mb,
                                  hidden_dim=hidden_dim,
                                  min_chunk_size_mb=min_chunk_size_mb)
        if stage == 3:
            self.zero_optim_config = dict(gpu_margin_mem_ratio=gpu_margin_mem_ratio)
        else:
            self.zero_optim_config = dict(reduce_bucket_size=reduce_bucket_size,
                                          overlap_communication=overlap_communication,
                                          cpu_offload=(placement_policy == 'cpu'))
        self.optim_kwargs = dict(initial_scale=initial_scale,
                                 growth_factor=growth_factor,
                                 backoff_factor=backoff_factor,
                                 growth_interval=growth_interval,
                                 hysteresis=hysteresis,
                                 min_scale=min_scale,
                                 max_scale=max_scale,
                                 max_norm=max_norm,
                                 norm_type=norm_type)

    def setup_distributed(self) -> None:
        colossalai.launch_from_torch({}, seed=self.seed)

    def model_init_context(self):
        if self.stage == 3:
            world_size = dist.get_world_size()
            shard_pg = ProcessGroup(tp_degree=world_size) if self.shard_init else None
            default_dist_spec = ShardSpec([-1], [world_size]) if self.shard_init else None
            return ColoInitContext(device=get_current_device(),
                                   dtype=torch.half,
                                   default_pg=shard_pg,
                                   default_dist_spec=default_dist_spec)
        return super().model_init_context()

    def setup_model(self, model: nn.Module) -> nn.Module:
        return zero_model_wrapper(model, zero_stage=self.stage, gemini_config=self.gemini_config)

    def setup_optimizer(self, optimizer: optim.Optimizer, model: nn.Module) -> optim.Optimizer:
        assert isinstance(optimizer, (CPUAdam, HybridAdam)), f'Unsupported optimizer {type(optimizer)}'
        return zero_optim_wrapper(model, optimizer, optim_config=self.zero_optim_config, **self.optim_kwargs)

    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        optimizer.backward(loss)

    def optimizer_step(self, optimizer: optim.Optimizer, **kwargs) -> None:
        optimizer.step()
