from types import MethodType
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from coati.models.base import get_base_model
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

import colossalai
from colossalai.booster.plugin.low_level_zero_plugin import (
    SUPPORTED_PRECISION,
    LowLevelZeroModel,
    LowLevelZeroOptimizer,
)
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.lazy import LazyInitContext

from .naive import NaiveStrategy
from .tp import tp_parallelize


class TPZeroStrategy(NaiveStrategy):

    def __init__(self,
                 tp_size: int,
                 zero_stage: int = 2,
                 precision: str = 'fp16',
                 initial_scale: float = 2**32,
                 min_scale: float = 1,
                 growth_factor: float = 2,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 1000,
                 hysteresis: int = 2,
                 max_scale: float = 2**32,
                 max_norm: float = 0.0,
                 norm_type: float = 2.0,
                 reduce_bucket_size_in_m: int = 12,
                 communication_dtype: Optional[torch.dtype] = None,
                 overlap_communication: bool = True,
                 cpu_offload: bool = False,
                 seed: int = 42) -> None:
        assert zero_stage in (1, 2), f'LowLevelZeroPlugin only supports stage 1/2 training'
        assert precision in SUPPORTED_PRECISION, f'LowLevelZeroPlugin only supports {SUPPORTED_PRECISION} training'
        self.tp_size = tp_size
        self.zero_stage = zero_stage
        self.precision = precision
        self.zero_optim_config = dict(reduce_bucket_size=reduce_bucket_size_in_m * 1024 * 1024,
                                      communication_dtype=communication_dtype,
                                      overlap_communication=overlap_communication,
                                      cpu_offload=cpu_offload)
        self.optim_kwargs = dict(initial_scale=initial_scale,
                                 growth_factor=growth_factor,
                                 backoff_factor=backoff_factor,
                                 growth_interval=growth_interval,
                                 hysteresis=hysteresis,
                                 min_scale=min_scale,
                                 max_scale=max_scale,
                                 max_norm=max_norm,
                                 norm_type=norm_type)
        self.seed = seed
        super().__init__()

    def setup_distributed(self) -> None:
        config = dict(parallel=dict(tensor=dict(
            mode='1d',
            size=self.tp_size,
        )))
        colossalai.launch_from_torch(config, seed=self.seed)
        self.zero_optim_config['zero_process_group'] = gpc.get_group(ParallelMode.DATA)
        self.zero_optim_config['tp_process_group'] = gpc.get_group(ParallelMode.PARALLEL_1D)

    def model_init_context(self):
        return LazyInitContext()

    def setup_model(self, model: torch.nn.Module) -> torch.nn.Module:
        tp_parallelize(model)
        model.to('cpu')
        model = LowLevelZeroModel(model, self.zero_stage, self.precision)
        model.to('cpu')
        return model

    def unwrap_model(self, model: LowLevelZeroModel) -> torch.nn.Module:
        assert isinstance(model, LowLevelZeroModel)
        return model.module

    def setup_optimizer(self, optimizer: Optimizer, model: LowLevelZeroModel) -> Optimizer:
        optim_ = LowLevelZeroOptimizer(model.unwrap(), optimizer, self.zero_optim_config, self.optim_kwargs)

        # Hack to/cuda/cpu of model

        def model_to(m: LowLevelZeroModel, device: str):
            for b in m.buffers():
                b.data = b.data.to(device)
            optim_.to(device)
            return m

        model.to = MethodType(model_to, model)
        model.cuda = MethodType(lambda m: model_to(m, 'cuda'), model)
        model.cpu = MethodType(lambda m: model_to(m, 'cpu'), model)
        return optim_

    def backward(self, loss: Tensor, model: Module, optimizer: Optimizer, **kwargs) -> None:
        optimizer.backward(loss)

    def optimizer_step(self, optimizer: Optimizer, **kwargs) -> None:
        optimizer.step()

    def save_model(self, model: Module, path: str, only_rank0: bool = True) -> None:
        if gpc.get_local_rank(ParallelMode.DATA) == 0:
            path = f'{path}.tr{gpc.get_local_rank(ParallelMode.PARALLEL_1D)}'
            state_dict = self.unwrap_model(model).state_dict()
            torch.save(state_dict, path)

    def load_model(self, model: Module, path: str, map_location: Any = None, strict: bool = True) -> None:
        path = f'{path}.tr{gpc.get_local_rank(ParallelMode.PARALLEL_1D)}'
        state_dict = torch.load(path, map_location=map_location)
        self.unwrap_model(model).load_state_dict(state_dict, strict=strict)

    def save_optimizer(self, optimizer: Optimizer, path: str, only_rank0: bool = False) -> None:
        path = f'{path}.r{gpc.get_global_rank()}'
        super().save_optimizer(optimizer, path, only_rank0)

    def load_optimizer(self, optimizer: Optimizer, path: str, map_location: Any = None) -> None:
        path = f'{path}.r{gpc.get_global_rank()}'
        super().load_optimizer(optimizer, path, map_location)
