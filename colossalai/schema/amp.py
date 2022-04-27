from pydantic import BaseModel, StrictBool, StrictInt, StrictFloat, StrictStr
from typing import Any
import torch

__all__ = ['NaiveAMPConfig', 'TorchAMPConfig', 'ApexAMPConfig']


class NaiveAMPConfig(BaseModel):
    # TODO: resolve circular import and replace Any with AMP_TYPE
    mode: Any
    log_num_zeros_in_grad: StrictInt = False
    initial_scale: StrictInt = 2**32
    min_scale: StrictInt = 1
    growth_factor: StrictInt = 2
    backoff_factor: StrictFloat = 0.5
    growth_interval: StrictInt = 1000
    hysteresis: StrictInt = 2


class TorchAMPConfig(BaseModel):
    mode: Any
    init_scale: StrictInt = 2**16
    growth_factor: StrictFloat = 2.0
    backoff_factor: StrictFloat = 0.5
    growth_interval: StrictInt = 2000
    enabled: StrictBool = True


class ApexAMPConfig(BaseModel):
    mode: Any
    enabled: StrictBool = True
    opt_level: StrictStr = 'O1'
    cast_model_type: torch.dtype = None
    patch_torch_functions: bool = None
    keep_batchnorm_fp32: bool = None
    master_weights: bool = None
    loss_scale: float = None
    cast_model_outputs: bool = None
    num_losses: StrictInt = 1
    verbosity: StrictInt = 1
    min_loss_scale: int = None
    max_loss_scale: float = 16777216.0

    class Config:
        arbitrary_types_allowed = True
