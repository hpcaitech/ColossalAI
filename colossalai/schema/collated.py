from pydantic import BaseModel, validator, StrictInt, StrictBool
from .zero import ZeroConfig
from .parallel import ParallelConfig
from .amp import ApexAMPConfig, NaiveAMPConfig, TorchAMPConfig
from typing import Union, List, Dict

__all__ = ['ColossalaiConfig']


class ColossalaiConfig(BaseModel):
    gradient_accumulation: StrictInt = 1
    clip_grad_norm: float = 1.0
    gradient_handler: List[Dict] = None
    cudnn_benchmark: StrictBool = True
    cudnn_deterministic: StrictBool = False
    amp: Union[ApexAMPConfig, NaiveAMPConfig, TorchAMPConfig] = None
    zero: ZeroConfig = None
    parallel: ParallelConfig = None

    @validator('clip_grad_norm')
    def check_grad_clip_norm(cls, v):
        assert v > 0, f'Gradient clipping max norm should be larger than 0, but got {v}'

    @validator('gradient_accumulation')
    def check_accum_size(cls, v):
        assert v > 0, f'graident accumulation size should be larger than 0, but got {v}'

    class Config:
        extra = 'ignore'
