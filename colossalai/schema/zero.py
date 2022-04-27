from pydantic import (BaseModel, StrictBool, StrictInt, StrictFloat, validator)
from typing import Any

__all__ = ['ZeroConfig']


class ZeroModelConfig(BaseModel):
    # TODO: resolve circular import and replace Any with BaseShardStrategy
    shard_strategy: Any
    reduce_scatter_bucket_size_mb: int = 25
    fp32_reduce_scatter: StrictBool = False
    offload_config: dict = dict(device="cpu")
    gradient_predivide_factor: StrictFloat = 1.0
    use_memory_tracer: StrictBool = False
    reuse_fp16_shard: StrictBool = False

    class Config:
        arbitrary_types_allowed = True

    @validator('offload_config')
    def check_offload_device(cls, v):
        assert 'device' in v and len(v) == 1, \
            f'offload_config should only have 1 filed which is device, but got extra unknown fields'
        assert v['device'] in ['cpu', 'gpu', 'auto'], \
            f"device can only be cpu, gpu, or auto, but got {v['device']}"


class ZeroOptimizerConfig(BaseModel):
    cpu_offload: StrictBool = False
    gpu_margin_mem_ratio: StrictFloat = 0.8
    initial_scale: StrictInt = 2**5
    min_scale: StrictInt = 1
    growth_factor: StrictInt = 2
    backoff_factor: StrictFloat = 0.5
    growth_interval: StrictInt = 1000
    hysteresis: StrictInt = 2
    max_scale: StrictInt = 2**32


class ZeroConfig(BaseModel):
    model_config: ZeroModelConfig = None
    optimizer_config: ZeroOptimizerConfig = None
