from pydantic import BaseModel, validator
from typing import Dict, Union

__all__ = ['ParallelConfig']


class ParallelConfig(BaseModel):
    pipeline: Union[Dict, int] = dict(size=1)
    tensor: Dict = dict(size=1, mode=None)

    @validator('tensor')
    def check_tensor_config(cls, v):
        assert 'size' in v, f'the field size is missing in the tensor parallel configuration'
        if 'mode' in v and v['mode'] == '2.5d':
            assert 'depth' in v, f'the field depth is missing in 2.5D tensor parallel configuration'

    @validator('pipeline')
    def check_pipeline_config(cls, v):
        if isinstance(v, dict):
            assert 'size' in v, f'the field size is missing in the pipeline parallel configuration'
            assert v['size'] > 0, f"the pipeline size should be larger then 0, but got {v['size']}"
        elif isinstance(v, int):
            assert v > 0, f"the pipeline size should be larger then 0, but got {v['size']}"
