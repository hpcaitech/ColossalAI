from typing import List

import torch

__all__ = 'InferenceConfig'


class InferenceConfig:
    '''
    InferenceConfig is a class that stores the configuration for inference.

    Args:
        pp_size (int): the number of pipeline stages.
        stage_unit (List[str]): the unit module name can be sliced as a stage, should be `nn.Module`.
        target_length (int): the target length of the input sequence.
        padding_token_id (int): the token id for padding.

    '''

    def __init__(
        self,
        pp_size: int,
        stage_unit: List[str],
        target_length: int = 32,
        padding_token_id: int = 0,
    ):
        assert isinstance(pp_size, int), f'pp_size must be an integer, got {type(pp_size)}'
        self.pp_size = pp_size
        self.stage_unit = stage_unit
        self.target_length = target_length
        self.padding_token_id = padding_token_id
