from dataclasses import dataclass
from typing import List

import torch

__all__ = 'InferenceConfig'


class InferenceConfig:
    '''
    InferenceConfig is a class that stores the configuration for inference.

    Args:
        pp_size (int): the number of pipeline stages.
        micro_batch_size (int): the micro batch size.
        new_length (int): the new length of the input sequence.
        padding_token_id (int): the token id for padding.

    '''

    def __init__(
        self,
        pp_size: int,
        micro_batch_size: int = 1,
        micro_batch_buffer_size: int = None,
        new_length: int = 32,
        padding_token_id: int = 0,
    ):
        assert isinstance(pp_size, int), f'pp_size must be an integer, got {type(pp_size)}'
        self.pp_size = pp_size
        self.micro_batch_size = micro_batch_size
        self.micro_batch_buffer_size = pp_size if micro_batch_buffer_size is None else micro_batch_buffer_size
        self.new_length = new_length
        self.padding_token_id = padding_token_id
