from dataclasses import dataclass
from typing import List, Literal

__all__ = ['ShardConfig']


@dataclass
class ShardConfig:
    r"""
    The config for sharding the huggingface model

    Args:
        data_parallel_size (int): The size of data parallel
        tensor_parallel_size (int): The size of tensor parallel
        pipeline_parallel_size (int): The size of pipeline parallel
        tensor_parallel_mode (List): The mode of tensor parallel, choose from `['1d','2d','2.5d','3d']
        inference_only (bool): Whether to use the inference only mode, when setting to `True`, the model
            will not calculate the loss and just return the output.
        gather_output (bool): Whether to gather the output of the model of the last layer
    """
    tensor_parallel_size: int
    # TODO: add support for tensor parallel
    # pipeline_parallel_size: int
    # data_parallel_size: int
    tensor_parallel_mode: Literal['1d', '2d', '2.5d', '3d']
    inference_only: bool = True
    gather_output: bool = True
