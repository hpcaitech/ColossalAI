from typing import Dict, List, NamedTuple

import torch

from colossalai.elixir.chunk import ChunkGroup


class ChunkPlan(NamedTuple):
    """ChunkPlan is a type of configuration used to instruct the initialization of a chunk.

    args:
        name_list: contains the names of parameters that should be pushed into this chunk
        chunk_size: the size of this chunk
        chunk_dtype: the dtype of this chunk
        kwargs: a dictionary used in __init__ function of Chunk
    """
    name_list: List[str]
    chunk_size: int
    chunk_dtype: torch.dtype
    kwargs: Dict


class SearchResult(object):

    def __init__(self,
                 chunk_group: ChunkGroup,
                 chunk_plans: List[ChunkPlan],
                 param_called_per_step: List[List[str]] = None) -> None:
        super().__init__()
        self.chunk_group = chunk_group
        self.param_chunk_plans = chunk_plans
        self.param_called_per_step = param_called_per_step
