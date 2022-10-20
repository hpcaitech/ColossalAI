from dataclasses import dataclass
from typing import List, Optional, Set, Dict


@dataclass
class ParamDistMeta:
    # parallel info
    dp_rank: int
    dp_world_size: int
    tp_rank: int
    tp_world_size: int
    # tp info
    tp_shard_dims: Optional[List[int]] = None
    tp_num_parts: Optional[List[int]] = None
    # zero info
    zero_numel: Optional[int] = None
    zero_orig_shape: Optional[List[int]] = None

    @property
    def used_tp(self) -> bool:
        return self.tp_shard_dims is not None and self.tp_num_parts is not None

    @property
    def used_zero(self) -> bool:
        return self.zero_numel is not None and self.zero_orig_shape is not None

    @property
    def parallel_meta(self) -> tuple:
        return self.dp_rank, self.dp_world_size, self.tp_rank, self.tp_world_size

    @property
    def tp_meta(self) -> tuple:
        return self.tp_shard_dims, self.tp_num_parts

    @property
    def zero_meta(self) -> tuple:
        return self.zero_numel, self.zero_orig_shape

    @staticmethod
    def from_dict(d: dict) -> 'ParamDistMeta':
        return ParamDistMeta(**d)


@dataclass
class ParamRedistMeta:
    # parallel info
    dp_world_size: int
    tp_world_size: int
    # tp info
    tp_shard_dims: Optional[List[int]] = None
    tp_num_parts: Optional[List[int]] = None
    # zero info
    zero_start_dp_rank: Optional[int] = None
    zero_offsets: Optional[List[int]] = None

    @property
    def used_tp(self) -> bool:
        return self.tp_shard_dims is not None and self.tp_num_parts is not None

    @property
    def used_zero(self) -> bool:
        return self.zero_start_dp_rank is not None and self.zero_offsets is not None


@dataclass
class RankRedistMeta:
    dp_rank: int
    tp_rank: int
    pp_rank: int


@dataclass
class PipelineRedistMeta:
    params: Set[str]


@dataclass
class RedistMeta:
    rank_meta: Dict[str, Dict[int, RankRedistMeta]]
    pipeline_meta: List[PipelineRedistMeta]
    param_meta: Dict[str, ParamRedistMeta]
