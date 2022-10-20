from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from torch import Tensor

from .distributed import merge_param, unmerge_param
from .meta import ParamDistMeta, RedistMeta
from .utils import (ModelCheckpointSharder, OptimizerCheckpointSharder, run_if_not_none)


class CheckpointConvertor(ABC):

    @abstractmethod
    def append(self, shard_dict: Dict[int, dict], dist_meta_list: List[Optional[Dict[str, ParamDistMeta]]]) -> None:
        pass

    @abstractmethod
    def complete(self) -> None:
        pass


class ModelCheckpointConvertor(CheckpointConvertor):

    def __init__(self, param_count: Dict[str, int]) -> None:
        super().__init__()
        self.param_count = param_count
        self.buffer: Dict[str, Dict[int, Tensor]] = defaultdict(dict)

    @abstractmethod
    def convert_tensors(self, key: str, tensors: List[Tensor], dist_metas: List[ParamDistMeta]) -> None:
        pass

    def append(self, shard_dict: Dict[int, dict], dist_meta_list: List[Optional[Dict[str, ParamDistMeta]]]) -> None:
        for rank, state_dict in shard_dict.items():
            for k, tensor in state_dict.items():
                self.buffer[k][rank] = tensor
        converted_keys = set()
        for k, rank_dict in self.buffer.items():
            if len(rank_dict) == self.param_count[k]:
                tensors = []
                dist_metas = []
                for rank, tensor in rank_dict.items():
                    tensors.append(tensor)
                    if dist_meta_list[rank] is not None:
                        dist_metas.append(dist_meta_list[rank][k])
                self.convert_tensors(k, tensors, dist_metas)
                converted_keys.add(k)
        for k in converted_keys:
            del self.buffer[k]

    def complete(self) -> None:
        assert len(self.buffer) == 0


class ModelCheckpointMerger(ModelCheckpointConvertor):

    def __init__(self, max_shard_size: int, save_fn: Callable[[dict], Any], param_count: Dict[str, int]) -> None:
        super().__init__(param_count)
        self.sharder = ModelCheckpointSharder(max_shard_size)
        self.save_fn = save_fn

    def convert_tensors(self, key: str, tensors: List[Tensor], dist_metas: List[ParamDistMeta]) -> None:
        assert len(dist_metas) == len(tensors)
        tensor = merge_param(tensors, dist_metas)
        shard = self.sharder.append(key, tensor)
        run_if_not_none(self.save_fn, shard)

    def complete(self) -> None:
        super().complete()
        run_if_not_none(self.save_fn, self.sharder.complete())


class ModelCheckpointRedistor(ModelCheckpointConvertor):

    def __init__(self, max_shard_size: int, save_fns: List[Callable[[dict], Any]], param_count: Dict[str, int],
                 redist_meta: RedistMeta) -> None:
        super().__init__(param_count)
        self.save_fns = save_fns
        self.redist_meta = redist_meta
        nprocs = len(save_fns)
        self.sharders = [ModelCheckpointSharder(max_shard_size) for _ in range(nprocs)]
        self.rank_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for k, rank_meta in redist_meta.rank_meta.items():
            for rank, rank_info in rank_meta.items():
                self.rank_map[k][rank_info.tp_rank][rank_info.dp_rank].append(rank)

    def convert_tensors(self, key: str, tensors: List[Tensor], dist_metas: List[ParamDistMeta]) -> None:
        if len(dist_metas) == 0:
            # already global
            tensor = tensors[0]
        else:
            assert len(dist_metas) == len(tensors)
            tensor = merge_param(tensors, dist_metas)
        for tp_rank, tensor_list in enumerate(unmerge_param(tensor, self.redist_meta.param_meta[key])):
            for dp_rank, t in enumerate(tensor_list):
                for rank in self.rank_map[key][tp_rank][dp_rank]:
                    shard = self.sharders[rank].append(key, t)
                    run_if_not_none(self.save_fns[rank], shard)

    def complete(self) -> None:
        super().complete()
        for rank, save_fn in enumerate(self.save_fns):
            run_if_not_none(save_fn, self.sharders[rank].complete())


class OptimizerCheckpointConvertor(CheckpointConvertor):

    def __init__(self, param_count: Dict[str, int], param_to_os: Optional[Dict[str, int]],
                 paired_os: Optional[Dict[int, dict]]) -> None:
        super().__init__()
        self.param_count = param_count
        self.param_to_os = param_to_os
        self.paired_os = paired_os
        self.buffer: Dict[int, Dict[int, dict]] = defaultdict(dict)
        self.os_to_param = {v: k for k, v in param_to_os.items()}

    @abstractmethod
    def setup(self, param_groups: dict) -> None:
        pass

    @abstractmethod
    def convert_states(self, idx: int, states: List[dict], dist_metas: List[ParamDistMeta]) -> None:
        pass

    def append(self, shard_dict: Dict[int, dict], dist_meta_list: List[Optional[Dict[str, ParamDistMeta]]]) -> None:
        for rank, state_dict in shard_dict.items():
            self.setup(state_dict['param_groups'])
            for idx, state in state_dict['state'].items():
                self.buffer[idx][rank] = state
        converted_indices = set()
        for idx, rank_dict in self.buffer.items():
            if len(rank_dict) == self.param_count[self.os_to_param[idx]]:
                states = []
                dist_metas = []
                for rank, state in rank_dict.items():
                    states.append(state)
                    if dist_meta_list[rank] is not None:
                        dist_metas.append(dist_meta_list[rank][self.os_to_param[idx]])
                self.convert_states(idx, states, dist_metas)
                converted_indices.add(idx)
        for idx in converted_indices:
            del self.buffer[idx]

    def complete(self) -> None:
        assert len(self.buffer) == 0


class OptimizerCheckpointMerger(OptimizerCheckpointConvertor):

    def __init__(self, max_shard_size: int, save_fn: Callable[[dict], Any], param_count: Dict[str, int],
                 param_to_os: Optional[Dict[str, int]], paired_os: Optional[Dict[int, dict]]) -> None:
        super().__init__(param_count, param_to_os, paired_os)
        self.max_shard_size = max_shard_size
        self.save_fn = save_fn
        self.sharder = None

    def setup(self, param_groups: dict) -> None:
        if self.sharder is None:
            self.sharder = OptimizerCheckpointSharder(self.max_shard_size, param_groups)

    def convert_states(self, idx: int, states: List[dict], dist_metas: List[ParamDistMeta]) -> None:
        assert len(dist_metas) == len(states)
        new_state = {}
        for state_key, state_tensor in states[0].items():
            if self.paired_os[idx][state_key]:
                new_state[state_key] = merge_param([state[state_key] for state in states], dist_metas)
            else:
                new_state[state_key] = state_tensor
        shard = self.sharder.append(idx, new_state)
        run_if_not_none(self.save_fn, shard)

    def complete(self) -> None:
        super().complete()
        run_if_not_none(self.save_fn, self.sharder.complete())


class OptimizerCheckpointRedistor(OptimizerCheckpointConvertor):

    def __init__(self, max_shard_size: int, save_fns: List[Callable[[dict], Any]], param_count: Dict[str, int],
                 param_to_os: Optional[Dict[str, int]], paired_os: Optional[Dict[int, dict]],
                 redist_meta: RedistMeta) -> None:
        super().__init__(param_count, param_to_os, paired_os)
        self.max_shard_size = max_shard_size
        self.save_fns = save_fns
        self.redist_meta = redist_meta
        self.sharders: List[OptimizerCheckpointSharder] = []
        self.rank_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for k, rank_meta in redist_meta.rank_meta.items():
            for rank, rank_info in rank_meta.items():
                self.rank_map[k][rank_info.tp_rank][rank_info.dp_rank].append(rank)

    def setup(self, param_groups: dict) -> None:
        if len(self.sharders) == 0:
            nprocs = len(self.save_fns)
            for _ in range(nprocs):
                self.sharders.append(OptimizerCheckpointSharder(self.max_shard_size, param_groups))

    def convert_states(self, idx: int, states: List[dict], dist_metas: List[ParamDistMeta]) -> None:
        need_merge: bool = True
        if len(dist_metas) == 0:
            need_merge = False
        else:
            assert len(dist_metas) == len(states)
        new_states = [{} for _ in range(len(self.save_fns))]
        for state_key, state_tensor in states[0].items():
            if self.paired_os[idx][state_key]:
                if need_merge:
                    tensor = merge_param([state[state_key] for state in states], dist_metas)
                else:
                    tensor = state_tensor
                for tp_rank, tensor_list in enumerate(
                        unmerge_param(tensor, self.redist_meta.param_meta[self.os_to_param[idx]])):
                    for dp_rank, t in enumerate(tensor_list):
                        for rank in self.rank_map[self.os_to_param[idx]][tp_rank][dp_rank]:
                            new_states[rank][state_key] = t
            else:
                for new_state in new_states:
                    new_state[state_key] = state_tensor
        for rank, new_state in enumerate(new_states):
            shard = self.sharders[rank].append(idx, new_state)
            run_if_not_none(self.save_fns[rank], shard)

    def complete(self) -> None:
        super().complete()
        for rank, save_fn in enumerate(self.save_fns):
            run_if_not_none(save_fn, self.sharders[rank].complete())
