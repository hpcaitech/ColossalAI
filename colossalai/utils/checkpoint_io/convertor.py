from torch import Tensor
from typing import Any, List, Tuple, Optional, Dict, Callable
from abc import abstractmethod, ABC
from .meta import ParamDistMeta, RedistMeta
from .utils import build_checkpoints, ModelCheckpointSharder, run_if_not_none, OptimizerCheckpointSharder
from .distributed import merge_param, unmerge_param
from collections import defaultdict


class ModelCheckpointConvertor(ABC):

    def __init__(self, param_count: Dict[str, int]) -> None:
        super().__init__()
        self.param_count = param_count
        self.buffer: Dict[str, Dict[int, Tensor]] = defaultdict(dict)

    @abstractmethod
    def handle_tensors(self, key: str, tensors: List[Tensor], dist_metas: List[ParamDistMeta]) -> None:
        pass

    def append(self, shard_dict: Dict[int, dict], dist_meta_list: List[Dict[str, ParamDistMeta]]) -> None:
        for rank, state_dict in shard_dict.items():
            for k, tensor in state_dict.items():
                self.buffer[k][rank] = tensor
        handled_keys = set()
        for k, rank_dict in self.buffer.items():
            if len(rank_dict) == self.param_count[k]:
                tensors = []
                dist_metas = []
                for rank, tensor in rank_dict.items():
                    tensors.append(tensor)
                    dist_metas.append(dist_meta_list[rank][k])
                self.handle_tensors(k, tensors, dist_metas)
                handled_keys.add(k)
        for k in handled_keys:
            del self.buffer[k]

    def complete(self) -> None:
        assert len(self.buffer) == 0


class ModelCheckpointMerger(ModelCheckpointConvertor):

    def __init__(self, max_shard_size: int, save_fn: Callable[[dict], Any], param_count: Dict[str, int]) -> None:
        super().__init__(param_count)
        self.sharder = ModelCheckpointSharder(max_shard_size)
        self.save_fn = save_fn

    def handle_tensors(self, key: str, tensors: List[Tensor], dist_metas: List[ParamDistMeta]) -> None:
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

    def handle_tensors(self, key: str, tensors: List[Tensor], dist_metas: List[ParamDistMeta]) -> None:
        tensor = merge_param(tensors, dist_metas)
        for tp_rank, tensor_list in enumerate(unmerge_param(tensor, self.redist_meta.param_meta[key])):
            for dp_rank, t in enumerate(tensor_list):
                for rank in self.rank_map[tp_rank][dp_rank]:
                    shard = self.sharders[rank].append(key, t)
                    run_if_not_none(self.save_fns[rank], shard)

    def complete(self) -> None:
        super().complete()
        for rank, save_fn in enumerate(self.save_fns):
            run_if_not_none(save_fn, self.sharders[rank].complete())


class OptimizerCheckpointConvertor(ABC):

    def __init__(self, param_count: Dict[str, int], param_to_os: Optional[Dict[str, int]],
                 paired_os: Optional[Dict[int, dict]]) -> None:
        super().__init__()
        self.param_count = param_count
        self.param_to_os = param_to_os
        self.paired_os = paired_os
        self.buffer: Dict[int, Dict[int, dict]] = defaultdict(dict)

    @abstractmethod
    def setup(self, param_groups: dict) -> None:
        pass

    @abstractmethod
    def handle_states(self, idx: int, states: List[dict], dist_metas: List[ParamDistMeta]) -> None:
        pass

    def append(self, shard_dict: Dict[str, dict], dist_meta_list: List[Dict[str, ParamDistMeta]]) -> None:
        os_to_param = {v: k for k, v in self.param_to_os.items()}
        for rank, state_dict in shard_dict.items():
            self.setup(state_dict['param_groups'])
            for idx, state in state_dict['state'].items():
                self.buffer[idx][rank] = state
        handled_indices = set()
        for idx, rank_dict in self.buffer.items():
            if len(rank_dict) == self.param_count[os_to_param[idx]]:
                states = []
                dist_metas = []
                for rank, state in rank_dict.items():
                    states.append(state)
                    dist_metas.append(dist_meta_list[rank][os_to_param[idx]])
                self.handle_states(idx, states, dist_metas)
                handled_indices.add(idx)
        for idx in handled_indices:
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

    def handle_states(self, idx: int, states: List[dict], dist_metas: List[ParamDistMeta]) -> None:
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
