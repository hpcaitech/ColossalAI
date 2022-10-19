from torch.nn import Module
from torch.optim import Optimizer
from torch import Tensor
from typing import Any, List, Tuple, Optional, Dict, Callable
from .meta import ParamDistMeta
from .utils import build_checkpoints, ModelCheckpointSharder, run_if_not_none, OptimizerCheckpointSharder
from .backend import get_backend
from .distributed import merge_param
from collections import defaultdict
import torch.distributed as dist


def save(path: str,
         model: Module,
         optimizer: Optional[Optimizer] = None,
         param_to_os: Optional[Dict[str, int]] = None,
         dist_meta: Optional[Dict[str, ParamDistMeta]] = None,
         max_shard_size_gb: float = 0.0,
         overwrite: bool = False,
         backend: str = 'disk',
         **kwargs: Any):
    io_backend = get_backend(backend)
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        assert dist_meta is not None
    else:
        rank = 0
        world_size = 1
        # global doesn't need dist_meta
        dist_meta = None
    max_shard_size = int(max_shard_size_gb * 1024**3)
    model_checkpoints, optimizer_checkpoints, meta_checkpoint = build_checkpoints(max_shard_size, model, optimizer,
                                                                                  param_to_os, dist_meta)
    writer = io_backend.get_writer(path, overwrite, rank, world_size)
    writer.save_others(kwargs)
    for model_checkpoint in model_checkpoints:
        writer.save_model(model_checkpoint)
    for optimizer_checkpoint in optimizer_checkpoints:
        writer.save_optimizer(optimizer_checkpoint)
    writer.save_meta(meta_checkpoint)


class ModelCheckpointConvertor:

    def __init__(self, max_shard_size: int, save_fn: Callable[[dict], Any]) -> None:
        self.sharder = ModelCheckpointSharder(max_shard_size)
        self.buffer: Dict[str, Dict[int, Tensor]] = defaultdict(dict)
        self.save_fn = save_fn

    def append(self, shard_dict: Dict[int, dict], dist_meta_list: List[Dict[str, ParamDistMeta]], n_procs: int) -> None:
        for rank, state_dict in shard_dict.items():
            for k, tensor in state_dict.items():
                self.buffer[k][rank] = tensor
        merged_keys = set()
        for k, rank_dict in self.buffer.items():
            if (not dist_meta_list[0][k].use_zero and len(rank_dict) == 1) or \
                    len(rank_dict) == n_procs:
                tensors = []
                dist_metas = []
                for rank, tensor in rank_dict.items():
                    tensors.append(tensor)
                    dist_metas.append(dist_meta_list[rank][k])
                tensor = merge_param(tensors, dist_metas)
                merged_keys.add(k)
                shard = self.sharder.append(k, tensor)
                run_if_not_none(self.save_fn, shard)
        for k in merged_keys:
            del self.buffer[k]

    def complete(self) -> None:
        assert len(self.buffer) == 0
        run_if_not_none(self.save_fn, self.sharder.complete())


class OptimizerCheckpointConvertor:

    def __init__(self, max_shard_size: int, save_fn: Callable[[dict], Any]) -> None:
        self.sharder = None
        self.max_shard_size = max_shard_size
        self.buffer: Dict[int, Dict[int, dict]] = defaultdict(dict)
        self.save_fn = save_fn

    def _setup(self, param_groups: dict) -> None:
        if self.sharder is None:
            self.sharder = OptimizerCheckpointSharder(self.max_shard_size, param_groups)

    def append(self, shard_dict: Dict[str, dict], dist_meta_list: List[Dict[str, ParamDistMeta]],
               param_to_os_list: List[Dict[str, int]], paired_os_list: List[Dict[int, dict]], n_procs: int) -> None:
        os_to_param = {v: k for k, v in param_to_os_list[0].items()}
        paired_os = paired_os_list[0]
        for rank, state_dict in shard_dict.items():
            self._setup(state_dict['param_groups'])
            for idx, state in state_dict['state'].items():
                self.buffer[idx][rank] = state
        merged_keys = set()
        for idx, rank_dict in self.buffer.items():
            if (not dist_meta_list[0][os_to_param[idx]].use_zero and len(rank_dict) == 1) or \
                    len(rank_dict) == n_procs:
                states = []
                dist_metas = []
                for rank, state in rank_dict.items():
                    states.append(state)
                    dist_metas.append(dist_meta_list[rank][os_to_param[idx]])
                new_state = {}
                for k, state_tensor in states[0].items():
                    if paired_os[idx][k]:
                        new_state[k] = merge_param([state[k] for state in states], dist_metas)
                    else:
                        new_state[k] = state_tensor
                merged_keys.add(idx)
                shard = self.sharder.append(idx, new_state)
                run_if_not_none(self.save_fn, shard)
        for idx in merged_keys:
            del self.buffer[idx]

    def complete(self) -> None:
        assert len(self.buffer) == 0
        run_if_not_none(self.save_fn, self.sharder.complete())


def merge(path: str, output_path: str, max_shard_size_gb: float = 0.0, overwrite: bool = False, backend: str = 'disk'):
    io_backend = get_backend(backend)
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    reader = io_backend.get_reader(path)
    if len(reader.meta_list) == 1:
        # already global
        # copy
        return
    dist_meta_list, param_to_os_list, paired_os_list = reader.load_meta()
    writer = io_backend.get_writer(output_path, overwrite=overwrite)
    writer.save_others(reader.load_others())
    max_shard_size = int(max_shard_size_gb * 1024**3)
    convertor = ModelCheckpointConvertor(max_shard_size, writer.save_model)
    for shard_dict in reader.load_model():
        convertor.append(shard_dict, dist_meta_list, len(reader.meta_list))
    convertor.complete()
    convertor = OptimizerCheckpointConvertor(max_shard_size, writer.save_optimizer)
    for shard_dict in reader.load_optimizer():
        convertor.append(shard_dict, dist_meta_list, param_to_os_list, paired_os_list, len(reader.meta_list))
    convertor.complete()
    meta_checkpoint = {'dist_meta': None}
    if param_to_os_list[0] is not None:
        meta_checkpoint['param_to_os'] = param_to_os_list[0]
        meta_checkpoint['paired_os'] = paired_os_list[0]
    writer.save_meta(meta_checkpoint)
