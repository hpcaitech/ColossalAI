from torch.nn import Module
from torch.optim import Optimizer
from typing import Any, List, Tuple, Optional, Dict
from .meta import ParamDistMeta
from .utils import build_checkpoints
from .writer import DiskCheckpointWriter
from .reader import DiskCheckpointReader
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
    assert backend == 'disk'
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
    writer = DiskCheckpointWriter(path, overwrite, rank, world_size)
    writer.save_others(kwargs)
    for model_checkpoint in model_checkpoints:
        writer.save_model(model_checkpoint)
    for optimizer_checkpoint in optimizer_checkpoints:
        writer.save_optimizer(optimizer_checkpoint)
    writer.save_meta(meta_checkpoint)


def merge(path: str, output_path: str, max_shard_size_gb: float = 0.0, overwrite: bool = False, backend: str = 'disk'):
    assert backend == 'disk'
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    reader = DiskCheckpointReader(path)
    if len(reader.meta_list) == 1:
        # already global
        # copy
        return
    dist_meta_list, param_to_os_list, paired_os_list = reader.load_meta()
    buffer = defaultdict(dict)
    merged_buffer = {}
    for shard_dict in reader.load_model():
        for rank, state_dict in shard_dict.items():
            for k, tensor in state_dict.items():
                buffer[k][rank] = tensor
        for k, rank_dict in buffer.items():
            if (not dist_meta_list[0][k].use_zero and len(rank_dict) == 1) or \
                    len(rank_dict) == len(reader.meta_list):
                tensors = []
                dist_metas = []
                for rank, tensor in rank_dict.items():
                    tensors.append(tensor)
                    dist_metas.append(dist_meta_list[rank][k])
                tensor = merge_param(tensors, dist_metas)
                merged_buffer[k] = tensor
        for k in merged_buffer:
            del buffer[k]
