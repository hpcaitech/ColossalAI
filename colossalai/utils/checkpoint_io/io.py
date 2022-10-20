from torch.nn import Module
from torch.optim import Optimizer
from torch import Tensor
from typing import Any, List, Tuple, Optional, Dict, Callable
from .meta import ParamDistMeta, RedistMeta
from .utils import build_checkpoints, ModelCheckpointSharder, run_if_not_none, OptimizerCheckpointSharder
from .backend import get_backend
from .convertor import ModelCheckpointMerger, OptimizerCheckpointMerger
import torch.distributed as dist
import warnings


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


def merge(path: str, output_path: str, max_shard_size_gb: float = 0.0, overwrite: bool = False, backend: str = 'disk'):
    io_backend = get_backend(backend)
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    reader = io_backend.get_reader(path)
    if len(reader.meta_list) == 1:
        # already global
        warnings.warn(f'Checkpoint at "{path}" is already global, nothing to do.')
        return
    dist_meta_list, param_count, param_to_os, paired_os = reader.load_meta()
    writer = io_backend.get_writer(output_path, overwrite=overwrite)
    writer.save_others(reader.load_others())
    max_shard_size = int(max_shard_size_gb * 1024**3)
    convertor = ModelCheckpointMerger(max_shard_size, writer.save_model, param_count)
    for shard_dict in reader.load_model():
        convertor.append(shard_dict, dist_meta_list)
    convertor.complete()
    convertor = OptimizerCheckpointMerger(max_shard_size, writer.save_optimizer, param_count, param_to_os, paired_os)
    for shard_dict in reader.load_optimizer():
        convertor.append(shard_dict, dist_meta_list)
    convertor.complete()
    meta_checkpoint = {'dist_meta': None, 'params': list(param_count.keys())}
    if param_to_os is not None:
        meta_checkpoint['param_to_os'] = param_to_os
        meta_checkpoint['paired_os'] = paired_os
    writer.save_meta(meta_checkpoint)
