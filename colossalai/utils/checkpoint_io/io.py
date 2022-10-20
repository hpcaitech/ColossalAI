import warnings
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import torch.distributed as dist
from torch.nn import Module
from torch.optim import Optimizer

from .backend import get_backend
from .convertor import (CheckpointConvertor, ModelCheckpointMerger, ModelCheckpointRedistor, OptimizerCheckpointMerger,
                        OptimizerCheckpointRedistor)
from .meta import ParamDistMeta, RedistMeta
from .utils import build_checkpoints


def save(path: str,
         model: Module,
         optimizer: Optional[Optimizer] = None,
         param_to_os: Optional[Dict[str, int]] = None,
         dist_meta: Optional[Dict[str, ParamDistMeta]] = None,
         max_shard_size_gb: float = 0.0,
         overwrite: bool = False,
         backend: str = 'disk',
         **kwargs: Any) -> None:
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


def merge(path: str,
          output_path: str,
          max_shard_size_gb: float = 0.0,
          overwrite: bool = False,
          backend: str = 'disk') -> None:
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
    _convert_shards(ModelCheckpointMerger(max_shard_size, writer.save_model, param_count), reader.load_models(),
                    dist_meta_list)
    _convert_shards(
        OptimizerCheckpointMerger(max_shard_size, writer.save_optimizer, param_count, param_to_os, paired_os),
        reader.load_optimizers(), dist_meta_list)
    meta_checkpoint = {'dist_meta': None, 'params': list(param_count.keys())}
    if param_to_os is not None:
        meta_checkpoint['param_to_os'] = param_to_os
        meta_checkpoint['paired_os'] = paired_os
    writer.save_meta(meta_checkpoint)


def redist(path: str,
           output_path: str,
           redist_meta: RedistMeta,
           dist_metas: List[Dict[str, ParamDistMeta]],
           max_shard_size_gb: float = 0.0,
           overwrite: bool = False,
           backend: str = 'disk') -> None:
    io_backend = get_backend(backend)
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    nprocs = len(dist_metas)
    reader = io_backend.get_reader(path)
    dist_meta_list, param_count, param_to_os, paired_os = reader.load_meta()
    do_redist: bool = False
    if len(dist_meta_list) == nprocs:
        for a, b in zip(dist_metas, dist_meta_list):
            if a != b:
                do_redist = True
                break
    else:
        do_redist = True
    if not do_redist:
        warnings.warn(f'Checkpoint at "{path}" is not required to redist, nothing to do.')
        return

    writers = [io_backend.get_writer(output_path, overwrite, rank, nprocs) for rank in range(nprocs)]
    writers[0].save_others(reader.load_others())
    max_shard_size = int(max_shard_size_gb * 1024**3)
    _convert_shards(
        ModelCheckpointRedistor(max_shard_size, [writer.save_model for writer in writers], param_count, redist_meta),
        reader.load_models(), dist_meta_list)
    _convert_shards(
        OptimizerCheckpointRedistor(max_shard_size, [writer.save_optimizer for writer in writers], param_count,
                                    param_to_os, paired_os, redist_meta), reader.load_optimizers(), dist_meta_list)
    for writer, dist_meta in zip(writers, dist_metas):
        meta_checkpoint = {'dist_meta': dist_meta, 'params': list(param_count.keys())}
        if param_to_os is not None:
            meta_checkpoint['param_to_os'] = param_to_os
            meta_checkpoint['paired_os'] = paired_os
        writer.save_meta(meta_checkpoint)


def _convert_shards(convertor: CheckpointConvertor, shard_generator: Generator[dict, None, None],
                    dist_meta_list: List[Optional[Dict[str, ParamDistMeta]]]) -> None:
    for shard_dict in shard_generator:
        convertor.append(shard_dict, dist_meta_list)
    convertor.complete()
