from torch.nn import Module
from torch.optim import Optimizer
from typing import Any, List, Tuple, Optional, Dict
from .meta import ParamDistMeta
from .utils import build_checkpoints
from .writer import DiskCheckpointWriter
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
        save_global = False
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        save_global = True
        rank = 0
        world_size = 1
    writer = DiskCheckpointWriter(path, overwrite, rank, world_size)
    max_shard_size = int(max_shard_size_gb * 1024**3)
    model_checkpoints, optimizer_checkpoints, meta_checkpoint = build_checkpoints(save_global, max_shard_size, model,
                                                                                  optimizer, param_to_os, dist_meta)
    checkpoints, checkpoint_names = writer.process_checkpoint(model_checkpoints, optimizer_checkpoints, meta_checkpoint,
                                                              **kwargs)
    writer.setup()
    for checkpoint, checkpoint_name in zip(checkpoints, checkpoint_names):
        writer.write(checkpoint_name, checkpoint)
