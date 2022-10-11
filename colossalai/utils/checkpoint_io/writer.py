from abc import ABC, abstractmethod
from torch.optim import Optimizer
from torch.nn import Module
from typing import Optional, Dict, Any, List
from .meta import ParamDistMeta
from .utils import get_param_to_os, shard_checkpoint
from .constant import CHECKPOINT_PREFIX, CHECKPOINT_FILE_NAME
import torch.distributed as dist
import torch
import os
import warnings


class CheckpointWriter(ABC):

    def __init__(self, max_size_gb: float = 0.0, overwrite: bool = False) -> None:
        super().__init__()
        self.max_size: int = int(max_size_gb * 1024**3)
        self.overwrite: bool = overwrite

    @property
    def save_global(self) -> bool:
        return not dist.is_initialized()

    @abstractmethod
    def write(self,
              name: str,
              model: Module,
              optimizer: Optional[Optimizer] = None,
              param_to_os: Optional[Dict[str, int]] = None,
              dist_meta: Optional[Dict[str, ParamDistMeta]] = None,
              **kwargs: Any) -> None:
        pass

    def build_checkpoint(self,
                         model: Module,
                         optimizer: Optional[Optimizer] = None,
                         param_to_os: Optional[Dict[str, int]] = None,
                         dist_meta: Optional[Dict[str, ParamDistMeta]] = None,
                         **kwargs: Any) -> List[dict]:
        if not self.save_global:
            assert dist_meta is not None, 'Expect dist_meta is not None, when not saving global'
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict() if optimizer else None
        if optimizer:
            param_to_os = param_to_os or get_param_to_os(model_state_dict)
        if not self.save_global:
            # filter dp replicated params
            model_state_dict = {
                k: v for k, v in model_state_dict.items() if dist_meta[k].used_zero or dist_meta[k].dp_rank == 0
            }
            if optimizer:
                optimizer_state_dict['state'] = {
                    param_to_os[k]: optimizer_state_dict['state'][param_to_os[k]]
                    for k in model_state_dict.items()
                    if dist_meta[k].used_zero or dist_meta[k].dp_rank == 0
                }
        checkpoints = []
        if len(model_state_dict) == 0:
            warnings.warn('model state dict is empty, checkpoint is not saved', category=RuntimeWarning)
            return checkpoints
        if self.max_size <= 0:
            checkpoint = {'model': model_state_dict}
            if optimizer is not None:
                checkpoint['optimizer'] = optimizer_state_dict
            checkpoints.append(checkpoint)
        else:
            checkpoints = shard_checkpoint(self.max_size, model_state_dict, optimizer_state_dict, param_to_os)
        if dist_meta is not None:
            checkpoints[0]['dist_meta'] = dist_meta
        if optimizer:
            checkpoints[0]['param_to_os'] = param_to_os
        checkpoints[0].update(kwargs)
        return checkpoints


class DiskCheckpointWriter(CheckpointWriter):

    def setup(self, dir_name: str):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        assert os.path.isdir(dir_name), f'"{dir_name}" is not a directory'
        for file_name in os.listdir(dir_name):
            if file_name.startswith(CHECKPOINT_PREFIX):
                if self.overwrite:
                    os.remove(os.path.join(dir_name, file_name))
                else:
                    raise RuntimeError(f'Cannot save checkpoint, because it already exists. (overwrite = False)')

    def get_checkpoint_names(self, n_shards: int) -> List[str]:
        checkpoint_names = []
        for i in range(n_shards):
            checkpoint_name = CHECKPOINT_FILE_NAME
            if not self.save_global:
                checkpoint_name = checkpoint_name.replace(
                    '.bin', f'rank{dist.get_rank()+1:05d}-of-{dist.get_world_size():05d}.bin')
            if n_shards > 1:
                checkpoint_name = checkpoint_name.replace('.bin', f'shard{i+1:05d}-of-{n_shards:05d}.bin')
            checkpoint_names.append(checkpoint_name)
        return checkpoint_names

    def write(self,
              dir_name: str,
              model: Module,
              optimizer: Optional[Optimizer] = None,
              param_to_os: Optional[Dict[str, int]] = None,
              dist_meta: Optional[Dict[str, ParamDistMeta]] = None,
              **kwargs: Any) -> None:
        self.setup(dir_name)
        checkpoints = self.build_checkpoint(model, optimizer, param_to_os, dist_meta, **kwargs)
        if len(checkpoints) == 0:
            return
        for checkpoint, checkpoint_name in zip(checkpoints, self.get_checkpoint_names(len(checkpoints))):
            torch.save(checkpoint, os.path.join(dir_name, checkpoint_name))
