from abc import ABC, abstractmethod
from typing import Any, List, Tuple
from .constant import CKPT_PAT, MODEL_CKPT_FILE_NAME, OPTIM_CKPT_FILE_NAME, META_CKPT_FILE_NAME, OTHER_CKPT_FILE_NAME, GLOBAL_META_FILE_NAME
import torch
import os


class CheckpointWriter(ABC):

    def __init__(self, base_name: str, overwrite: bool = False, rank: int = 0, world_size: int = 1) -> None:
        super().__init__()
        self.base_name = base_name
        self.overwrite = overwrite
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        self.is_main_process = rank == 0

    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def process_checkpoint(self, model_checkpoints: List[dict], optimizer_checkpoints: List[dict],
                           meta_checkpoint: dict, **kwargs: Any) -> Tuple[List[str], List[dict]]:
        pass

    @abstractmethod
    def write(self, name: str, state_dict: dict) -> None:
        pass


class DiskCheckpointWriter(CheckpointWriter):

    def setup(self):
        dir_name = self.base_name
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        assert os.path.isdir(dir_name), f'"{dir_name}" is not a directory'
        for file_name in os.listdir(dir_name):
            if CKPT_PAT.match(file_name):
                if self.overwrite:
                    os.remove(os.path.join(dir_name, file_name))
                else:
                    raise RuntimeError(f'Cannot save checkpoint, because it already exists. (overwrite = False)')

    def get_checkpoint_names(self, n_shards: int, base_name: str) -> List[str]:
        checkpoint_names = []
        for i in range(n_shards):
            checkpoint_name = base_name
            if self.is_distributed:
                checkpoint_name = checkpoint_name.replace('.bin', f'rank{self.rank+1:05d}-of-{self.world_size:05d}.bin')
            if n_shards > 1:
                checkpoint_name = checkpoint_name.replace('.bin', f'shard{i+1:05d}-of-{n_shards:05d}.bin')
            checkpoint_names.append(checkpoint_name)
        return checkpoint_names

    def process_checkpoint(self, model_checkpoints: List[dict], optimizer_checkpoints: List[dict],
                           meta_checkpoint: dict, **kwargs: Any) -> Tuple[List[str], List[dict]]:
        checkpoints = []
        checkpoint_names = []
        if self.is_main_process:
            global_meta = {'meta': self.get_checkpoint_names(1, META_CKPT_FILE_NAME)}
            checkpoints.append(global_meta)
            checkpoint_names.append(GLOBAL_META_FILE_NAME)
            checkpoints.append(kwargs)
            checkpoint_names.append(OTHER_CKPT_FILE_NAME)
        if len(model_checkpoints) > 0:
            model_checkpoint_names = self.get_checkpoint_names(len(model_checkpoints), MODEL_CKPT_FILE_NAME)
            meta_checkpoint['model'] = model_checkpoint_names
            checkpoints.extend(model_checkpoints)
            checkpoint_names.extend(model_checkpoint_names)
        if len(optimizer_checkpoints) > 0:
            optimizer_checkpoint_names = self.get_checkpoint_names(len(optimizer_checkpoints), OPTIM_CKPT_FILE_NAME)
            meta_checkpoint['optimizer'] = optimizer_checkpoint_names
            checkpoints.extend(optimizer_checkpoints)
            checkpoint_names.extend(optimizer_checkpoint_names)
        checkpoints.append(meta_checkpoint)
        checkpoint_names.append(self.get_checkpoint_names(1, META_CKPT_FILE_NAME))
        checkpoint_names = [os.path.join(self.base_name, name) for name in checkpoint_names]
        return checkpoints, checkpoint_names

    def write(self, name: str, state_dict: dict) -> None:
        torch.save(state_dict, name)
