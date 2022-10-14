from abc import ABC, abstractmethod
from typing import Any, List, Tuple
from .constant import MODEL_CKPT_FILE_NAME, OPTIM_CKPT_FILE_NAME, META_CKPT_FILE_NAME, OTHER_CKPT_FILE_NAME, GLOBAL_META_FILE_NAME
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
    def process_checkpoint(self, model_checkpoints: List[dict], optimizer_checkpoints: List[dict],
                           meta_checkpoint: dict, **kwargs: Any) -> Tuple[List[str], List[dict]]:
        pass

    @abstractmethod
    def write(self, name: str, state_dict: dict) -> None:
        pass


class DiskCheckpointWriter(CheckpointWriter):

    def __init__(self, base_name: str, overwrite: bool = False, rank: int = 0, world_size: int = 1) -> None:
        super().__init__(base_name, overwrite, rank, world_size)
        if not os.path.exists(base_name):
            os.makedirs(base_name)
        assert os.path.isdir(base_name), f'"{base_name}" is not a directory'

    def get_checkpoint_names(self, n_shards: int, base_name: str) -> List[str]:
        checkpoint_names = []
        for i in range(n_shards):
            checkpoint_name = base_name
            if self.is_distributed:
                checkpoint_name = checkpoint_name.replace('.bin',
                                                          f'-rank{self.rank+1:05d}-of-{self.world_size:05d}.bin')
            if n_shards > 1:
                checkpoint_name = checkpoint_name.replace('.bin', f'-shard{i+1:05d}-of-{n_shards:05d}.bin')
            checkpoint_names.append(checkpoint_name)
        return checkpoint_names

    def process_checkpoint(self, model_checkpoints: List[dict], optimizer_checkpoints: List[dict],
                           meta_checkpoint: dict, **kwargs: Any) -> Tuple[List[str], List[dict]]:
        checkpoints = []
        checkpoint_names = []
        if self.is_main_process:
            global_meta = {'meta': []}
            if self.is_distributed:
                for i in range(self.world_size):
                    global_meta['meta'].append(
                        META_CKPT_FILE_NAME.replace('.bin', f'-rank{i+1:05d}-of-{self.world_size:05d}.bin'))
            else:
                global_meta['meta'].append(META_CKPT_FILE_NAME)
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
        checkpoint_names.extend(self.get_checkpoint_names(1, META_CKPT_FILE_NAME))
        return checkpoints, checkpoint_names

    def write(self, name: str, state_dict: dict) -> None:
        path = os.path.join(self.base_name, name)
        if os.path.exists(path) and not self.overwrite:
            raise RuntimeError(f'Save error: Checkpoint "{path}" exists. (overwrite = False)')
        torch.save(state_dict, path)
