from abc import ABC, abstractmethod
from typing import Optional
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
    def write(self, name: str, state_dict: dict) -> None:
        pass

    @abstractmethod
    def save_model(self, model_checkpoint: dict) -> None:
        pass

    @abstractmethod
    def save_optimizer(self, optimizer_checkpoint: dict) -> None:
        pass

    @abstractmethod
    def save_meta(self, meta_checkpoint: dict) -> None:
        pass

    @abstractmethod
    def save_others(self, kwargs: dict) -> None:
        pass


class DiskCheckpointWriter(CheckpointWriter):

    def __init__(self, base_name: str, overwrite: bool = False, rank: int = 0, world_size: int = 1) -> None:
        super().__init__(base_name, overwrite, rank, world_size)
        if not os.path.exists(base_name):
            os.makedirs(base_name)
        assert os.path.isdir(base_name), f'"{base_name}" is not a directory'
        self.model_checkpoint_names = []
        self.optimizer_checkpoint_names = []
        self.is_meta_saved: bool = False
        self._save_global_meta()

    def write(self, name: str, state_dict: dict) -> None:
        path = os.path.join(self.base_name, name)
        if os.path.exists(path) and not self.overwrite:
            raise RuntimeError(f'Save error: Checkpoint "{path}" exists. (overwrite = False)')
        torch.save(state_dict, path)

    def _save_global_meta(self) -> None:
        if self.is_main_process:
            global_meta = {'meta': []}
            if self.is_distributed:
                for i in range(self.world_size):
                    global_meta['meta'].append(META_CKPT_FILE_NAME.replace('.bin', f'-rank{i}.bin'))
            else:
                global_meta['meta'].append(META_CKPT_FILE_NAME)
            self.write(GLOBAL_META_FILE_NAME, global_meta)

    def _get_checkpoint_name(self, base_name: str, shard_idx: Optional[int] = None) -> str:
        checkpoint_name = base_name
        if self.is_distributed:
            checkpoint_name = checkpoint_name.replace('.bin', f'-rank{self.rank}.bin')
        if shard_idx is not None:
            checkpoint_name = checkpoint_name.replace('.bin', f'-shard{shard_idx}.bin')
        return checkpoint_name

    def save_model(self, model_checkpoint: dict) -> None:
        assert not self.is_meta_saved, 'Cannot save model after saving meta'
        name = self._get_checkpoint_name(MODEL_CKPT_FILE_NAME, len(self.model_checkpoint_names))
        self.write(name, model_checkpoint)
        self.model_checkpoint_names.append(name)

    def save_optimizer(self, optimizer_checkpoint: dict) -> None:
        assert not self.is_meta_saved, 'Cannot save optimizer after saving meta'
        name = self._get_checkpoint_name(OPTIM_CKPT_FILE_NAME, len(self.optimizer_checkpoint_names))
        self.write(name, optimizer_checkpoint)
        self.optimizer_checkpoint_names.append(name)

    def save_meta(self, meta_checkpoint: dict) -> None:
        if len(self.model_checkpoint_names) > 0:
            meta_checkpoint['model'] = self.model_checkpoint_names
        if len(self.optimizer_checkpoint_names) > 0:
            meta_checkpoint['optimizer'] = self.optimizer_checkpoint_names
        self.write(self._get_checkpoint_name(META_CKPT_FILE_NAME), meta_checkpoint)
        self.is_meta_saved = True

    def save_others(self, kwargs: dict) -> None:
        if self.is_main_process:
            self.write(OTHER_CKPT_FILE_NAME, kwargs)
