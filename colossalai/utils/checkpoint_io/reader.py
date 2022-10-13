from abc import ABC, abstractmethod
from typing import Generator, List, Tuple, Optional, Dict
from .constant import GLOBAL_META_FILE_NAME, OTHER_CKPT_FILE_NAME
import torch
import os


class CheckpointReader(ABC):

    def __init__(self, base_name: str) -> None:
        super().__init__()
        self.base_name = base_name

    @abstractmethod
    def read(self, name: str) -> dict:
        pass

    @abstractmethod
    def load_meta(self) -> Tuple[List[Optional[List]], List[Optional[List]], List[Optional[List]]]:
        pass

    @abstractmethod
    def load_model(self) -> Generator[Dict[int, dict]]:
        pass

    @abstractmethod
    def load_optimizer(self) -> Generator[Dict[int, dict]]:
        pass

    @abstractmethod
    def load_other(self) -> dict:
        pass


class DiskCheckpointReader(CheckpointReader):

    def __init__(self, base_name: str) -> None:
        super().__init__(base_name)
        assert os.path.isdir(base_name), f'"{base_name}" is not a directory'
        self.meta_list = []
        global_meta = self.read(GLOBAL_META_FILE_NAME)
        for meta_file_name in global_meta['meta']:
            meta = self.read(meta_file_name)
            if meta.get('dist_meta', None) is None:
                # only global checkpoint can have empty dist_meta
                assert len(global_meta['meta']) == 1
            self.meta_list.append(meta)

    def read(self, name: str) -> dict:
        return torch.load(os.path.join(self.base_name, name))

    def load_meta(self) -> Tuple[List[Optional[List]], List[Optional[List]], List[Optional[List]]]:
        meta_infos = [(meta.get('dist_meta', None), meta.get('param_to_os', None), meta.get('paired_os', None))
                      for meta in self.meta_list]
        dist_meta_list, param_to_os_list, paired_os_list = zip(*meta_infos)
        return list(dist_meta_list), list(param_to_os_list), list(paired_os_list)

    def load_model(self) -> Generator[Dict[int, dict]]:
        indices = [0] * len(self.meta_list)
        while True:
            shards = {}
            for i, meta in enumerate(self.meta_list):
                model_checkpoint_names = meta.get('model', [])
                if indices[i] < len(model_checkpoint_names):
                    shards[i] = self.read(model_checkpoint_names[indices[i]])
                    indices[i] += 1
            if len(shards) > 0:
                yield shards
            else:
                break

    def load_optimizer(self) -> Generator[Dict[int, dict]]:
        indices = [0] * len(self.meta_list)
        param_groups = []
        while True:
            shards = {}
            for i, meta in enumerate(self.meta_list):
                optimizer_checkpoint_names = meta.get('optimizer', [])
                if indices[i] < len(optimizer_checkpoint_names):
                    shards[i] = self.read(optimizer_checkpoint_names[indices[i]])
                    if indices[i] == 0:
                        param_groups.append(shards[i]['param_groups'])
                    else:
                        shards[i]['param_groups'] = param_groups[i]
                    indices[i] += 1
            if len(shards) > 0:
                yield shards
            else:
                break

    def load_other(self) -> dict:
        return self.read(OTHER_CKPT_FILE_NAME)
