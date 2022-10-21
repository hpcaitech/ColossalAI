import os
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, Generator, List, Optional, Tuple

import torch

from .constant import GLOBAL_META_FILE_NAME, OTHER_CKPT_FILE_NAME
from .meta import ParamDistMeta
from .utils import is_duplicated_list


class CheckpointReader(ABC):

    def __init__(self, base_name: str) -> None:
        super().__init__()
        self.base_name = base_name
        self.meta_list = []

    @abstractmethod
    def read(self, name: str) -> dict:
        pass

    @abstractmethod
    def load_meta(
            self) -> Tuple[List[Optional[Dict[str, ParamDistMeta]]], Dict[str, int], Optional[dict], Optional[dict]]:
        pass

    @abstractmethod
    def load_model(self, rank: int) -> Generator[dict, None, None]:
        pass

    @abstractmethod
    def load_models(self) -> Generator[Dict[int, dict], None, None]:
        pass

    @abstractmethod
    def load_optimizer(self, rank: int) -> Generator[dict, None, None]:
        pass

    @abstractmethod
    def load_optimizers(self) -> Generator[Dict[int, dict], None, None]:
        pass

    @abstractmethod
    def load_others(self) -> dict:
        pass


class DiskCheckpointReader(CheckpointReader):

    def __init__(self, base_name: str) -> None:
        super().__init__(base_name)
        assert os.path.isdir(base_name), f'"{base_name}" is not a directory'
        global_meta = self.read(GLOBAL_META_FILE_NAME)
        for meta_file_name in global_meta['meta']:
            meta = self.read(meta_file_name)
            if meta.get('dist_meta', None) is None:
                # only global checkpoint can have empty dist_meta
                assert len(global_meta['meta']) == 1
            self.meta_list.append(meta)

    def read(self, name: str) -> dict:
        return torch.load(os.path.join(self.base_name, name))

    def load_meta(
            self) -> Tuple[List[Optional[Dict[str, ParamDistMeta]]], Dict[str, int], Optional[dict], Optional[dict]]:
        meta_infos = [(meta.get('dist_meta', None), meta['params'], meta.get('param_to_os',
                                                                             None), meta.get('paired_os', None))
                      for meta in self.meta_list]
        dist_meta_list, params_list, param_to_os_list, paired_os_list = zip(*meta_infos)
        # reduce param_count
        param_count = Counter(p for params in params_list for p in params)
        # validate param_to_os
        assert is_duplicated_list(param_to_os_list)
        assert is_duplicated_list(paired_os_list)
        return list(dist_meta_list), param_count, param_to_os_list[0], paired_os_list[0]

    def _load_shard(self, shard_type: str, rank: int) -> Generator[dict, None, None]:
        meta = self.meta_list[rank]
        checkpoint_names = meta.get(shard_type, [])
        for name in checkpoint_names:
            yield self.read(name)

    def load_model(self, rank: int) -> Generator[dict, None, None]:
        return self._load_shard('model', rank)

    def load_models(self) -> Generator[Dict[int, dict], None, None]:
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

    def load_optimizer(self, rank: int) -> Generator[dict, None, None]:
        param_groups = None
        for shard in self._load_shard('optimizer', rank):
            if param_groups is None:
                param_groups = shard['param_groups']
            else:
                shard['param_groups'] = param_groups
            yield shard

    def load_optimizers(self) -> Generator[Dict[int, dict], None, None]:
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

    def load_others(self) -> dict:
        return self.read(OTHER_CKPT_FILE_NAME)
