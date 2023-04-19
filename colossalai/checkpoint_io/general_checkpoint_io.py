from pathlib import Path

import torch.nn as nn
from torch.optim import Optimizer
import logging
import os
import json
import gc
from typing import Optional, Iterator, OrderedDict

from .checkpoint_io_base import CheckpointIO
from .index_file import CheckpointIndexFile
from .utils import (
    has_index_file, 
    load_state_dict, 
    save_state_dict, 
    is_safetensors_available,
    shard_checkpoint,
    load_shard_state_dict,
    load_state_dict_into_model,
    add_variant
    )
from .utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME

# from checkpoint_io_base import CheckpointIO
# from index_file import CheckpointIndexFile
# from utils import (
#     has_index_file, 
#     load_state_dict, 
#     save_state_dict, 
#     is_safetensors_available,
#     shard_checkpoint,
#     load_shard_state_dict,
#     load_state_dict_into_model,
#     build_index,
#     write_model_files
#     )


__all__ = ['GeneralCheckpointIO']


class GeneralCheckpointIO(CheckpointIO):
    """
    Checkpoint IO
    """
    def load_unsharded_model(self, model: nn.Module, checkpoint: str, strict: bool):
        checkpoint = load_state_dict(checkpoint)
        model.load_state_dict(checkpoint, strict=strict)

    def save_unsharded_model(self, model: nn.Module, checkpoint: str, gather_dtensor: bool, use_safetensors: bool):
        state_dict = model.state_dict()

        # TODO(FrankLeeeee): add support for gather_dtensor
        if gather_dtensor:
            pass

        # save the checkpoint
        save_state_dict(state_dict, checkpoint, use_safetensors)

    def load_sharded_optimizer(self, optimizer: Optimizer, checkpoint: Path, prefix: str, size_per_shard: int):
        raise NotImplementedError("Sharded optimizer checkpoint is not supported yet.")

    def load_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: Path):
        checkpoint = load_state_dict(checkpoint)
        optimizer.load_state_dict(checkpoint)

    def save_sharded_optimizer(
        self,
        optimizer: Optimizer,
        checkpoint: Path,
        gather_dtensor: bool,
        prefix: str,
        size_per_shard: int,
    ):
        raise NotImplementedError("Sharded optimizer checkpoint is not supported yet.")

    def save_unsharded_optimizer(
        self,
        optimizer: Optimizer,
        checkpoint: Path,
        gather_dtensor: bool,
    ):
        # TODO(FrankLeeeee): handle distributed tensors
        save_state_dict(optimizer.state_dict(), checkpoint, use_safetensors=False)


    def save_sharded_model(self, model: nn.Module, checkpoint_path: str, gather_dtensor:bool = False, 
                           variant: Optional[str] = None, max_shard_size: int = 1024, use_safetensors: bool = False):
        """ 
        implement this method as it can be supported by Huggingface model,
        save shard model, save model to multiple files
        """
        if os.path.isfile(checkpoint_path):
            logging.error(f"Provided path ({checkpoint_path}) should be a directory, not a file")
            return
        
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        
        # shard checkpoint
        state_dict = model.state_dict()
        sharded_state_dicts, total_size = shard_checkpoint(state_dict, max_shard_size=max_shard_size)
        # let's build the index
        shards, shards_index = build_index(sharded_state_dicts, total_size, use_safetensors, variant)
        write_model_files(shards, shards_index, checkpoint_path, use_safetensors)


    def load_sharded_model(self, model: nn.Module, checkpoint_index_file: Path, strict: bool = False, use_safetensors: bool = False):
        """
        load shard model, load model from multiple files
        """
        use_safetensors = False
        if "safetensors" in checkpoint_index_file.name:
            use_safetensors = True

        if use_safetensors and not is_safetensors_available():
            raise ImportError("`safe_serialization` requires the `safetensors` library: `pip install safetensors`.")
        
        # read checkpoint index file
        ckpt_index_file = CheckpointIndexFile.from_file(checkpoint_index_file)
        checkpoint_files, _ = ckpt_index_file.get_checkpoint_fileanames()
        missing_keys = ckpt_index_file.get_all_param_names()

        for shard_file in checkpoint_files:
            state_dict = load_shard_state_dict(Path(shard_file), use_safetensors)
            load_state_dict_into_model(model, state_dict, missing_keys, strict)
            del state_dict
            gc.collect()

        if strict and len(missing_keys) > 0:
            error_msgs = 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys))
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               self.__class__.__name__, "\n\t".join(error_msgs)))

    def save_gemini_shard_ckp(self, state_dict_shard: Iterator[OrderedDict], checkpoint_path: str, gather_dtensor: bool = False, variant: Optional[str] = None, use_safetensors: bool = False):
        # gather all shards
        sharded_state_dicts = []
        total_size = 0
        for shard, s_size in state_dict_shard:
            sharded_state_dicts = sharded_state_dicts.append(shard)
            total_size = total_size + s_size

        shards, shards_index = build_index(sharded_state_dicts, total_size, use_safetensors, variant)
        write_model_files(shards, shards_index, checkpoint_path, use_safetensors)

