import gc
import logging
import os
from functools import reduce
from pathlib import Path
from typing import Iterator, Optional, OrderedDict, Tuple

import torch.nn as nn
from torch.optim import Optimizer

from .checkpoint_io_base import CheckpointIO
from .index_file import CheckpointIndexFile
from .utils import (
    get_base_filenames,
    get_shard_filename,
    has_index_file,
    is_safetensors_available,
    load_shard_state_dict,
    load_state_dict,
    load_state_dict_into_model,
    save_state_dict,
    shard_checkpoint,
)

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

    def save_sharded_model(self,
                           model: nn.Module,
                           checkpoint_path: str,
                           gather_dtensor: bool = False,
                           variant: Optional[str] = None,
                           max_shard_size: int = 1024,
                           use_safetensors: bool = False):
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
        state_dict_shard = shard_checkpoint(state_dict, max_shard_size=max_shard_size)

        weights_name, save_index_file = get_base_filenames(variant, use_safetensors)
        total_size = 0
        index_file = CheckpointIndexFile(checkpoint_path)
        for idx, shard_pair in enumerate(state_dict_shard):
            shard = shard_pair[0]
            shard_file = get_shard_filename(weights_name, idx)
            total_size = total_size + shard_pair[1]
            for key in shard.keys():
                index_file.append_weight_map(key, shard_file)

            checkpoint_file_path = os.path.join(checkpoint_path, shard_file)
            save_state_dict(shard, checkpoint_file_path, use_safetensors)

        index_file.append_meta_data("total_size", total_size)
        index_file.write_index_file(save_index_file)
        logging.info(f"The model is going to be split to checkpoint shards. "
                     f"You can find where each parameters has been saved in the "
                     f"index located at {save_index_file}.")

    def load_sharded_model(self,
                           model: nn.Module,
                           checkpoint_index_file: Path,
                           strict: bool = False,
                           use_safetensors: bool = False,
                           load_sub_module: bool = True):
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
        missing_keys = []

        for shard_file in checkpoint_files:
            state_dict = load_shard_state_dict(Path(shard_file), use_safetensors)
            load_state_dict_into_model(model, state_dict, missing_keys, strict, load_sub_module)
            del state_dict
            gc.collect()

        if strict:
            remain_keys = reduce(lambda a, b: a & b, map(set, missing_keys))
            if len(remain_keys) > 0:
                error_msgs = 'Missing key(s) in state_dict: {}. '.format(', '.join(
                    '"{}"'.format(k) for k in missing_keys))
                raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                    self.__class__.__name__, "\n\t".join(error_msgs)))
