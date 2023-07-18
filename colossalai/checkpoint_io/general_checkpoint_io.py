import gc
import logging
import os
from functools import reduce
from pathlib import Path
from typing import Iterator, Optional, OrderedDict, Tuple

import torch.nn as nn
from torch.optim import Optimizer

from colossalai.interface import OptimizerWrapper

from .checkpoint_io_base import CheckpointIO
from .index_file import CheckpointIndexFile
from .utils import (
    get_model_base_filenames,
    get_optimizer_base_filenames,
    get_shard_filename,
    has_index_file,
    is_safetensors_available,
    load_param_groups_into_optimizer,
    load_shard_state_dict,
    load_state_dict,
    load_state_dict_into_model,
    load_states_into_optimizer,
    save_param_groups,
    save_state_dict,
    shard_model_checkpoint,
    shard_optimizer_checkpoint,
    sharded_optimizer_loading_epilogue,
    unwrap_optimizer,
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

    def load_sharded_optimizer(self, optimizer: Optimizer, index_file_path: str, prefix: str):
        """
        Load sharded optimizer with the given path to index file.
        """

        # If optimizer is wrapped, unwrap it.
        if isinstance(optimizer, OptimizerWrapper):
            optimizer = unwrap_optimizer(optimizer)

        # Read checkpoint index file.
        ckpt_index_file = CheckpointIndexFile.from_file(index_file_path)

        # Load param_groups
        param_group_path = ckpt_index_file.get_param_group_filename()
        if param_group_path is None:
            raise RuntimeError(f'Invalid index file path {index_file_path} for an optimizer. \
                               Lacking param group file under current directory.')
        id_map = load_param_groups_into_optimizer(optimizer, param_group_path)

        checkpoint_files, _ = ckpt_index_file.get_checkpoint_filenames()

        for shard_file in checkpoint_files:
            state_dict = load_shard_state_dict(Path(shard_file), use_safetensors=False)
            load_states_into_optimizer(optimizer, state_dict, id_map)
            del state_dict
            gc.collect()

        sharded_optimizer_loading_epilogue(optimizer)

    def save_sharded_optimizer(
        self,
        optimizer: Optimizer,
        checkpoint: Path,
        gather_dtensor: bool,
        prefix: str,
        size_per_shard: int,
    ):
        """
        Save sharded optimizer checkpoint under the given checkpointing path.
        The following files will be created under the path:
        - An index file (pytorch_optim.bin.index.json) containing a map between optimizer states and file names
        - A group file (pytorch_optim_group.bin) recording information of param_groups
        - Multiple files (pytorch_optim-000XX.bin) that store state tensors of optimizer in a sharding way
        """

        # If optimizer is wrapped, unwrap it.
        if isinstance(optimizer, OptimizerWrapper):
            optimizer = unwrap_optimizer(optimizer)

        if os.path.isfile(checkpoint):
            logging.error(f"Provided path ({checkpoint}) should be a directory, not a file")
            return

        Path(checkpoint).mkdir(parents=True, exist_ok=True)

        # Offload optimizer states. States are broken into shards within max_shard_size.
        state_dict = optimizer.state_dict()
        sharded_state = shard_optimizer_checkpoint(state_dict, max_shard_size=size_per_shard)

        # Preparing file paths and index file.
        states_name, save_index_file, param_group_file = get_optimizer_base_filenames(prefix)
        index_file = CheckpointIndexFile(checkpoint)

        # Store the information of param groups to param_group_file.
        index_file.append_meta_data("param_groups", param_group_file)
        group_file_path = os.path.join(checkpoint, param_group_file)
        save_param_groups(state_dict, group_file_path)

        # Save shards of optimizer states.
        total_size = 0
        for idx, shard_pair in enumerate(sharded_state):
            shard, current_size = shard_pair
            shard_file = get_shard_filename(states_name, idx)
            total_size = total_size + current_size
            for key in shard.keys():
                index_file.append_weight_map(key, shard_file)
            checkpoint_file_path = os.path.join(checkpoint, shard_file)
            save_state_dict(shard, checkpoint_file_path, use_safetensors=False)

        # Wrap up index file.
        index_file.append_meta_data("total_size", total_size)
        index_file.write_index_file(save_index_file)
        logging.info(f"The optimizer is going to be split to checkpoint shards. "
                     f"You can find where each parameters has been saved in the "
                     f"index located at {save_index_file}.")

    def load_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: Path):
        checkpoint = load_state_dict(checkpoint)
        optimizer.load_state_dict(checkpoint)

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
                           prefix: Optional[str] = None,
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
        state_dict_shard = shard_model_checkpoint(state_dict, max_shard_size=max_shard_size)

        weights_name, save_index_file = get_model_base_filenames(prefix, use_safetensors)
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
        checkpoint_files, _ = ckpt_index_file.get_checkpoint_filenames()
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
