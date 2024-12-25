import logging
import os
from functools import reduce
from pathlib import Path
from typing import Optional

import torch.nn as nn
from torch.optim import Optimizer

from colossalai.utils.safetensors import load_flat

from .checkpoint_io_base import CheckpointIO
from .index_file import CheckpointIndexFile
from .utils import (
    async_move_save_state_dict_shards,
    create_pinned_state_dict,
    get_model_base_filenames,
    get_optimizer_base_filenames,
    is_safetensors_available,
    load_param_groups_into_optimizer,
    load_shard_state_dict,
    load_state_dict,
    load_state_dict_into_model,
    load_states_into_optimizer,
    save_config_file,
    save_param_groups,
    save_state_dict,
    save_state_dict_shards,
    shard_model_checkpoint,
    shard_optimizer_checkpoint,
    sharded_optimizer_loading_epilogue,
)

__all__ = ["GeneralCheckpointIO"]


class GeneralCheckpointIO(CheckpointIO):
    """
    Checkpoint IO
    """

    def load_unsharded_model(
        self,
        model: nn.Module,
        checkpoint: str,
        strict: bool,
        low_cpu_mem_mode: bool = True,
        num_threads: int = 1,
    ):
        checkpoint = load_state_dict(checkpoint)
        if not low_cpu_mem_mode:
            checkpoint = create_pinned_state_dict(checkpoint, empty=False, num_threads=num_threads)
        model.load_state_dict(checkpoint, strict=strict)

    def save_unsharded_model(
        self, model: nn.Module, checkpoint: str, gather_dtensor: bool, use_safetensors: bool, use_async: bool = False
    ):
        state_dict = model.state_dict()

        if use_async:
            from colossalai.utils.safetensors import move_and_save

            if id(model) not in self.pinned_state_dicts:
                self.pinned_state_dicts[id(model)] = create_pinned_state_dict(state_dict)
            writer = move_and_save(checkpoint, state_dict, self.pinned_state_dicts[id(model)])
            self.async_writers.append(writer)
        else:
            # save the checkpoint
            save_state_dict(state_dict, checkpoint, use_safetensors)

    def load_sharded_optimizer(
        self,
        optimizer: Optimizer,
        index_file_path: str,
        prefix: str,
        low_cpu_mem_mode: bool = True,
        num_threads: int = 1,
    ):
        """
        Load sharded optimizer with the given path to index file.
        """

        # Read checkpoint index file.
        ckpt_index_file = CheckpointIndexFile.from_file(index_file_path)

        # Load param_groups
        param_group_path = ckpt_index_file.get_param_group_filename()
        if param_group_path is None:
            raise RuntimeError(
                f"Invalid index file path {index_file_path} for an optimizer. \
                               Lacking param group file under current directory."
            )
        id_map = load_param_groups_into_optimizer(optimizer, param_group_path)

        checkpoint_files, _ = ckpt_index_file.get_checkpoint_filenames()

        for shard_file in checkpoint_files:
            if shard_file.endswith(".safetensors"):
                state_dict = load_flat(shard_file)
            else:
                state_dict = load_shard_state_dict(Path(shard_file), use_safetensors=False)
            if not low_cpu_mem_mode:
                state_dict = create_pinned_state_dict(state_dict, empty=False, num_threads=num_threads)
            load_states_into_optimizer(optimizer, state_dict, id_map)

        sharded_optimizer_loading_epilogue(optimizer)

    def save_sharded_optimizer(
        self,
        optimizer: Optimizer,
        checkpoint: Path,
        gather_dtensor: bool,
        prefix: str,
        size_per_shard: int,
        use_async: bool = False,
    ):
        """
        Save sharded optimizer checkpoint under the given checkpointing path.
        The following files will be created under the path:
        - An index file (pytorch_optim.bin.index.json) containing a map between optimizer states and file names
        - A group file (pytorch_optim_group.bin) recording information of param_groups
        - Multiple files (pytorch_optim-000XX.bin) that store state tensors of optimizer in a sharding way
        """

        if os.path.isfile(checkpoint):
            logging.error(f"Provided path ({checkpoint}) should be a directory, not a file")
            return

        Path(checkpoint).mkdir(parents=True, exist_ok=True)

        # Offload optimizer states. States are broken into shards within max_shard_size.
        state_dict = optimizer.state_dict()
        sharded_state = shard_optimizer_checkpoint(state_dict, max_shard_size=size_per_shard)

        # Preparing file paths and index file.
        states_name, save_index_file, param_group_file = get_optimizer_base_filenames(prefix, use_safetensors=use_async)
        index_file = CheckpointIndexFile(checkpoint)

        # Store the information of param groups to param_group_file.
        index_file.append_meta_data("param_groups", param_group_file)
        group_file_path = os.path.join(checkpoint, param_group_file)
        save_param_groups(state_dict, group_file_path)

        # Save shards of optimizer states.
        # In general cases, is_master is set to True to get the right behavior.
        if use_async:
            pinned_state_dict = self.pinned_state_dicts.get(id(optimizer), None)
            total_size, new_pinned_state_dict, writers = async_move_save_state_dict_shards(
                sharded_state_dict=sharded_state,
                checkpoint=checkpoint,
                index_file=index_file,
                base_filename=states_name,
                is_master=True,
                pinned_state_dict=pinned_state_dict,
                state_preprocess=True,
            )
            self.pinned_state_dicts[id(optimizer)] = new_pinned_state_dict
            self.async_writers.extend(writers)
        else:
            total_size = save_state_dict_shards(
                sharded_state_dict=sharded_state,
                checkpoint=checkpoint,
                index_file=index_file,
                base_filename=states_name,
                is_master=True,
                use_safetensors=False,
            )

        # Wrap up index file.
        index_file.append_meta_data("total_size", total_size)
        index_file.write_index_file(save_index_file)
        logging.info(
            f"The optimizer is going to be split to checkpoint shards. "
            f"You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )

    def load_unsharded_optimizer(
        self, optimizer: Optimizer, checkpoint: Path, low_cpu_mem_mode: bool = True, num_threads: int = 1
    ):
        if checkpoint.endswith(".safetensors"):
            checkpoint = load_flat(checkpoint)
        else:
            checkpoint = load_state_dict(checkpoint)
        if not low_cpu_mem_mode:
            checkpoint = create_pinned_state_dict(checkpoint, empty=False, num_threads=num_threads)
        optimizer.load_state_dict(checkpoint)

    def save_unsharded_optimizer(
        self,
        optimizer: Optimizer,
        checkpoint: Path,
        gather_dtensor: bool,
        use_async: bool = False,
    ):
        # TODO(FrankLeeeee): handle distributed tensors
        state_dict = optimizer.state_dict()
        if use_async:
            from colossalai.utils.safetensors import _flatten_optim_state_dict, move_and_save

            flatten_state_dict, metadata = _flatten_optim_state_dict(state_dict)
            if id(optimizer) not in self.pinned_state_dicts:
                self.pinned_state_dicts[id(optimizer)] = create_pinned_state_dict(flatten_state_dict)
            writer = move_and_save(
                path=checkpoint,
                state_dict=flatten_state_dict,
                state_dict_pinned=self.pinned_state_dicts[id(optimizer)],
                metadata=metadata,
            )
            self.async_writers.append(writer)
        else:
            save_state_dict(state_dict, checkpoint, use_safetensors=False)

    def save_sharded_model(
        self,
        model: nn.Module,
        checkpoint_path: str,
        gather_dtensor: bool = False,
        prefix: Optional[str] = None,
        max_shard_size: int = 1024,
        use_safetensors: bool = False,
        use_async: bool = False,
    ):
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
        index_file = CheckpointIndexFile(checkpoint_path)

        if use_async:
            pinned_state_dict = self.pinned_state_dicts.get(id(model), None)
            total_size, new_pinned_state_dict, writers = async_move_save_state_dict_shards(
                sharded_state_dict=state_dict_shard,
                checkpoint=checkpoint_path,
                index_file=index_file,
                base_filename=weights_name,
                is_master=True,
                pinned_state_dict=pinned_state_dict,
            )
            self.pinned_state_dicts[id(model)] = new_pinned_state_dict
            self.async_writers.extend(writers)
        else:
            # Save shards of optimizer states.
            # In general cases, is_master is set to True to get the right behavior.
            total_size = save_state_dict_shards(
                sharded_state_dict=state_dict_shard,
                checkpoint=checkpoint_path,
                index_file=index_file,
                base_filename=weights_name,
                is_master=True,
                use_safetensors=use_safetensors,
            )

        index_file.append_meta_data("total_size", total_size)
        index_file.write_index_file(save_index_file)
        save_config_file(model, checkpoint_path, is_master=True)
        logging.info(
            f"The model is going to be split to checkpoint shards. "
            f"You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )

    def load_sharded_model(
        self,
        model: nn.Module,
        checkpoint_index_file: Path,
        strict: bool = False,
        use_safetensors: bool = False,
        load_sub_module: bool = True,
        low_cpu_mem_mode: bool = True,
        num_threads: int = 1,
    ):
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
            if not low_cpu_mem_mode:
                state_dict = create_pinned_state_dict(state_dict, empty=False, num_threads=num_threads)
            load_state_dict_into_model(model, state_dict, missing_keys, strict, load_sub_module)

        if strict:
            remain_keys = reduce(lambda a, b: a & b, map(set, missing_keys))
            if len(remain_keys) > 0:
                error_msgs = [
                    "Missing key(s) in state_dict: {}. ".format(", ".join('"{}"'.format(k) for k in remain_keys))
                ]
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        self.__class__.__name__, "\n\t".join(error_msgs)
                    )
                )

    def save_lora_as_pretrained(self, model: nn.Module, checkpoint: str, use_safetensors: bool = False) -> None:
        raise NotImplementedError
