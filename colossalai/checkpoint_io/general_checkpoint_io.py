from pathlib import Path

import torch.nn as nn
from torch.optim import Optimizer

from .checkpoint_io_base import CheckpointIO
from .index_file import CheckpointIndexFile
from .utils import has_index_file, load_state_dict, save_state_dict

__all__ = ['GeneralCheckpointIO']


class GeneralCheckpointIO(CheckpointIO):

    def load_sharded_model(self, model: nn.Module, index_file_path: Path, strict: bool):
        # load the index file
        index_file = CheckpointIndexFile.from_file(index_file_path)

        # iterate over the shard checkpoint files
        # and load each
        index_file.assert_no_dtensor_checkpoint()
        checkpoint_file_list, _ = index_file.get_checkpoint_fileanames()
        for shard_file in checkpoint_file_list:
            shard_checkpoint = load_state_dict(shard_file)
            model.load_state_dict(shard_checkpoint, strict=strict)

    def load_unsharded_model(self, model: nn.Module, checkpoint: str, strict: bool):
        checkpoint = load_state_dict(checkpoint)
        model.load_state_dict(checkpoint, strict=strict)

    def save_sharded_model(self, model: nn.Module, checkpoint: Path, gather_dtensor: bool, prefix: str,
                           size_per_shard: int, use_safetensors: bool):
        # TODO(FrankLeeeee): implement this method as it can be supported by Huggingface model
        raise NotImplementedError("Sharded model checkpoint is not supported yet.")

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
