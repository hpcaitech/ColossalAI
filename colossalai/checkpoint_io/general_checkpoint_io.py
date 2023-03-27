from pathlib import Path

import torch.nn as nn
from torch.optim import Optimizer

from .checkpoint_io_base import CheckpointIO

__all__ = ['GeneralCheckpointIO']


class GeneralCheckpointIO(CheckpointIO):

    def load_sharded_model(self, model: nn.Module, checkpoint: Path, strict: bool):
        index_file_path = self.get_sharded_checkpoint_index_file(checkpoint)

        # iterate over the shard checkpoint files
        # and load each
        shard_files = self.get_checkpoint_shard_filenames(index_file_path)
        for shard_file in shard_files:
            shard_checkpoint = self.load_state_dict(shard_file)
            model.load_state_dict(shard_checkpoint, strict=strict)

    def load_unsharded_model(self, model: nn.Module, checkpoint: Path, strict: bool):
        checkpoint = self.load_state_dict(str(checkpoint))
        model.load_state_dict(checkpoint, strict=strict)

    def save_sharded_model(self, model: nn.Module, checkpoint: Path, prefix: str, size_per_shard: int):
        # TODO(FrankLeeeee): implement this method as it can be supported by Huggingface model
        raise NotImplementedError("Sharded model checkpoint is not supported yet.")

    def save_unsharded_model(self, model: nn.Module, checkpoint: Path):
        self.save_checkpoint(model.state_dict(), checkpoint)

    def load_sharded_optimizer(self, optimizer: Optimizer, checkpoint: Path, prefix: str, size_per_shard: int):
        raise NotImplementedError("Sharded optimizer checkpoint is not supported yet.")

    def load_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: Path):
        checkpoint = self.load_state_dict(checkpoint)
        optimizer.load_state_dict(checkpoint)

    def save_sharded_optimizer(self, optimizer: Optimizer, checkpoint: Path, prefix: str, size_per_shard: int):
        raise NotImplementedError("Sharded optimizer checkpoint is not supported yet.")

    def save_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: Path):
        self.save_checkpoint(optimizer.state_dict(), checkpoint)
