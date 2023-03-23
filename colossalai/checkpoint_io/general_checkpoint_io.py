from pathlib import Path

import torch.nn as nn
from torch.optim import Optimizer

from .checkpoint_io_base import CheckpointIO

__all__ = ['GeneralCheckpointIO']


class GeneralCheckpointIO(CheckpointIO):

    def load_model(self, model: nn.Module, checkpoint: str, strict: bool = True):
        checkpoint = Path(checkpoint)
        is_sharded = self.is_sharded_checkpoint(checkpoint)

        if not is_sharded:
            checkpoint = self.load_state_dict(checkpoint)
            model.load_state_dict(checkpoint, strict=strict)
        else:
            # find the index file
            checkpoint_path = Path(checkpoint)
            index_file_path = self.get_sharded_checkpoint_index_file(checkpoint_path)

            # iterate over the shard checkpoint files
            # and load each
            shard_files = self.get_checkpoint_shard_filenames(index_file_path)
            for shard_file in shard_files:
                shard_checkpoint = self.load_state_dict(shard_file)
                model.load_state_dict(shard_checkpoint, strict=strict)

        return model

    def save_model(self,
                   model: nn.Module,
                   checkpoint: str,
                   prefix: str = None,
                   shard: bool = False,
                   size_per_shard: int = 1024):
        checkpoint = Path(checkpoint)
        if shard:
            # TODO(FrankLeeeee): implement checkpoint saving to sharded checkpoint
            raise NotImplementedError("Not implemented yet")
        else:
            self.save_checkpoint(model.state_dict(), checkpoint)

    def load_optimizer(self, optimizer: Optimizer, checkpoint: str):
        checkpoint = Path(checkpoint)
        is_sharded = self.is_sharded_checkpoint(checkpoint)

        if not is_sharded:
            checkpoint = self.load_state_dict(checkpoint)
            optimizer.load_state_dict(checkpoint)
        else:
            # TODO(FrankLeeeee): implement checkpoint loading from sharded checkpoint
            # This is not an urgent feature, so we can leave it for later
            # let's implement this when we test large-scale models
            pass
        return optimizer

    def save_optimizer(self, optimizer: Optimizer, checkpoint: str, shard: bool = False, size_per_shard: int = 1024):
        if shard:
            # TODO(FrankLeeeee): implement checkpoint saving to sharded checkpoint
            pass
        else:
            self.save_checkpoint(optimizer.state_dict(), checkpoint)
