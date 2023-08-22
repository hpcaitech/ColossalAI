from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer

from colossalai.checkpoint_io import CheckpointIO
from colossalai.tensor.moe_tensor.api import get_ep_group


class MoeCheckpintIO(CheckpointIO):

    def __init__(self) -> None:
        super().__init__()

    def load_unsharded_model(self, model: nn.Module, checkpoint: str, strict: bool):
        state_dict = torch.load(checkpoint)
        for name, module in model.named_parameters():
            if '.experts.' in name:
                ep_rank = dist.get_rank(get_ep_group(module))
                ep_size = dist.get_world_size(get_ep_group(module))
                for rank in range(ep_size):
                    new_name = name.replace('.experts.', f'.experts.{rank}.')
                    if rank == ep_rank:
                        state_dict[name] = state_dict.pop(new_name)
                    else:
                        state_dict.pop(new_name)

        model.load_state_dict(state_dict, strict=strict)

    def save_unsharded_model(self, model: nn.Module, checkpoint: str, gather_dtensor: bool, use_safetensors: bool):
        state_dict = model.state_dict()
        if dist.get_rank() == 0:
            torch.save(state_dict, checkpoint)
        dist.barrier()

    def load_sharded_model(self, model: nn.Module, index_file_path: str, strict: bool):
        raise NotImplementedError()

    def save_sharded_model(self, model: nn.Module, checkpoint: str, gather_dtensor: bool, prefix: Optional[str],
                           size_per_shard: int, use_safetensors: bool):
        raise NotImplementedError()

    # ========================================================
    # Abstract methods for optimizer loading/saving implementation
    # ========================================================

    def load_sharded_optimizer(self, optimizer: Optimizer, index_file_path: str, prefix: str):
        raise NotImplementedError()

    def load_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: Path):
        raise NotImplementedError()

    def save_sharded_optimizer(self, optimizer: Optimizer, checkpoint: Path, gather_dtensor: bool, prefix: str,
                               size_per_shard: int):
        raise NotImplementedError()

    def save_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: Path, gather_dtensor: bool):
        raise NotImplementedError()
