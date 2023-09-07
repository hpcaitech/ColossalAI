from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer

from colossalai.checkpoint_io import GeneralCheckpointIO
from colossalai.tensor.moe_tensor.api import get_dp_rank, get_ep_group, get_ep_rank, get_ep_size, is_moe_tensor


class MoeCheckpintIO(GeneralCheckpointIO):

    def __init__(self) -> None:
        super().__init__()

    def load_unsharded_model(self, model: nn.Module, checkpoint: str, strict: bool):
        state_dict = torch.load(checkpoint)
        for name, param in state_dict.items():
            if '.experts.' in name:
                model_param = dict(model.named_parameters())[name]
                if is_moe_tensor(model_param):
                    ep_rank = get_ep_rank(model_param)
                    ep_size = get_ep_size(model_param)
                    expert_num = param.shape[0] // ep_size
                    assert param.shape[0] % ep_size == 0
                    param = param[ep_rank * expert_num:(ep_rank + 1) * expert_num]
                    state_dict[name] = param

        model.load_state_dict(state_dict, strict=strict)

    def save_unsharded_model(self, model: nn.Module, checkpoint: str, gather_dtensor: bool, use_safetensors: bool):
        state_dict = model.state_dict()
        for name, param in model.named_parameters():
            if '.experts.' in name and is_moe_tensor(param):
                ep_group = get_ep_group(param)
                ep_rank = get_ep_rank(param)
                ep_size = get_ep_size(param)
                dp_rank = get_dp_rank(param)
                if dp_rank == 0:
                    param = param.data.cuda()
                    all_param = [deepcopy(param) for _ in range(ep_size)]
                    # gather param from every ep rank
                    dist.all_gather(all_param, param, group=ep_group)
                    if ep_rank == 0:
                        assert dist.get_rank() == 0
                        all_param = torch.cat(all_param, dim=0)
                        state_dict[name] = all_param.cpu()
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
