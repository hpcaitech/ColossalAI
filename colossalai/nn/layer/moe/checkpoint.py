import torch
import torch.distributed as dist
import torch.nn as nn

from colossalai.tensor.moe_tensor.api import get_ep_group


def save_moe_model(model: nn.Module, save_path: str):
    state_dict = model.state_dict()
    if dist.get_rank() == 0:
        torch.save(state_dict, save_path)
    dist.barrier()


def load_moe_model(model: nn.Module, load_path: str):
    state_dict = torch.load(load_path)

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

    model.load_state_dict(state_dict)
