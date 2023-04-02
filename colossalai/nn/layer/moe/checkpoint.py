import torch
import torch.distributed as dist
import torch.nn as nn

from .experts import MoeExperts


def save_moe_model(model: nn.Module, save_path: str):
    state_dict = model.state_dict()
    if dist.get_rank() == 0:
        torch.save(state_dict, save_path)
    dist.barrier()


def load_moe_model(model: nn.Module, load_path: str):
    state_dict = torch.load(load_path)

    for prefix, module in model.named_modules():
        if prefix.endswith('.moe_layer.experts'):
            # this module should be an Experts instance
            assert isinstance(module, MoeExperts)

            ep_rank = dist.get_rank(module.dist_info.ep_group)
            num_local = module.num_local_experts
            for i in range(num_local):
                expert_id = ep_rank * num_local + i
                for name, _ in module.experts[i].named_parameters():
                    cur_key = f'{prefix}.experts.{i}.{name}'
                    param_key = f'{prefix}.experts.{expert_id}.{name}'
                    load_param = state_dict[param_key]
                    state_dict[cur_key] = load_param

            for name, _ in module.experts[0].named_parameters():
                pop_pre = f'{prefix}.experts.'
                pop_suf = f'.{name}'
                for i in range(num_local, module.num_total_experts):
                    pop_key = f'{pop_pre}{i}{pop_suf}'
                    state_dict.pop(pop_key)

    model.load_state_dict(state_dict)
