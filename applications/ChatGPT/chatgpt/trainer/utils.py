import torch.distributed as dist
from chatgpt.models.bloom import BLOOMActor, BLOOMCritic
from chatgpt.models.gpt import GPTActor, GPTCritic
from chatgpt.models.opt import OPTActor, OPTCritic
import torch
def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_cuda_actor_critic_from_args(model:str, pretrained: str = None, lora_rank=0):
    if model == 'gpt2':
        actor = GPTActor(pretrained=pretrained, lora_rank=lora_rank).to(torch.cuda.current_device())
        critic = GPTCritic(pretrained=pretrained, lora_rank=lora_rank).to(torch.cuda.current_device())
    elif model == 'bloom':
        actor = BLOOMActor(pretrained=pretrained, lora_rank=lora_rank).to(torch.cuda.current_device())
        critic = BLOOMCritic(pretrained=pretrained, lora_rank=lora_rank).to(torch.cuda.current_device())
    elif model == 'opt':
        actor = OPTActor(pretrained=pretrained, lora_rank=lora_rank).to(torch.cuda.current_device())
        critic = OPTCritic(pretrained=pretrained, lora_rank=lora_rank).to(torch.cuda.current_device())
    else:
        raise ValueError(f'Unsupported model "{model}"')
    return actor, critic