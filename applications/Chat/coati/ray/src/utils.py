import torch.distributed as dist
from typing import Any, Callable, Dict, List, Optional
from coati.models.bloom import BLOOMActor, BLOOMCritic
from coati.models.gpt import GPTActor, GPTCritic
from coati.models.opt import OPTActor, OPTCritic
from coati.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
import torch
import os

def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_cuda_actor_critic_from_args(model: str, pretrained: str = None, lora_rank=0):
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


def get_strategy_from_args(strategy: str):
    if strategy == 'naive':
        strategy_ = NaiveStrategy()
    elif strategy == 'ddp':
        strategy_ = DDPStrategy()
    elif strategy == 'colossalai_gemini':
        strategy_ = ColossalAIStrategy(stage=3, placement_policy='cuda', initial_scale=2**5)
    elif strategy == 'colossalai_zero2':
        strategy_ = ColossalAIStrategy(stage=2, placement_policy='cuda')
    else:
        raise ValueError(f'Unsupported strategy "{strategy}"')
    return strategy_


def set_dist_env(env_info: Dict[str, str]):
    os.environ["RANK"] = env_info['rank']
    os.environ["LOCAL_RANK"] = env_info['local_rank']
    os.environ["WORLD_SIZE"] = env_info['world_size']
    os.environ['MASTER_PORT'] = env_info['master_port']
    os.environ['MASTER_ADDR'] = env_info['master_addr']
