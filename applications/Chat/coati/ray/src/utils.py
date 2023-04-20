import torch.distributed as dist
from typing import Any, Callable, Dict, List, Optional
from coati.models.bloom import BLOOMActor, BLOOMCritic, BLOOMRM
from coati.models.gpt import GPTActor, GPTCritic, GPTRM
from coati.models.opt import OPTActor, OPTCritic, OPTRM
from coati.models.roberta import RoBERTaRM, RoBERTaActor, RoBERTaCritic
from coati.models.llama import LlamaActor, LlamaCritic, LlamaRM

from coati.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
import torch
import os


def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_actor_from_args(model: str, pretrained: str = None, lora_rank = 0):
    if model == 'gpt2':
        actor = GPTActor(pretrained=pretrained, lora_rank=lora_rank)
    elif model == 'bloom':
        actor = BLOOMActor(pretrained=pretrained, lora_rank=lora_rank)
    elif model == 'opt':
        actor = OPTActor(pretrained=pretrained, lora_rank=lora_rank)
    elif model == 'llama':
        actor = LlamaActor(pretrained=pretrained, lora_rank=lora_rank)
    elif model == 'roberta':
        actor = RoBERTaActor(pretrained=pretrained, lora_rank=lora_rank)
    else:
        raise ValueError(f'Unsupported actor model "{model}"')
    return actor

def get_critic_from_args(model: str, pretrained: str = None, lora_rank = 0):
    if model == 'gpt2':
        critic = GPTCritic(pretrained=pretrained, lora_rank=lora_rank, use_action_mask=True)
    elif model == 'bloom':
        critic = BLOOMCritic(pretrained=pretrained, lora_rank=lora_rank, use_action_mask=True)
    elif model == 'opt':
        critic = OPTCritic(pretrained=pretrained, lora_rank=lora_rank, use_action_mask=True)
    elif model == 'llama':
        critic = LlamaCritic(pretrained=pretrained, lora_rank=lora_rank, use_action_mask=True)
    elif model == 'roberta':
        critic = RoBERTaCritic(pretrained=pretrained, lora_rank=lora_rank, use_action_mask=True)
    else:
        raise ValueError(f'Unsupported reward model "{model}"')
    return critic

def get_reward_model_from_args(model: str, pretrained: str = None):
    if model == 'gpt2':
        reward_model = GPTRM(pretrained=pretrained)
    elif model == 'bloom':
        reward_model = BLOOMRM(pretrained=pretrained)
    elif model == 'opt':
        reward_model = OPTRM(pretrained=pretrained)
    elif model == 'llama':
        reward_model = LlamaRM(pretrained=pretrained)
    elif model == 'roberta':
        reward_model = RoBERTaRM(pretrained=pretrained)
    else:
        raise ValueError(f'Unsupported reward model "{model}"')
    return reward_model

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


from transformers import AutoTokenizer, BloomTokenizerFast, GPT2Tokenizer, LlamaTokenizer, RobertaTokenizer
from coati.utils import prepare_llama_tokenizer_and_embedding

def get_tokenizer_from_args(model: str, **kwargs):
    if model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif model == 'bloom':
        tokenizer = BloomTokenizerFast.from_pretrained('bigscience/bloom-560m')
    elif model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    elif model == 'llama':
        pretrain_path = kwargs["pretrain"]
        tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
    elif model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    else:
        raise ValueError(f'Unsupported model "{model}"')

    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def set_dist_env(env_info: Dict[str, str]):
    os.environ["RANK"] = env_info['rank']
    os.environ["LOCAL_RANK"] = env_info['local_rank']
    os.environ["WORLD_SIZE"] = env_info['world_size']
    os.environ['MASTER_PORT'] = env_info['master_port']
    os.environ['MASTER_ADDR'] = env_info['master_addr']


def state_dict_to(state_dict: Dict[str, Any], dtype: torch.dtype = torch.float16, device: torch.device = torch.device('cpu')):
    '''
        keep state_dict intact
    '''
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k] = v.to(dtype = dtype, device = device)
    return new_state_dict