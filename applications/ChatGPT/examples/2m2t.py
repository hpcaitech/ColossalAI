import argparse
from copy import deepcopy

import pandas as pd
import torch
from chatgpt.trainer import PPOTrainer, DetachedPPOTrainer
from chatgpt.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from chatgpt.experience_maker import NaiveExperienceMaker, ExperienceMakerHolder
from torch.optim import Adam
from transformers import AutoTokenizer, BloomTokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

from colossalai.nn.optimizer import HybridAdam

import ray
import os

def launch_trainer(args, env_info):
    # manually set environ
    os.environ["RANK"] = env_info.rank
    os.environ["LOCAL_RANK"] = env_info.local_rank
    os.environ["WORLD_SIZE"] = env_info.world_size
    os.environ["MASTER_ADDR"] = env_info.master_addr
    os.environ["MASTER_PORT"] = env_info.master_port
    
    
    
    # configure Trainer strategy
    if args.strategy == 'naive':
        trainer_strategy = NaiveStrategy()
    elif args.strategy == 'ddp':
        trainer_strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        trainer_strategy = ColossalAIStrategy(stage=3, placement_policy='cuda', initial_scale=2**5)
    elif args.strategy == 'colossalai_zero2':
        trainer_strategy = ColossalAIStrategy(stage=2, placement_policy='cuda')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')
    
    
def main(args):
    

    # configure Maker strategy
    maker_strategy = NaiveStrategy()