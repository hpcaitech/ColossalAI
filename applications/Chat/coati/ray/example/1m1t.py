import argparse
from copy import deepcopy

import pandas as pd
import torch
from coati.trainer import PPOTrainer


from coati.ray.src.experience_maker_holder import ExperienceMakerHolder
from coati.ray.src.detached_trainer_ppo import DetachedPPOTrainer

from coati.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from coati.experience_maker import NaiveExperienceMaker
from torch.optim import Adam
from transformers import AutoTokenizer, BloomTokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

from colossalai.nn.optimizer import HybridAdam

import ray
import os
import socket

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def get_local_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(('8.8.8.8', 80))
        return s.getsockname()[0]
    
def main(args):
    master_addr = str(get_local_ip())
    # trainer_env_info
    trainer_port = str(get_free_port())
    env_info_trainer = {'local_rank' : '0',
                          'rank' : '0',
                          'world_size' : '1',
                          'master_port' : trainer_port,
                          'master_addr' : master_addr}
    
    # maker_env_info
    maker_port = str(get_free_port())
    env_info_maker = {'local_rank' : '0',
                        'rank' : '0',
                        'world_size' : '1',
                        'master_port' : maker_port,
                        'master_addr' : master_addr}

    # configure tokenizer
    if args.model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'bloom':
        tokenizer = BloomTokenizerFast.from_pretrained(args.pretrain)
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    # configure Trainer
    trainer_ref = DetachedPPOTrainer.options(name="trainer1", num_gpus=1, max_concurrency=2).remote(
        experience_maker_holder_name_list=["maker1"],
        strategy=args.trainer_strategy,
        model=args.model,
        env_info = env_info_trainer,
        pretrained=args.pretrain,
        lora_rank=args.lora_rank,
        train_batch_size=args.train_batch_size,
        buffer_limit=16,
        experience_batch_size=args.experience_batch_size,
        max_epochs=args.max_epochs,
        #kwargs:
        max_length=128,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        debug=args.debug,
    )

    # configure Experience Maker
    experience_holder_ref = ExperienceMakerHolder.options(name="maker1", num_gpus=1, max_concurrency=2).remote(
        detached_trainer_name_list=["trainer1"],
        strategy=args.maker_strategy,
        env_info = env_info_maker,
        experience_batch_size=args.experience_batch_size,
        kl_coef=0.1,
        #kwargs:
        max_length=128,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        debug=args.debug,
    )

    # trainer send its actor and critic to experience holders.
    ray.get(trainer_ref.initialize_remote_makers.remote())

    # configure sampler
    dataset = pd.read_csv(args.prompt_path)['prompt']

    def tokenize_fn(texts):
        # MUST padding to max length to ensure inputs of all ranks have the same length
        # Different length may lead to hang when using gemini, as different generation steps
        batch = tokenizer(texts, return_tensors='pt', max_length=96, padding='max_length', truncation=True)
        return {k: v.cuda() for k, v in batch.items()}

    trainer_done_ref = trainer_ref.fit.remote(num_episodes=args.num_episodes, max_timesteps=args.max_timesteps, update_timesteps=args.update_timesteps)
    num_exp_per_maker = args.num_episodes * args.max_timesteps // args.update_timesteps * args.max_epochs + 3 # +3 for fault tolerance
    maker_done_ref = experience_holder_ref.workingloop.remote(dataset, tokenize_fn, times=num_exp_per_maker)
    
    ray.get([trainer_done_ref, maker_done_ref])

    # save model checkpoint after fitting
    trainer_ref.strategy_save_actor.remote(args.save_path, only_rank0=True)
    # save optimizer checkpoint on all ranks
    if args.need_optim_ckpt:
        trainer_ref.strategy_save_actor_optim.remote('actor_optim_checkpoint_prompts_%d.pt' % (torch.cuda.current_device()),
                                                     only_rank0=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt_path')
    parser.add_argument('--trainer_strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2'],
                        default='naive')
    parser.add_argument('--maker_strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2'],
                        default='naive')
    parser.add_argument('--model', default='gpt2', choices=['gpt2', 'bloom', 'opt'])
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='actor_checkpoint_prompts.pt')
    parser.add_argument('--need_optim_ckpt', type=bool, default=False)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--max_timesteps', type=int, default=10)
    parser.add_argument('--update_timesteps', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--experience_batch_size', type=int, default=8)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    ray.init(namespace=os.environ["RAY_NAMESPACE"])
    main(args)
