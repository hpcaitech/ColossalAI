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
import socket
import multiprocessing


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def get_local_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(('8.8.8.8', 80))
        return s.getsockname()[0]


def launch_trainer(args, env_info):
    ray.init()
    # manually set environs
    os.environ["RANK"] = env_info['rank']
    os.environ["LOCAL_RANK"] = env_info['local_rank']
    os.environ["WORLD_SIZE"] = env_info['world_size']
    os.environ['MASTER_PORT'] = env_info['master_port']
    os.environ['MASTER_ADDR'] = env_info['master_addr']
    rank = int(os.environ['RANK'])

    # configure Trainer strategy
    # ! Supposed to be DDP !
    if args.trainer_strategy == 'naive':
        strategy = NaiveStrategy()
    elif args.trainer_strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.trainer_strategy == 'colossalai_gemini':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cuda', initial_scale=2**5)
    elif args.trainer_strategy == 'colossalai_zero2':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cuda')
    else:
        raise ValueError(f'Unsupported strategy "{args.trainer_strategy}"')

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
    if rank == 0:
        name = "trainer1"
    elif rank == 1:
        name = "trainer2"
    trainer_ref = DetachedPPOTrainer.options(name=name, namespace=os.environ["RAY_NAMESPACE"], num_gpus=1, max_concurrency=2).remote(
        experience_maker_holder_name_list=["maker1", "maker2"],
        strategy=strategy,
        model=args.model,
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

    # trainer send its actor and critic to experience holder.
    ray.get(trainer_ref.initialize_remote_makers.remote())

    trainer_done_ref = trainer_ref.fit.remote(num_episodes=args.num_episodes, max_timesteps=args.max_timesteps, update_timesteps=args.update_timesteps)
    ray.get(trainer_done_ref)

    # save model checkpoint after fitting
    trainer_ref.strategy_save_actor.remote(args.save_path, only_rank0=True)
    # save optimizer checkpoint on all ranks
    if args.need_optim_ckpt:
        trainer_ref.strategy_save_actor_optim.remote('actor_optim_checkpoint_prompts_%d.pt' % (torch.cuda.current_device()),
                                                     only_rank0=False)



def launch_maker(args, env_info):
    ray.init()
    os.environ["RANK"] = env_info['rank']
    os.environ["LOCAL_RANK"] = env_info['local_rank']
    os.environ["WORLD_SIZE"] = env_info['world_size']
    os.environ['MASTER_PORT'] = env_info['master_port']
    os.environ['MASTER_ADDR'] = env_info['master_addr']
    rank = int(os.environ['RANK'])

    # configure Trainer strategy
    # ! Supposed to be DDP !
    if args.maker_strategy == 'naive':
        strategy = NaiveStrategy()
    elif args.maker_strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.maker_strategy == 'colossalai_gemini':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cuda', initial_scale=2**5)
    elif args.maker_strategy == 'colossalai_zero2':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cuda')
    else:
        raise ValueError(f'Unsupported strategy "{args.maker_strategy}"')

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

    # configure Experience Maker
    if rank == 0:
        name = "maker1"
    elif rank == 1:
        name = "maker2"
    experience_holder_ref = ExperienceMakerHolder.options(name=name, namespace=os.environ["RAY_NAMESPACE"], num_gpus=1, max_concurrency=2).remote(
        detached_trainer_name_list=["trainer1", "trainer2"],
        strategy=strategy,
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
    # configure sampler
    dataset = pd.read_csv(args.prompt_path)['prompt']
    sampler = strategy.setup_sampler(dataset)

    def tokenize_fn(texts):
        # MUST padding to max length to ensure inputs of all ranks have the same length
        # Different length may lead to hang when using gemini, as different generation steps
        batch = tokenizer(texts, return_tensors='pt', max_length=96, padding='max_length', truncation=True)
        return {k: v.cuda() for k, v in batch.items()}

    num_exp_per_maker = args.num_episodes * args.max_timesteps // args.update_timesteps * args.max_epochs + 3  # +3 for fault tolerance
    maker_done_ref = experience_holder_ref.workingloop.remote(sampler, tokenize_fn, times=num_exp_per_maker)

    ray.get(maker_done_ref)

def spawn_fn(rank, args, env_info_list):
    if rank == 0 or rank == 1:
        launch_trainer(args, env_info_list[rank])
    elif rank == 2 or rank == 3:
        launch_maker(args, env_info_list[rank])
    

def main(args):
    master_addr = str(get_local_ip())
    # trainer_env_info
    trainer_port = str(get_free_port())
    env_info_trainer_1 = {'local_rank' : '0',
                          'rank' : '0',
                          'world_size' : '2',
                          'master_port' : trainer_port,
                          'master_addr' : master_addr}
    env_info_trainer_2 = {'local_rank' : '0',
                          'rank' : '1',
                          'world_size' : '2',
                          'master_port' : trainer_port,
                          'master_addr' : master_addr}
    # maker_env_info
    maker_port = str(get_free_port())
    env_info_maker_1 = {'local_rank' : '0',
                        'rank' : '0',
                        'world_size' : '2',
                        'master_port' : maker_port,
                        'master_addr' : master_addr}
    env_info_maker_2 = {'local_rank' : '0',
                        'rank' : '1',
                        'world_size' : '2',
                        'master_port': maker_port,
                        'master_addr' : master_addr}
    
    torch.multiprocessing.spawn(spawn_fn, args=(args, [env_info_trainer_1,
                                                       env_info_trainer_2,
                                                       env_info_maker_1,
                                                       env_info_maker_2]),
                                nprocs=4)


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
    main(args)
