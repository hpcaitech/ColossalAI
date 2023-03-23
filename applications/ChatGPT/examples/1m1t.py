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

# TODO: update maker actor/critic


def main(args):
    # configure strategy
    if args.strategy == 'naive':
        strategy = NaiveStrategy()
    elif args.strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cuda', initial_scale=2**5)
    elif args.strategy == 'colossalai_zero2':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cuda')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

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
    trainer_ref = DetachedPPOTrainer.options(name="trainer1", num_gpus=1, max_concurrency=3).remote(
        experience_maker_holder_name_list=["maker1"],
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

    def tokenize_fn(texts):
        # MUST padding to max length to ensure inputs of all ranks have the same length
        # Different length may lead to hang when using gemini, as different generation steps
        batch = tokenizer(texts, return_tensors='pt', max_length=96, padding='max_length', truncation=True)
        return {k: v.cuda() for k, v in batch.items()}

    # configure Experience Maker
    experience_holder_ref = ExperienceMakerHolder.options(name="maker1", num_gpus=1, max_concurrency=2).remote(
        detached_trainer_name_list=["trainer1"],
        strategy=strategy,
        model=args.model,
        pretrained=args.pretrain,
        lora_rank=args.lora_rank,
        kl_coef=0.1,
        experience_batch_size=args.experience_batch_size,
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

    print("waiting for trainer...")
    ray.get(trainer_ref.ready.remote())
    print("...ready")

    trainer_done_ref = trainer_ref.fit.remote(num_episodes=args.num_episodes, max_timesteps=args.max_timesteps, update_timesteps=args.update_timesteps)
    maker_done_ref = experience_holder_ref.workingloop.remote(sampler, tokenize_fn, times=args.num_episodes * args.max_timesteps * args.update_timesteps + 3)

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
    parser.add_argument('--strategy',
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
    ray.init()
    main(args)
