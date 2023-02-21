import argparse
from copy import deepcopy

import pandas as pd
import torch
from chatgpt.nn import BLOOMActor, BLOOMCritic, GPTActor, GPTCritic, OPTActor, OPTCritic, RewardModel
from chatgpt.trainer import PPOTrainer
from chatgpt.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from torch.optim import Adam
from transformers import AutoTokenizer, BloomTokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

from colossalai.nn.optimizer import HybridAdam


def main(args):
    # configure strategy
    if args.strategy == 'naive':
        strategy = NaiveStrategy()
    elif args.strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cuda')
    elif args.strategy == 'colossalai_zero2':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cuda')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    # configure model
    with strategy.model_init_context():
        if args.model == 'gpt2':
            actor = GPTActor().cuda()
            critic = GPTCritic().cuda()
        elif args.model == 'bloom':
            actor = BLOOMActor(pretrained=args.pretrain, lora_rank=args.lora_rank).cuda()
            critic = BLOOMCritic(pretrained=args.pretrain, lora_rank=args.lora_rank).cuda()
        elif args.model == 'opt':
            actor = OPTActor(lora_rank=args.lora_rank).cuda()
            critic = OPTCritic(lora_rank=args.lora_rank).cuda()
        else:
            raise ValueError(f'Unsupported model "{args.model}"')

        initial_model = deepcopy(actor)
        reward_model = RewardModel(deepcopy(critic.model), deepcopy(critic.value_head)).cuda()

    # configure optimizer
    if args.strategy.startswith('colossalai'):
        actor_optim = HybridAdam(actor.parameters(), lr=5e-6)
        critic_optim = HybridAdam(critic.parameters(), lr=5e-6)
    else:
        actor_optim = Adam(actor.parameters(), lr=5e-6)
        critic_optim = Adam(critic.parameters(), lr=5e-6)

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

    dataset = pd.read_csv(args.prompt_path)['prompt']

    def tokenize_fn(texts):
        batch = tokenizer(texts, return_tensors='pt', max_length=96, padding=True, truncation=True)
        return {k: v.cuda() for k, v in batch.items()}

    (actor, actor_optim), (critic, critic_optim), reward_model, initial_model = strategy.prepare(
        (actor, actor_optim), (critic, critic_optim), reward_model, initial_model)

    # configure trainer
    trainer = PPOTrainer(
        strategy,
        actor,
        critic,
        reward_model,
        initial_model,
        actor_optim,
        critic_optim,
        max_epochs=args.max_epochs,
        train_batch_size=args.train_batch_size,
        tokenizer=tokenize_fn,
        max_length=128,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    trainer.fit(dataset,
                num_episodes=args.num_episodes,
                max_timesteps=args.max_timesteps,
                update_timesteps=args.update_timesteps)
    # save model checkpoint after fitting on only rank0
    strategy.save_model(actor, 'actor_checkpoint_prompts.pt', only_rank0=True)
    # save optimizer checkpoint on all ranks
    strategy.save_optimizer(actor_optim,
                            'actor_optim_checkpoint_prompts_%d.pt' % (torch.cuda.current_device()),
                            only_rank0=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt_path')
    parser.add_argument('--strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2'],
                        default='naive')
    parser.add_argument('--model', default='gpt2', choices=['gpt2', 'bloom', 'opt'])
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--max_timesteps', type=int, default=10)
    parser.add_argument('--update_timesteps', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    args = parser.parse_args()
    main(args)
