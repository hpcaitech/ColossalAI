import argparse
from copy import deepcopy

import torch
from chatgpt.nn import BLOOMActor, BLOOMCritic, GPTActor, GPTCritic, OPTActor, OPTCritic, RewardModel
from chatgpt.trainer.ppo_deepspeed import DeepSpeedPPOTrainer
from chatgpt.trainer.strategies import DDPStrategy, DeepspeedStrategy, NaiveStrategy
from torch.optim import Adam
from transformers import AutoTokenizer, BloomTokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer


def preprocess_batch(samples):
    input_ids = torch.stack(samples)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}


def main(args):
    # configure strategy
    if args.strategy == 'deepspeed':
        strategy = DeepspeedStrategy(stage=args.stage,
                                     train_batch_size=args.train_batch_size,
                                     offload_optimizer=args.offload_optimizer,
                                     offload_param=args.offload_param)
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    strategy.setup_distributed()

    # configure model
    with strategy.model_init_context():
        if args.model == 'gpt2':
            actor = GPTActor().cuda()
            critic = GPTCritic().cuda()
        elif args.model == 'bloom':
            actor = BLOOMActor(pretrained=args.pretrain, lora_rank=args.lora_rank).cuda()
            critic = BLOOMCritic(pretrained=args.pretrain, lora_rank=args.lora_rank).cuda()
        elif args.model == 'opt':
            actor = OPTActor().cuda()
            critic = OPTCritic().cuda()
        else:
            raise ValueError(f'Unsupported model "{args.model}"')

        initial_model = deepcopy(actor).cuda()
        reward_model = RewardModel(deepcopy(critic.model), deepcopy(critic.value_head)).cuda()

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

    # configure trainer
    trainer = DeepSpeedPPOTrainer(
        strategy,
        actor,
        critic,
        reward_model,
        initial_model,
        max_epochs=args.max_epochs,
        train_batch_size=args.train_batch_size,
        tokenizer=preprocess_batch,
        max_length=128,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    random_prompts = torch.randint(tokenizer.vocab_size, (1000, 64), device=torch.cuda.current_device())
    trainer.fit(random_prompts,
                num_episodes=args.num_episodes,
                max_timesteps=args.max_timesteps,
                update_timesteps=args.update_timesteps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', choices=['deepspeed'], default='deepspeed')
    parser.add_argument('--stage', type=int, default=2)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--offload_optimizer', type=str, default='none')
    parser.add_argument('--offload_param', type=str, default='none')
    parser.add_argument('--model', type=str, default='gpt2', choices=['gpt2', 'bloom', 'opt'])
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--max_timesteps', type=int, default=10)
    parser.add_argument('--update_timesteps', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    main(args)
