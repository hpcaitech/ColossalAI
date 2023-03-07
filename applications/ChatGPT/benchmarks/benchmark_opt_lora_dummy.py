import argparse
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn
from chatgpt.models.base import RewardModel
from chatgpt.models.opt import OPTActor, OPTCritic
from chatgpt.trainer import PPOTrainer
from chatgpt.trainer.callbacks import PerformanceEvaluator
from chatgpt.trainer.strategies import ColossalAIStrategy, DDPStrategy, Strategy
from torch.optim import Adam
from transformers import AutoTokenizer
from transformers.models.opt.configuration_opt import OPTConfig

from colossalai.nn.optimizer import HybridAdam


def get_model_numel(model: nn.Module, strategy: Strategy) -> int:
    numel = sum(p.numel() for p in model.parameters())
    if isinstance(strategy, ColossalAIStrategy) and strategy.stage == 3 and strategy.shard_init:
        numel *= dist.get_world_size()
    return numel


def preprocess_batch(samples) -> dict:
    input_ids = torch.stack(samples)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}


def print_rank_0(*args, **kwargs) -> None:
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def print_model_numel(model_dict: dict) -> None:
    B = 1024**3
    M = 1024**2
    K = 1024
    outputs = ''
    for name, numel in model_dict.items():
        outputs += f'{name}: '
        if numel >= B:
            outputs += f'{numel / B:.2f} B\n'
        elif numel >= M:
            outputs += f'{numel / M:.2f} M\n'
        elif numel >= K:
            outputs += f'{numel / K:.2f} K\n'
        else:
            outputs += f'{numel}\n'
    print_rank_0(outputs)


def get_gpt_config(model_name: str) -> OPTConfig:
    model_map = {
        '125m': OPTConfig.from_pretrained('facebook/opt-125m'),
        '350m': OPTConfig(hidden_size=1024, ffn_dim=4096, num_hidden_layers=24, num_attention_heads=16),
        '700m': OPTConfig(hidden_size=1280, ffn_dim=5120, num_hidden_layers=36, num_attention_heads=20),
        '1.3b': OPTConfig.from_pretrained('facebook/opt-1.3b'),
        '2.7b': OPTConfig.from_pretrained('facebook/opt-2.7b'),
        '3.5b': OPTConfig(hidden_size=3072, ffn_dim=12288, num_hidden_layers=32, num_attention_heads=32),
        '5.5b': OPTConfig(hidden_size=3840, ffn_dim=15360, num_hidden_layers=32, num_attention_heads=32),
        '6.7b': OPTConfig.from_pretrained('facebook/opt-6.7b'),
        '10b': OPTConfig(hidden_size=5120, ffn_dim=20480, num_hidden_layers=32, num_attention_heads=32),
        '13b': OPTConfig.from_pretrained('facebook/opt-13b'),
    }
    try:
        return model_map[model_name]
    except KeyError:
        raise ValueError(f'Unknown model "{model_name}"')


def main(args):
    if args.strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cuda', initial_scale=2**5)
    elif args.strategy == 'colossalai_gemini_cpu':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cpu', initial_scale=2**5)
    elif args.strategy == 'colossalai_zero2':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cuda')
    elif args.strategy == 'colossalai_zero2_cpu':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cpu')
    elif args.strategy == 'colossalai_zero1':
        strategy = ColossalAIStrategy(stage=1, placement_policy='cuda')
    elif args.strategy == 'colossalai_zero1_cpu':
        strategy = ColossalAIStrategy(stage=1, placement_policy='cpu')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    torch.cuda.set_per_process_memory_fraction(args.cuda_mem_frac)

    model_config = get_gpt_config(args.model)

    with strategy.model_init_context():
        actor = OPTActor(config=model_config, lora_rank=args.lora_rank).cuda()
        critic = OPTCritic(config=model_config, lora_rank=args.lora_rank).cuda()

        initial_model = deepcopy(actor).cuda()
        reward_model = RewardModel(deepcopy(critic.model), deepcopy(critic.value_head)).cuda()

    actor_numel = get_model_numel(actor, strategy)
    critic_numel = get_model_numel(critic, strategy)
    initial_model_numel = get_model_numel(initial_model, strategy)
    reward_model_numel = get_model_numel(reward_model, strategy)
    print_model_numel({
        'Actor': actor_numel,
        'Critic': critic_numel,
        'Initial model': initial_model_numel,
        'Reward model': reward_model_numel
    })
    performance_evaluator = PerformanceEvaluator(actor_numel,
                                                 critic_numel,
                                                 initial_model_numel,
                                                 reward_model_numel,
                                                 enable_grad_checkpoint=False,
                                                 ignore_episodes=1)

    if args.strategy.startswith('colossalai'):
        actor_optim = HybridAdam(actor.parameters(), lr=5e-6)
        critic_optim = HybridAdam(critic.parameters(), lr=5e-6)
    else:
        actor_optim = Adam(actor.parameters(), lr=5e-6)
        critic_optim = Adam(critic.parameters(), lr=5e-6)

    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-350m')
    tokenizer.pad_token = tokenizer.eos_token

    (actor, actor_optim), (critic, critic_optim), reward_model, initial_model = strategy.prepare(
        (actor, actor_optim), (critic, critic_optim), reward_model, initial_model)

    trainer = PPOTrainer(strategy,
                         actor,
                         critic,
                         reward_model,
                         initial_model,
                         actor_optim,
                         critic_optim,
                         max_epochs=args.max_epochs,
                         train_batch_size=args.train_batch_size,
                         experience_batch_size=args.experience_batch_size,
                         tokenizer=preprocess_batch,
                         max_length=512,
                         do_sample=True,
                         temperature=1.0,
                         top_k=50,
                         pad_token_id=tokenizer.pad_token_id,
                         eos_token_id=tokenizer.eos_token_id,
                         callbacks=[performance_evaluator])

    random_prompts = torch.randint(tokenizer.vocab_size, (1000, 400), device=torch.cuda.current_device())
    trainer.fit(random_prompts,
                num_episodes=args.num_episodes,
                max_timesteps=args.max_timesteps,
                update_timesteps=args.update_timesteps)

    print_rank_0(f'Peak CUDA mem: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='125m')
    parser.add_argument('--strategy',
                        choices=[
                            'ddp', 'colossalai_gemini', 'colossalai_gemini_cpu', 'colossalai_zero2',
                            'colossalai_zero2_cpu', 'colossalai_zero1', 'colossalai_zero1_cpu'
                        ],
                        default='ddp')
    parser.add_argument('--num_episodes', type=int, default=3)
    parser.add_argument('--max_timesteps', type=int, default=8)
    parser.add_argument('--update_timesteps', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--experience_batch_size', type=int, default=8)
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--cuda_mem_frac', type=float, default=1.0)
    args = parser.parse_args()
    main(args)
