import argparse

import torch
import torch.distributed as dist
from coati.dataset import DataCollatorForSupervisedDataset, PromptDataset, SupervisedDataset
from coati.models.bloom import BLOOMRM, BLOOMActor, BLOOMCritic
from coati.models.gpt import GPTRM, GPTActor, GPTCritic
from coati.models.llama import LlamaActor, LlamaCritic, LlamaRM
from coati.models.opt import OPTRM, OPTActor, OPTCritic
from coati.trainer import PPOTrainer
from coati.trainer.strategies import DDPStrategy, GeminiStrategy, LowLevelZeroStrategy
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, BloomTokenizerFast, GPT2Tokenizer, LlamaTokenizer

from colossalai.nn.optimizer import HybridAdam


def main(args):
    # configure strategy
    if args.strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        strategy = GeminiStrategy(placement_policy='cuda', initial_scale=2**5)
    elif args.strategy == 'colossalai_zero2':
        strategy = LowLevelZeroStrategy(stage=2, placement_policy='cuda')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    if args.rm_path is not None:
        state_dict = torch.load(args.rm_path, map_location='cpu')

    with strategy.model_init_context():
        # configure model
        if args.model == 'gpt2':
            initial_model = GPTActor(pretrained=args.pretrain)
        elif args.model == 'bloom':
            initial_model = BLOOMActor(pretrained=args.pretrain)
        elif args.model == 'opt':
            initial_model = OPTActor(pretrained=args.pretrain)
        elif args.model == 'llama':
            initial_model = LlamaActor(pretrained=args.pretrain)
        else:
            raise ValueError(f'Unsupported actor model "{args.model}"')

        if args.rm_model is None:
            rm_model_name = args.model
        else:
            rm_model_name = args.rm_model

        if rm_model_name == 'gpt2':
            reward_model = GPTRM(pretrained=args.rm_pretrain)
        elif rm_model_name == 'bloom':
            reward_model = BLOOMRM(pretrained=args.rm_pretrain)
        elif rm_model_name == 'opt':
            reward_model = OPTRM(pretrained=args.rm_pretrain)
        elif rm_model_name == 'llama':
            reward_model = LlamaRM(pretrained=args.rm_pretrain)
        else:
            raise ValueError(f'Unsupported reward model "{rm_model_name}"')

        if args.rm_path is not None:
            reward_model.load_state_dict(state_dict)

        initial_model.to(torch.float16).to(torch.cuda.current_device())
        reward_model.to(torch.float16).to(torch.cuda.current_device())

        if args.model == 'gpt2':
            actor = GPTActor(pretrained=args.pretrain, lora_rank=args.lora_rank)
        elif args.model == 'bloom':
            actor = BLOOMActor(pretrained=args.pretrain, lora_rank=args.lora_rank)
        elif args.model == 'opt':
            actor = OPTActor(pretrained=args.pretrain, lora_rank=args.lora_rank)
        elif args.model == 'llama':
            actor = LlamaActor(pretrained=args.pretrain, lora_rank=args.lora_rank)
        else:
            raise ValueError(f'Unsupported actor model "{args.model}"')

        if rm_model_name == 'gpt2':
            critic = GPTCritic(pretrained=args.rm_pretrain, lora_rank=args.lora_rank, use_action_mask=True)
        elif rm_model_name == 'bloom':
            critic = BLOOMCritic(pretrained=args.rm_pretrain, lora_rank=args.lora_rank, use_action_mask=True)
        elif rm_model_name == 'opt':
            critic = OPTCritic(pretrained=args.rm_pretrain, lora_rank=args.lora_rank, use_action_mask=True)
        elif rm_model_name == 'llama':
            critic = LlamaCritic(pretrained=args.rm_pretrain, lora_rank=args.lora_rank, use_action_mask=True)
        else:
            raise ValueError(f'Unsupported reward model "{rm_model_name}"')

        if args.rm_path is not None:
            critic.load_state_dict(state_dict)
            del state_dict

    if args.strategy != 'colossalai_gemini':
        critic.to(torch.float16).to(torch.cuda.current_device())
        actor.to(torch.float16).to(torch.cuda.current_device())

    # configure optimizer
    if args.strategy.startswith('colossalai'):
        actor_optim = HybridAdam(actor.parameters(), lr=1e-7)
        critic_optim = HybridAdam(critic.parameters(), lr=1e-7)
    else:
        actor_optim = Adam(actor.parameters(), lr=1e-7)
        critic_optim = Adam(critic.parameters(), lr=1e-7)

    # configure tokenizer
    if args.model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'bloom':
        tokenizer = BloomTokenizerFast.from_pretrained('bigscience/bloom-560m')
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained(args.pretrain)
        tokenizer.eos_token = '<\s>'
        tokenizer.pad_token = tokenizer.unk_token
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    prompt_dataset = PromptDataset(tokenizer=tokenizer, data_path=args.prompt_dataset, max_datasets_size=16384)
    if dist.is_initialized() and dist.get_world_size() > 1:
        prompt_sampler = DistributedSampler(prompt_dataset, shuffle=True, seed=42, drop_last=True)
    else:
        prompt_sampler = None
    prompt_dataloader = DataLoader(prompt_dataset,
                                   shuffle=(prompt_sampler is None),
                                   sampler=prompt_sampler,
                                   batch_size=args.experience_batch_size)

    pretrain_dataset = SupervisedDataset(tokenizer=tokenizer,
                                         data_path=args.pretrain_dataset,
                                         max_datasets_size=16384,
                                         max_length=args.max_input_len)
    if dist.is_initialized() and dist.get_world_size() > 1:
        pretrain_sampler = DistributedSampler(pretrain_dataset, shuffle=True, seed=42, drop_last=True)
    else:
        pretrain_sampler = None
    pretrain_dataloader = DataLoader(pretrain_dataset,
                                     shuffle=(pretrain_sampler is None),
                                     sampler=pretrain_sampler,
                                     batch_size=args.ptx_batch_size,
                                     collate_fn=data_collator)

    # NOTE: For small models like opt-1.3b, reward model and initial model are not required to be parallelized.
    (actor, actor_optim), (critic, critic_optim), reward_model, initial_model = \
        strategy.prepare((actor, actor_optim), (critic, critic_optim), reward_model, initial_model)

    # configure trainer
    trainer = PPOTrainer(
        strategy,
        actor,
        critic,
        reward_model,
        initial_model,
        actor_optim,
        critic_optim,
        kl_coef=args.kl_coef,
        ptx_coef=args.ptx_coef,
        train_batch_size=args.train_batch_size,
        max_length=args.max_seq_len,
        use_cache=True,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        offload_inference_models=args.strategy != 'colossalai_gemini'
    )

    trainer.fit(prompt_dataloader=prompt_dataloader,
                pretrain_dataloader=pretrain_dataloader,
                num_episodes=args.num_episodes,
                num_collect_steps=args.num_collect_steps,
                num_update_steps=args.num_update_steps)

    # save model checkpoint after fitting
    strategy.save_model(actor, args.save_path, only_rank0=True)
    # save optimizer checkpoint on all ranks
    if args.need_optim_ckpt:
        strategy.save_optimizer(actor_optim,
                                'actor_optim_checkpoint_prompts_%d.pt' % (torch.cuda.current_device()),
                                only_rank0=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_dataset', type=str, default=None, help='path to the prompt dataset')
    parser.add_argument('--pretrain_dataset', type=str, default=None, help='path to the pretrained dataset')
    parser.add_argument('--strategy',
                        choices=['ddp', 'colossalai_gemini', 'colossalai_zero2'],
                        default='colossalai_zero2',
                        help='strategy to use')
    parser.add_argument('--model', default='gpt2', choices=['gpt2', 'bloom', 'opt', 'llama'])
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--rm_model', default=None, choices=['gpt2', 'bloom', 'opt', 'llama'])
    parser.add_argument('--rm_path', type=str, default=None)
    parser.add_argument('--rm_pretrain', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='actor_checkpoint_prompts')
    parser.add_argument('--need_optim_ckpt', type=bool, default=False)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--num_collect_steps', type=int, default=10)
    parser.add_argument('--num_update_steps', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--ptx_batch_size', type=int, default=1)
    parser.add_argument('--experience_batch_size', type=int, default=8)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument('--kl_coef', type=float, default=0.1)
    parser.add_argument('--ptx_coef', type=float, default=0.9)
    parser.add_argument('--max_input_len', type=int, default=96)
    parser.add_argument('--max_seq_len', type=int, default=128)
    args = parser.parse_args()
    main(args)
