import argparse

import pandas as pd
import torch
import torch.distributed as dist
from coati.dataset import DataCollatorForSupervisedDataset, PromptDataset, SupervisedDataset
from coati.models.bloom import BLOOMRM, BLOOMCritic
from coati.models.gpt import GPTRM, GPTActor, GPTCritic
from coati.models.llama import LlamaActor, LlamaCritic, LlamaRM
from coati.models.opt import OPTRM, OPTActor, OPTCritic
from trainer import PPOTrainer
from coati.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from coati.utils import prepare_llama_tokenizer_and_embedding
from easy_dataset import EasyPromptsDataset, EasySupervisedDataset
from easy_models import BLOOMActor,ChatGlmActor,ChatGLMRM,ChatGLMCritic
from peft import PeftModel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, BloomTokenizerFast

from colossalai.nn.optimizer import HybridAdam


def main(args):
    # configure strategy
    if args.strategy == 'naive':
        strategy = NaiveStrategy()
    elif args.strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cpu', initial_scale=2**5)
    elif args.strategy == 'colossalai_zero2':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cpu')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')


    # configure model
    if args.model == 'bloom':
        # initial_model = BLOOMActor(pretrained=args.pretrain)
        print('Using peft lora to load Bloom model as inital_model')
        initial_model = BLOOMActor(pretrained=args.pretrain, lora_path=args.sft_lora_path)
        #first offload the model to cpu and half precision
        initial_model = initial_model.half().cpu()
        print('Using peft lora to load Bloom model as initial_model (Done)')
    elif args.model == 'chatglm':
        initial_model = ChatGlmActor(pretrained=args.pretrain, lora_path=args.sft_lora_path)
        initial_model = initial_model.half().cpu()
    else:
        raise ValueError(f'Unsupported actor model "{args.model}"')
    
    #print current cuda memory usage
    print(f'Step 0. load initial model. Current cuda memory usage: {torch.cuda.memory_allocated()/1024/1024} MB')

    #we must make rm_model the same as model because in PPO stage, the Actor's output ids must be the from the same tokenizer that reward model has
    #TODO:: I'll change the bloom model to the same implementation as ChatGLMM later
    rm_model_name = args.model
    if rm_model_name == 'bloom':
        print("load bloom reward model ", args.pretrain)
        reward_model = BLOOMRM(pretrained=args.pretrain)
        #first offload the model to cpu and half precision
        reward_model = reward_model.half().cpu()
        print("load bloom reward model (Done) ")
    elif rm_model_name == 'chatglm':
        print("load chatglm reward model ", args.pretrain," with lora path ",args.rm_lora_path)
        reward_model = ChatGLMRM(pretrained=args.pretrain,lora_path=args.rm_lora_path)
        #first offload the model to cpu and half precision
        reward_model = reward_model.half().cpu()
    else:
        raise ValueError(f'Unsupported reward model "{rm_model_name}"')
    #print current cuda memory usage
    print(f'Step 1. load reward model. Current cuda memory usage: {torch.cuda.memory_allocated()/1024/1024} MB')


    with strategy.model_init_context():
        if args.model == 'bloom':
            # actor = BLOOMActor(pretrained=args.pretrain, lora_rank=args.lora_rank)
            print('Using peft lora to load Bloom model as Actor')
            actor = BLOOMActor(pretrained=args.pretrain, lora_path=args.sft_lora_path)
            #first offload the model to cpu and half precision
            actor = actor.half().cpu()
            print('Using peft lora to load Bloom model as Actor (Done)')
        elif args.model == 'chatglm':
            print(f'load glm actor model from {args.pretrain} with loar path {args.sft_lora_path}')
            actor = ChatGlmActor(pretrained=args.pretrain, lora_path=args.sft_lora_path)
            actor = initial_model.half().cpu()
        else:
            raise ValueError(f'Unsupported actor model "{args.model}"')
        #print current cuda memory usage
        print(f'Step 2. actor model is loaded. Current cuda memory usage: {torch.cuda.memory_allocated()/1024/1024} MB')
        if rm_model_name == 'bloom':
            print("load bloom critic ", args.rm_pretrain, " lora_rank ", args.lora_rank, " use_action_mask ", True)
            critic = BLOOMCritic(pretrained=args.rm_pretrain, lora_rank=args.lora_rank, use_action_mask=False)
            #first offload the model to cpu and half precision
            critic = critic.half().cpu()
            print("load bloom critic (Done) ")
        elif rm_model_name == 'chatglm':
            print("load chatglm critic ", args.pretrain," with lora path ",args.rm_lora_path)
            critic = ChatGLMCritic(pretrained=args.pretrain,lora_path=args.rm_lora_path)
            #first offload the model to cpu and half precision
            critic = critic.half().cpu()
        else:
            raise ValueError(f'Unsupported reward model "{rm_model_name}"')
        #print current cuda memory usage
        print(f'Step 3. critic model is loaded. Current cuda memory usage: {torch.cuda.memory_allocated()/1024/1024} MB')


    # configure optimizer
    if args.strategy.startswith('colossalai'):
        actor_optim = HybridAdam(actor.parameters(), lr=1e-7)
        critic_optim = HybridAdam(critic.parameters(), lr=1e-7)
    else:
        actor_optim = Adam(actor.parameters(), lr=1e-7)
        critic_optim = Adam(critic.parameters(), lr=1e-7)

    # configure tokenizer
    if args.model == 'bloom':
        tokenizer = BloomTokenizerFast.from_pretrained(args.pretrain)
    elif args.model == 'chatglm':
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain, trust_remote_code=True)
    else:
        raise ValueError(f'Unsupported model "{args.model}"')
    



    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    prompt_dataset = EasyPromptsDataset(args.prompt_path, tokenizer)
    if dist.is_initialized() and dist.get_world_size() > 1:
        prompt_sampler = DistributedSampler(prompt_dataset, shuffle=True, seed=42, drop_last=True)
    else:
        prompt_sampler = None
    prompt_dataloader = DataLoader(prompt_dataset,
                                   shuffle=(prompt_sampler is None),
                                   sampler=prompt_sampler,
                                   batch_size=args.train_batch_size)

    pretrain_dataset = EasySupervisedDataset(args.pretrain_dataset, tokenizer)
    if dist.is_initialized() and dist.get_world_size() > 1:
        pretrain_sampler = DistributedSampler(pretrain_dataset, shuffle=True, seed=42, drop_last=True)
    else:
        pretrain_sampler = None
    pretrain_dataloader = DataLoader(pretrain_dataset,
                                     shuffle=(pretrain_sampler is None),
                                     sampler=pretrain_sampler,
                                     batch_size=args.ptx_batch_size,
                                     collate_fn=data_collator)

    def tokenize_fn(texts):
        # MUST padding to max length to ensure inputs of all ranks have the same length
        # Different length may lead to hang when using gemini, as different generation steps
        batch = tokenizer(texts, return_tensors='pt', max_length=96, padding='max_length', truncation=True)
        return {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}

    (actor, actor_optim), (critic, critic_optim) = strategy.prepare((actor, actor_optim), (critic, critic_optim))

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
        max_epochs=args.max_epochs,
        train_batch_size=args.train_batch_size,
        experience_batch_size=args.experience_batch_size,
        tokenizer=tokenize_fn,
        max_length=256,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    trainer.fit(prompt_dataloader=prompt_dataloader,
                pretrain_dataloader=pretrain_dataloader,
                num_episodes=args.num_episodes,
                max_timesteps=args.max_timesteps,
                update_timesteps=args.update_timesteps)

    # save model checkpoint after fitting
    trainer.save_model(args.save_path, only_rank0=True, tokenizer=tokenizer)
    # save optimizer checkpoint on all ranks
    if args.need_optim_ckpt:
        strategy.save_optimizer(actor_optim,
                                'actor_optim_checkpoint_prompts_%d.pt' % (torch.cuda.current_device()),
                                only_rank0=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_path', type=str, default=None, help='path to the prompt dataset')
    parser.add_argument('--pretrain_dataset', type=str, default=None, help='path to the pretrained dataset')
    parser.add_argument('--strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2'],
                        default='naive',
                        help='strategy to use')
    parser.add_argument('--model', default='gpt2', choices=['bloom', 'chatglm'])
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--sft_lora_path', type=str, default=None)
    parser.add_argument('--rm_lora_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='actor_checkpoint_prompts')
    parser.add_argument('--need_optim_ckpt', type=bool, default=False)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--max_timesteps', type=int, default=10)
    parser.add_argument('--update_timesteps', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--ptx_batch_size', type=int, default=1)
    parser.add_argument('--experience_batch_size', type=int, default=2)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument('--kl_coef', type=float, default=0.1)
    parser.add_argument('--ptx_coef', type=float, default=0.9)
    args = parser.parse_args()
    main(args)
