import argparse
import math
import os

import loralib as lora
import torch
import torch.distributed as dist
from coati.dataset import DataCollatorForSupervisedDataset, SFTDataset, SupervisedDataset
from coati.models import convert_to_lora_module
from coati.trainer import SFTTrainer
from coati.trainer.strategies import DDPStrategy, GeminiStrategy, LowLevelZeroStrategy
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, BloomConfig, BloomForCausalLM, BloomTokenizerFast, LlamaConfig, LlamaForCausalLM
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.trainer import get_scheduler

from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.tensor import ColoParameter


def train(args):
    # configure strategy
    if args.strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        raise NotImplementedError(
            'Gemini is not supported .from_pretrained() yet. We will update this after checkpoint io is ready.')
        strategy = GeminiStrategy(placement_policy='cuda')
    elif args.strategy == 'colossalai_zero2':
        strategy = LowLevelZeroStrategy(stage=2, placement_policy='cuda')
    elif args.strategy == 'colossalai_zero2_cpu':
        strategy = LowLevelZeroStrategy(stage=2, placement_policy='cpu')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    # configure model
    with strategy.model_init_context():
        if args.model == 'bloom':
            model = convert_to_lora_module(BloomForCausalLM.from_pretrained(args.pretrain),
                                           args.lora_rank).half().cuda()
        elif args.model == 'opt':
            model = convert_to_lora_module(OPTForCausalLM.from_pretrained(args.pretrain), args.lora_rank).half().cuda()
        elif args.model == 'gpt2':
            model = convert_to_lora_module(GPT2LMHeadModel.from_pretrained(args.pretrain), args.lora_rank).half().cuda()
        elif args.model == 'llama':
            model = convert_to_lora_module(LlamaForCausalLM.from_pretrained(args.pretrain),
                                           args.lora_rank).half().cuda()
        else:
            raise ValueError(f'Unsupported model "{args.model}"')
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

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
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrain,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.eos_token = '<\s>'
        tokenizer.pad_token = tokenizer.unk_token
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    if args.model == 'llama' and args.strategy == 'colossalai_gemini':
        # this is a hack to deal with the resized embedding
        # to make sure all parameters are ColoParameter for Colossal-AI Gemini Compatibility
        for name, param in model.named_parameters():
            if not isinstance(param, ColoParameter):
                sub_module_name = '.'.join(name.split('.')[:-1])
                weight_name = name.split('.')[-1]
                sub_module = model.get_submodule(sub_module_name)
                setattr(sub_module, weight_name, ColoParameter(param))

    # configure optimizer
    if args.strategy.startswith('colossalai'):
        optim = HybridAdam(model.parameters(), lr=args.lr, clipping_norm=1.0)
    else:
        optim = Adam(model.parameters(), lr=args.lr)

    logger = get_dist_logger()

    # configure dataset
    if args.dataset == 'yizhongw/self_instruct':
        train_data = load_dataset(args.dataset, 'super_natural_instructions', split='train')
        eval_data = load_dataset(args.dataset, 'super_natural_instructions', split='test')

        train_dataset = SFTDataset(train_data, tokenizer, args.max_len)
        eval_dataset = SFTDataset(eval_data, tokenizer, args.max_len)

    else:
        train_dataset = SupervisedDataset(tokenizer=tokenizer,
                                          data_path=args.dataset,
                                          max_datasets_size=args.max_datasets_size,
                                          max_length=args.max_len)
        eval_dataset = None
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    if dist.is_initialized() and dist.get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset,
                                           shuffle=True,
                                           seed=42,
                                           drop_last=True,
                                           rank=dist.get_rank(),
                                           num_replicas=dist.get_world_size())
        if eval_dataset is not None:
            eval_sampler = DistributedSampler(eval_dataset,
                                              shuffle=False,
                                              seed=42,
                                              drop_last=False,
                                              rank=dist.get_rank(),
                                              num_replicas=dist.get_world_size())
    else:
        train_sampler = None
        eval_sampler = None

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=(train_sampler is None),
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=data_collator,
                                  pin_memory=True)
    if eval_dataset is not None:
        eval_dataloader = DataLoader(eval_dataset,
                                     shuffle=(eval_sampler is None),
                                     sampler=eval_sampler,
                                     batch_size=args.batch_size,
                                     collate_fn=data_collator,
                                     pin_memory=True)
    else:
        eval_dataloader = None

    num_update_steps_per_epoch = len(train_dataloader) // args.accumulation_steps
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)
    lr_scheduler = get_scheduler("cosine",
                                 optim,
                                 num_warmup_steps=math.ceil(max_steps * 0.03),
                                 num_training_steps=max_steps)
    strategy_dict = strategy.prepare(
        dict(model=model, optimizer=optim, lr_scheduler=lr_scheduler)
    )
    model = strategy_dict['model']
    optim = strategy_dict['optimizer']
    lr_scheduler = strategy_dict['lr_scheduler']
    trainer = SFTTrainer(model=model,
                         strategy=strategy,
                         optim=optim,
                         lr_scheduler=lr_scheduler,
                         max_epochs=args.max_epochs,
                         accumulation_steps=args.accumulation_steps)

    trainer.fit(train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                logger=logger,
                use_wandb=args.use_wandb)

    # save model checkpoint after fitting on only rank0
    strategy.save_pretrained(model, path=args.save_path, only_rank0=True, tokenizer=tokenizer)
    # save optimizer checkpoint on all ranks
    if args.need_optim_ckpt:
        strategy.save_optimizer(trainer.optimizer,
                                'rm_optim_checkpoint_%d.pt' % (torch.cuda.current_device()),
                                only_rank0=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy',
                        choices=['ddp', 'colossalai_gemini', 'colossalai_zero2', 'colossalai_zero2_cpu'],
                        default='colossalai_zero2')
    parser.add_argument('--model', choices=['gpt2', 'bloom', 'opt', 'llama'], default='bloom')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--max_datasets_size', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='output')
    parser.add_argument('--need_optim_ckpt', type=bool, default=False)
    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument('--log_interval', type=int, default=100, help="how many steps to log")
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--use_wandb', default=False, action='store_true')
    parser.add_argument('--grad_checkpoint', default=False, action='store_true')
    args = parser.parse_args()
    train(args)
