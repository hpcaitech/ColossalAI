import argparse
import os

import loralib as lora
import torch
import torch.distributed as dist
from coati.dataset import DataCollatorForSupervisedDataset, SFTDataset, SupervisedDataset
from coati.models.base import RewardModel
from coati.models.bloom import BLOOMLM
from coati.models.gpt import GPTLM
from coati.models.llama import LlamaLM
from coati.models.opt import OPTLM
from coati.trainer import SFTTrainer
from coati.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from coati.utils import prepare_llama_tokenizer_and_embedding
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, BloomTokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.tensor import ColoParameter


def train(args):
    # configure strategy
    if args.strategy == 'naive':
        strategy = NaiveStrategy()
    elif args.strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cuda')
    elif args.strategy == 'colossalai_zero2':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cuda')
    elif args.strategy == 'colossalai_zero2_cpu':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cpu')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    # configure model
    with strategy.model_init_context():
        if args.model == 'bloom':
            model = BLOOMLM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'opt':
            model = OPTLM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'gpt2':
            model = GPTLM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'llama':
            model = LlamaLM(pretrained=args.pretrain, lora_rank=args.lora_rank,
                            checkpoint=True).to(torch.float16).to(torch.cuda.current_device())
        else:
            raise ValueError(f'Unsupported model "{args.model}"')

    # configure tokenizer
    if args.model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'bloom':
        tokenizer = BloomTokenizerFast.from_pretrained(args.pretrain)
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    elif args.model == 'llama':
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrain,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.eos_token = '<\s>'
    else:
        raise ValueError(f'Unsupported model "{args.model}"')
    tokenizer.pad_token = tokenizer.eos_token
    max_len = args.max_len
    if args.model == 'llama':
        tokenizer = prepare_llama_tokenizer_and_embedding(tokenizer, model)

        if args.strategy == 'colossalai_gemini':
            # this is a hack to deal with the resized embedding
            # to make sure all parameters are ColoParameter for Colossal-AI Gemini Compatiblity
            for name, param in model.named_parameters():
                if not isinstance(param, ColoParameter):
                    sub_module_name = '.'.join(name.split('.')[:-1])
                    weight_name = name.split('.')[-1]
                    sub_module = model.get_submodule(sub_module_name)
                    setattr(sub_module, weight_name, ColoParameter(param))
    else:
        tokenizer.pad_token = tokenizer.eos_token

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

        train_dataset = SFTDataset(train_data, tokenizer, max_len)
        eval_dataset = SFTDataset(eval_data, tokenizer, max_len)

    else:
        train_dataset = SupervisedDataset(tokenizer=tokenizer,
                                          data_path=args.dataset,
                                          max_datasets_size=args.max_datasets_size,
                                          max_length=max_len)
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

    trainer = SFTTrainer(model=model,
                         strategy=strategy,
                         optim=optim,
                         train_dataloader=train_dataloader,
                         eval_dataloader=eval_dataloader,
                         batch_size=args.batch_size,
                         max_epochs=args.max_epochs,
                         accimulation_steps=args.accimulation_steps)

    trainer.fit(logger=logger, log_interval=args.log_interval)

    # save model checkpoint after fitting on only rank0
    trainer.save_model(path=args.save_path, only_rank0=True, tokenizer=tokenizer)
    # save optimizer checkpoint on all ranks
    if args.need_optim_ckpt:
        strategy.save_optimizer(trainer.optimizer,
                                'rm_optim_checkpoint_%d.pt' % (torch.cuda.current_device()),
                                only_rank0=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2', 'colossalai_zero2_cpu'],
                        default='naive')
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
    parser.add_argument('--accimulation_steps', type=int, default=8)
    args = parser.parse_args()
    train(args)
