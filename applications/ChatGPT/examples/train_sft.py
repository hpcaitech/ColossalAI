import argparse

import loralib as lora
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from chatgpt.dataset import SFTDataset, AlpacaDataset, AlpacaDataCollator
from chatgpt.models.base import RewardModel
from chatgpt.models.bloom import BLOOMLM
from chatgpt.models.gpt import GPTLM
from chatgpt.models.opt import OPTLM
from chatgpt.models.llama import LlamaLM
from chatgpt.trainer import SFTTrainer
from chatgpt.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from chatgpt.utils import prepare_llama_tokenizer_and_embedding
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BloomTokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

from colossalai.nn.optimizer import HybridAdam
from colossalai.logging import get_dist_logger


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
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    # configure model
    with strategy.model_init_context():
        if args.model == 'bloom':
            model = BLOOMLM(pretrained=args.pretrain, lora_rank=args.lora_rank).cuda()
        elif args.model == 'opt':
            model = OPTLM(pretrained=args.pretrain, lora_rank=args.lora_rank).cuda()
        elif args.model == 'gpt2':
            model = GPTLM(pretrained=args.pretrain, lora_rank=args.lora_rank).cuda()
        elif args.model == 'llama':
            model = LlamaLM(pretrained=args.pretrain, lora_rank=args.lora_rank).cuda()
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
    else:
        raise ValueError(f'Unsupported model "{args.model}"')
    
    if args.model == 'llama':
        tokenizer = prepare_llama_tokenizer_and_embedding(tokenizer, model)
    else:
        tokenizer.pad_token = tokenizer.eos_token

    max_len = 512

    # configure optimizer
    if args.strategy.startswith('colossalai'):
        optim = HybridAdam(model.parameters(), lr=5e-5)
    else:
        optim = Adam(model.parameters(), lr=5e-5)

    logger = get_dist_logger()

    # configure dataset
    if args.dataset == 'yizhongw/self_instruct':
        train_data = load_dataset(args.dataset, 'super_natural_instructions', split='train')
        eval_data = load_dataset(args.dataset, 'super_natural_instructions', split='test')

        train_dataset = SFTDataset(train_data, tokenizer, max_len)
        eval_dataset = SFTDataset(eval_data, tokenizer, max_len)

    elif 'alpaca' in args.dataset:
        train_dataset = AlpacaDataset(tokenizer=tokenizer, data_path=args.dataset)
        eval_dataset = None
        data_collator = AlpacaDataCollator(tokenizer=tokenizer)

    if dist.is_initialized() and dist.get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=42, drop_last=True)
        if eval_dataset is not None:
            eval_sampler = DistributedSampler(eval_dataset, shuffle=False, seed=42, drop_last=False)
    else:
        train_sampler = None
        eval_sampler = None

    train_dataloader = DataLoader(train_dataset, shuffle=(train_sampler is None), sampler=train_sampler, batch_size=args.batch_size, collate_fn=data_collator)
    if eval_dataset is not None:
        eval_dataloader = DataLoader(eval_dataset, shuffle=(eval_sampler is None), sampler=eval_sampler, batch_size=args.batch_size, collate_fn=data_collator)
    else:
        eval_dataloader = None

    trainer = SFTTrainer(model=model,
                         strategy=strategy,
                         optim=optim,
                         train_dataloader=train_dataloader,
                         eval_dataloader=eval_dataloader,
                         batch_size=args.batch_size,
                         max_epochs=args.max_epochs)

    trainer.fit(logger=logger, use_lora=args.lora_rank, log_interval=args.log_interval)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, 'sft_checkpoint.pt', only_rank0=True)
    # save optimizer checkpoint on all ranks
    strategy.save_optimizer(optim, 'sft_optim_checkpoint_%d.pt' % (torch.cuda.current_device()), only_rank0=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2'],
                        default='naive')
    parser.add_argument('--model', choices=['gpt2', 'bloom', 'opt', 'llama'], default='bloom')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='yizhongw/self_instruct')
    parser.add_argument('--save_path', type=str, default='sft_ckpt.pth')
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument('--log_interval', type=int, default=100, help="how many steps to log")
    args = parser.parse_args()
    train(args)

