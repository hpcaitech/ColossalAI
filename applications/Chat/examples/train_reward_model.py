import argparse
from random import randint

import torch
import torch.distributed as dist
from coati.dataset import HhRlhfDataset, RmStaticDataset
from coati.models import LogExpLoss, LogSigLoss
from coati.models.bloom import BLOOMRM
from coati.models.gpt import GPTRM
from coati.models.llama import LlamaRM
from coati.models.opt import OPTRM
from coati.trainer import RewardModelTrainer
from coati.trainer.strategies import DDPStrategy, GeminiStrategy, LowLevelZeroStrategy
from datasets import load_dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, BloomTokenizerFast, LlamaTokenizer
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

from colossalai.nn.optimizer import HybridAdam


def train(args):
    # configure strategy
    if args.strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        strategy = GeminiStrategy(placement_policy='cuda')
    elif args.strategy == 'colossalai_zero2':
        strategy = LowLevelZeroStrategy(stage=2, placement_policy='cuda')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    # configure model
    with strategy.model_init_context():
        if args.model == 'bloom':
            model = BLOOMRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'opt':
            model = OPTRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'gpt2':
            model = GPTRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'llama':
            model = LlamaRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        else:
            raise ValueError(f'Unsupported model "{args.model}"')

        if args.model_path is not None:
            state_dict = torch.load(args.model_path)
            model.load_state_dict(state_dict)

    model = model.to(torch.float16)

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
        tokenizer.pad_token = tokenizer.unk_token
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    # configure optimizer
    if args.strategy.startswith('colossalai'):
        optim = HybridAdam(model.parameters(), lr=5e-6)
    else:
        optim = Adam(model.parameters(), lr=5e-6)

    # configure loss function
    if args.loss_fn == 'log_sig':
        loss_fn = LogSigLoss()
    elif args.loss_fn == 'log_exp':
        loss_fn = LogExpLoss()
    else:
        raise ValueError(f'Unsupported loss function "{args.loss_fn}"')

    # prepare for data and dataset
    if args.subset is not None:
        data = load_dataset(args.dataset, data_dir=args.subset)
    else:
        data = load_dataset(args.dataset)

    if args.test:
        train_data = data['train'].select(range(100))
        eval_data = data['test'].select(range(10))
    else:
        train_data = data['train']
        eval_data = data['test']
    valid_data = data['test'].select((randint(0, len(eval_data) - 1) for _ in range(len(eval_data) // 5)))

    if args.dataset == 'Dahoas/rm-static':
        train_dataset = RmStaticDataset(train_data, tokenizer, args.max_len)
        valid_dataset = RmStaticDataset(valid_data, tokenizer, args.max_len)
        eval_dataset = RmStaticDataset(eval_data, tokenizer, args.max_len)
    elif args.dataset == 'Anthropic/hh-rlhf':
        train_dataset = HhRlhfDataset(train_data, tokenizer, args.max_len)
        valid_dataset = HhRlhfDataset(valid_data, tokenizer, args.max_len)
        eval_dataset = HhRlhfDataset(eval_data, tokenizer, args.max_len)
    else:
        raise ValueError(f'Unsupported dataset "{args.dataset}"')

    if dist.is_initialized() and dist.get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset,
                                           shuffle=True,
                                           seed=42,
                                           drop_last=True,
                                           rank=dist.get_rank(),
                                           num_replicas=dist.get_world_size())
        valid_sampler = DistributedSampler(valid_dataset,
                                           shuffle=True,
                                           seed=42,
                                           drop_last=True,
                                           rank=dist.get_rank(),
                                           num_replicas=dist.get_world_size())
        eval_sampler = DistributedSampler(eval_dataset,
                                          shuffle=True,
                                          seed=42,
                                          drop_last=True,
                                          rank=dist.get_rank(),
                                          num_replicas=dist.get_world_size())
    else:
        train_sampler = None
        valid_sampler = None
        eval_sampler = None

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=(train_sampler is None),
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  pin_memory=True)

    valid_dataloader = DataLoader(valid_dataset,
                                  shuffle=(valid_sampler is None),
                                  sampler=valid_sampler,
                                  batch_size=args.batch_size,
                                  pin_memory=True)

    eval_dataloader = DataLoader(eval_dataset,
                                 shuffle=(eval_sampler is None),
                                 sampler=eval_sampler,
                                 batch_size=args.batch_size,
                                 pin_memory=True)

    lr_scheduler = CosineAnnealingLR(optim, train_dataloader.__len__() // 100)
    strategy_dict = strategy.prepare(
        dict(model=model, optimizer=optim, lr_scheduler=lr_scheduler)
    )
    model = strategy_dict['model']
    optim = strategy_dict['optimizer']
    lr_scheduler = strategy_dict['lr_scheduler']
    trainer = RewardModelTrainer(model=model,
                                 strategy=strategy,
                                 optim=optim,
                                 lr_scheduler=lr_scheduler,
                                 loss_fn=loss_fn,
                                 max_epochs=args.max_epochs)

    trainer.fit(train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                eval_dataloader=eval_dataloader)
    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, args.save_path, only_rank0=True)
    # save optimizer checkpoint on all ranks
    if args.need_optim_ckpt:
        strategy.save_optimizer(trainer.optimizer,
                                'rm_optim_checkpoint_%d.pt' % (torch.cuda.current_device()),
                                only_rank0=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy',
                        choices=['ddp', 'colossalai_gemini', 'colossalai_zero2'],
                        default='colossalai_zero2')
    parser.add_argument('--model', choices=['gpt2', 'bloom', 'opt', 'llama'], default='bloom')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--need_optim_ckpt', type=bool, default=False)
    parser.add_argument('--dataset',
                        type=str,
                        choices=['Anthropic/hh-rlhf', 'Dahoas/rm-static'],
                        default='Dahoas/rm-static')
    parser.add_argument('--subset', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='rm_ckpt')
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument('--loss_fn', type=str, default='log_sig', choices=['log_sig', 'log_exp'])
    parser.add_argument('--test', type=bool, default=False)
    args = parser.parse_args()
    train(args)
